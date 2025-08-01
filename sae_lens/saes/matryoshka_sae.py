# Based : https://github.com/bartbussmann/matryoshka_sae/blob/main/sae.py
from dataclasses import dataclass, field
from typing import Any, Callable, List

import torch
from jaxtyping import Float
from torch import nn
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig, JumpReLUSAEConfig
from sae_lens.saes.sae import (
    SAEConfig,
    TrainStepInput,
    TrainStepOutput
)

class BatchTopK(nn.Module):
    """BatchTopK activation function"""

    def __init__(
        self,
        k: int,
    ):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = x.relu()
        flat_acts = acts.flatten()
        acts_topk_flat = torch.topk(flat_acts, self.k * acts.shape[0], dim=-1)
        return (
            torch.zeros_like(flat_acts)
            .scatter(-1, acts_topk_flat.indices, acts_topk_flat.values)
            .reshape(acts.shape)
        )


@dataclass
class MatryoshkaSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for a MatryoshkaSAE.
    """
    group_sizes: List[int] = field(default_factory=lambda: [128, 512, 2048, 8196])
    k_per_group: List[int] = field(default_factory=lambda: [16, 32, 32, 32])

    @override
    @classmethod
    def architecture(cls) -> str:
        return "matryoshka"

class MatryoshkaSAE(BatchTopKTrainingSAE):
    """todo
    """

    topk_threshold: torch.Tensor

    def __init__(self, cfg: BatchTopKTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

        self.register_buffer(
            "topk_threshold",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return BatchTopK(self.cfg.k)

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Encode the input tensor into the feature space.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)
        # Compute the pre-activation values
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        # Apply the activation function (e.g., ReLU, depending on config)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decode the feature activations back to the input space.
        Now, if hook_z reshaping is turned on, we reverse the flattening.
        """
        # 1) linear transform
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        # 2) hook reconstruction
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        # 4) optional out-normalization (e.g. constant_norm_rescale)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        # 5) if hook_z is enabled, rearrange back to (..., n_heads, d_head).
        return self.reshape_fn_out(sae_out_pre, self.d_head)


@dataclass
class MatryoshkaTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for training a MatryoshkaTrainingSAE.
    """

    group_sizes: List[int] = field(default_factory=lambda: [128, 512, 2048, 8196])

    @override
    @classmethod
    def architecture(cls) -> str:
        return "matryoshka"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return JumpReLUSAEConfig


class MatryoshkaTrainingSAE(BatchTopKTrainingSAE):
    """
    MatryoshkaTrainingSAE
    """

    cfg: MatryoshkaTrainingSAEConfig  # type: ignore[assignment]

    def __init__(self, cfg: MatryoshkaTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.group_indices = [0] + list(torch.cumsum(torch.tensor(cfg.group_sizes), dim=0))

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        """Forward pass during training."""
        #output = super().training_forward_pass(step_input)
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)

        # Matryoshka Core
        intermediate_reconstructs = []
        x_reconstruct = self.b_dec
        # feature_acts = self.apply_finetuning_scaling_factor(feature_acts)

        for i in range(len(self.group_indices) -1):
            start_idx = self.group_indices[i]
            end_idx = self.group_indices[i+1]
            W_dec_slice = self.W_dec[start_idx:end_idx, :]
            acts_topk = feature_acts[:, start_idx:end_idx]
            x_reconstruct = acts_topk @ W_dec_slice + x_reconstruct
            intermediate_reconstructs.append(x_reconstruct)

        total_l2_loss = torch.tensor(0.0, dtype=torch.double, device=step_input.sae_in.device)
        l2_losses = torch.tensor([], device=step_input.sae_in.device)
        for intermediate_reconstruct in intermediate_reconstructs:
            l2_losses = torch.cat([l2_losses, (intermediate_reconstruct.float() -
                                             step_input.sae_in.float()).pow(2).mean().unsqueeze(0)])
            total_l2_loss += (intermediate_reconstruct.float() - step_input.sae_in.float()).pow(2).mean()

        mean_l2_loss = total_l2_loss / (len(intermediate_reconstructs) + 1)

        sae_out = self.hook_sae_recons(x_reconstruct)
        sae_out = self.run_time_activation_norm_fn_out(sae_out)
        sae_out = self.reshape_fn_out(sae_out, self.d_head)
        # Calculate MSE loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # Calculate architecture-specific auxiliary losses
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        # Total loss is MSE plus all auxiliary losses
        total_loss = mean_l2_loss + aux_losses["auxiliary_reconstruction_loss"]

        # Create losses dictionary with mse_loss
        losses = {"mse_loss": mse_loss, "aux_losses": aux_losses["auxiliary_reconstruction_loss"]}
        for n, level_l2_loss in enumerate(l2_losses):
            losses[f"l2_loss_level_{n}"] = level_l2_loss.item()

        # Add architecture-specific losses to the dictionary
        # Make sure aux_losses is a dictionary with string keys and tensor values
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)

        # Sum all losses for total_loss
        if isinstance(aux_losses, dict):
            for loss_value in aux_losses.values():
                total_loss = total_loss + loss_value
        else:
            # Handle case where aux_losses is a tensor
            total_loss = total_loss + aux_losses

        output = TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )

        self.update_topk_threshold(output.feature_acts)
        output.metrics["topk_threshold"] = self.topk_threshold
        return output
