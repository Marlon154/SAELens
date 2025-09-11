from dataclasses import dataclass

import numpy as np
import torch
import wandb
from jaxtyping import Float
from numpy.typing import NDArray
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
)


@dataclass
class MPSAEConfig(SAEConfig):
    """
    Configuration class for a MPSAE.
    """

    k: int = 32
    dropout: float = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return "mp"


class MPSAE(SAE[MPSAEConfig]):
    """
    MPSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a simple linear encoder and decoder.

    It implements the required abstract methods from BaseSAE:
      - initialize_weights: sets up simple parameter initializations for W_enc, b_enc, W_dec, and b_dec.
      - encode: computes the feature activations from an input.
      - decode: reconstructs the input from the feature activations.

    The BaseSAE.forward() method automatically calls encode and decode,
    including any error-term processing if configured.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: MPSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_standard(self)

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Encode the input tensor into the feature space.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)
        # Compute the pre-activation values
        zhat = torch.zeros(*sae_in.shape[:-1], self.W_enc.shape[1], device=self.device, dtype=sae_in.dtype)
        zhat_ = torch.zeros_like(zhat, device=self.device, dtype=zhat.dtype)
        r = sae_in.clone() - self.b_dec

        for _ in range(self.cfg.k):
            values, indices = torch.max(torch.relu(r @ self.W_enc), dim=-1, keepdim=True)

            zhat_.zero_()
            zhat_.scatter_(-1, indices, values.to(zhat_.dtype))
            zhat += zhat_
            r = r - (zhat @ self.W_dec)
        # no self.hook_sae_acts_pre as no activation function may add in the loop?

        return self.hook_sae_acts_post(zhat)

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
class MPTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a StandardTrainingSAE.
    """

    l1_coefficient: float = 1.0
    lp_norm: float = 1.0
    l1_warm_up_steps: int = 0
    k: int = 32
    dropout: float = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return "mp"


class MPTrainingSAE(TrainingSAE[MPTrainingSAEConfig]):
    """
    MPTrainingSAE is a concrete implementation of BaseTrainingSAE using the "standard" SAE architecture.
    It implements:
      - initialize_weights: basic weight initialization for encoder/decoder.
      - encode: inference encoding (invokes encode_with_hidden_pre).
      - decode: a simple linear decoder.
      - encode_with_hidden_pre: computes activations and pre-activations.
      - calculate_aux_loss: computes a sparsity penalty based on the (optionally scaled) p-norm of feature activations.
    """

    b_enc: nn.Parameter

    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_standard(self)

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "l1": TrainCoefficientConfig(
                value=self.cfg.l1_coefficient,
                warm_up_steps=self.cfg.l1_warm_up_steps,
            ),
        }

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        # Process the input (including dtype conversion, hook call, and any activation normalization)
        sae_in = self.process_sae_in(x)
        # Compute the pre-activation (and allow for a hook if desired)
        zhat = torch.zeros(*sae_in.shape[:-1], self.W_enc.shape[1], device=self.device, dtype=sae_in.dtype)
        zhat_ = torch.zeros_like(zhat, device=self.device, dtype=zhat.dtype)
        r = sae_in.clone() - self.b_dec
        # torch.bernoulli(torch.full((self.W_enc.shape[1],), 0.8))
        dropout = nn.Dropout(p=0.75)

        for i in range(self.cfg.k):
            hidden_pre = torch.relu(r @ self.W_enc)
            # hidden_pre = dropout(hidden_pre)
            values, indices = torch.max(hidden_pre, dim=-1, keepdim=True)
            zhat[indices] += values
            r = r - (values @ self.W_dec[indices])
            wandb.log({f"iter_{i}": values.mean(),
                       f"iter_{i}_min": values.min(),
                       f"iter_{i}_max": values.max(),
            })
        # no self.hook_sae_acts_pre as no activation function may add in the loop?
        #       hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)  # type: ignore
        feature_acts = self.hook_sae_acts_post(zhat)

        # Apply the activation function (and any post-activation hook)
        # feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, feature_acts

    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Forward pass during training."""
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

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
        total_loss = mse_loss

        # Create losses dictionary with mse_loss
        losses = {"mse_loss": mse_loss}

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

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {}
        # # The "standard" auxiliary loss is a sparsity penalty on the feature activations
        # weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
        #
        # # Compute the p-norm (set by cfg.lp_norm) over the feature dimension
        # sparsity = weighted_feature_acts.norm(p=self.cfg.lp_norm, dim=-1)
        # l1_loss = (step_input.coefficients["l1"] * sparsity).mean()
        #
        # return {"l1_loss": l1_loss}

    def log_histograms(self) -> dict[str, NDArray[np.generic]]:
        """Log histograms of the weights and biases."""
        b_e_dist = self.b_enc.detach().float().cpu().numpy()
        return {
            **super().log_histograms(),
            "weights/b_e": b_e_dist,
        }


def _init_weights_standard(
    sae: SAE[MPSAEConfig] | TrainingSAE[MPTrainingSAEConfig],
) -> None:
    sae.b_enc = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )
