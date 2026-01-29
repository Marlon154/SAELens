"""Tests for LoggingConfig and wandb logging functionality."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from sae_lens.config import LoggingConfig


class TestLoggingConfigWeightsUpload:
    """Tests for the log_weights_to_wandb configuration option."""

    def test_log_weights_to_wandb_default_is_true(self):
        """Test that log_weights_to_wandb defaults to True to maintain backward compatibility."""
        cfg = LoggingConfig()
        assert cfg.log_weights_to_wandb is True

    def test_log_weights_to_wandb_can_be_set_to_false(self):
        """Test that log_weights_to_wandb can be set to False."""
        cfg = LoggingConfig(log_weights_to_wandb=False)
        assert cfg.log_weights_to_wandb is False

    @patch("sae_lens.config.wandb")
    def test_log_uploads_weights_when_flag_is_true(self, mock_wandb):
        """Test that weights are uploaded when log_weights_to_wandb is True."""
        cfg = LoggingConfig(log_weights_to_wandb=True)

        mock_trainer = Mock()
        mock_trainer.sae.get_name.return_value = "test_sae"
        mock_trainer.cfg.__dict__ = {"test": "config"}

        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        weights_path = Path("/tmp/weights.pt")
        cfg_path = Path("/tmp/cfg.json")
        sparsity_path = Path("/tmp/sparsity.pt")

        cfg.log(mock_trainer, weights_path, cfg_path, sparsity_path)

        # Verify the model artifact was created
        assert mock_wandb.Artifact.call_count == 2  # model + sparsity

        # Verify weights file was added to model artifact
        model_artifact_calls = [
            call for call in mock_artifact.add_file.call_args_list
        ]
        # Should have 2 calls for model artifact (weights + cfg)
        # and 1 call for sparsity artifact
        assert len(model_artifact_calls) == 3
        # First call should be weights, second should be cfg
        assert str(weights_path) in str(model_artifact_calls[0])
        assert str(cfg_path) in str(model_artifact_calls[1])

    @patch("sae_lens.config.wandb")
    def test_log_skips_weights_when_flag_is_false(self, mock_wandb):
        """Test that weights are NOT uploaded when log_weights_to_wandb is False."""
        cfg = LoggingConfig(log_weights_to_wandb=False)

        mock_trainer = Mock()
        mock_trainer.sae.get_name.return_value = "test_sae"
        mock_trainer.cfg.__dict__ = {"test": "config"}

        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        weights_path = Path("/tmp/weights.pt")
        cfg_path = Path("/tmp/cfg.json")
        sparsity_path = Path("/tmp/sparsity.pt")

        cfg.log(mock_trainer, weights_path, cfg_path, sparsity_path)

        # Verify the model artifact was created
        assert mock_wandb.Artifact.call_count == 2  # model + sparsity

        # Verify only cfg file was added to model artifact (not weights)
        model_artifact_calls = [
            call for call in mock_artifact.add_file.call_args_list
        ]
        # Should have 1 call for model artifact (cfg only, no weights)
        # and 1 call for sparsity artifact
        assert len(model_artifact_calls) == 2

        # Verify cfg was added but weights were not
        add_file_args = [str(call) for call in model_artifact_calls]
        assert any(str(cfg_path) in arg for arg in add_file_args)
        assert not any(str(weights_path) in arg for arg in add_file_args)

    @patch("sae_lens.config.wandb")
    def test_log_still_uploads_config_and_sparsity_when_weights_disabled(
        self, mock_wandb
    ):
        """Test that config and sparsity are still uploaded when weights are disabled."""
        cfg = LoggingConfig(log_weights_to_wandb=False)

        mock_trainer = Mock()
        mock_trainer.sae.get_name.return_value = "test_sae"
        mock_trainer.cfg.__dict__ = {"test": "config"}

        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        weights_path = Path("/tmp/weights.pt")
        cfg_path = Path("/tmp/cfg.json")
        sparsity_path = Path("/tmp/sparsity.pt")

        cfg.log(mock_trainer, weights_path, cfg_path, sparsity_path)

        # Verify both artifacts were logged
        assert mock_wandb.log_artifact.call_count == 2

        # Verify the add_file calls
        model_artifact_calls = [
            call for call in mock_artifact.add_file.call_args_list
        ]
        assert len(model_artifact_calls) == 2  # cfg + sparsity
