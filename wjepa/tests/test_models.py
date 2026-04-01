"""Test suite for W-JEPA models."""

import pytest
import torch
from wjepa.models import encoder, predictor


class TestAudioTransformer:
     """Tests for AudioTransformer encoder."""
    
    @pytest.fixture
    def audio_transformer(self):
        return encoder.AudioTransformer(
            seq_len=1024,
            patch_size=16,
            embed_dim=768,
            depth=12,
        )
    
    def test_forward_shape(self, audio_transformer):
        x = torch.randn(2, 1, 1024)  # (B, C, T)
        output = audio_transformer(x)
        assert output.shape[0] == 2  # Batch size preserved
        
    def test_hierarchical_output(self, audio_transformer):
        x = torch.randn(1, 1, 1024)
        output = audio_transformer(x)
        
        # Should output hierarchical representation
        assert len(output.shape) == 3
   
