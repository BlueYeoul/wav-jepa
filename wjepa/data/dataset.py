"""
Audio dataset for W-JEPA pre-training.
"""

import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """
    Audio dataset class.
    
    Args:
        data_path: Path to audio data directory
        seq_len: Sequence length
        patch_size: Patch size
    """
    
    def __init__(self, data_path, seq_len=1024, patch_size=16):
        super().__init__()
        self.data_path = data_path
        self.seq_len = seq_len
        self.patch_size = patch_size
    
    def __len__(self):
        return len(self._get_audio_files())
    
    def __getitem__(self, idx):
        audio_file = self._get_audio_files()[idx]
        audio = self._load_audio(audio_file)
        
        # Apply patches
        patches = self._apply_patches(audio)
        
        return {"audio": audio, "patches": patches}
    
    def _get_audio_files(self):
        """Get list of audio files."""
        import os
        audio_files = []
        for ext in ["*.wav", "*.flac", "*.mp3"]:
            audio_files.extend(f"{self.data_path}/{ext}")
        return audio_files
    
    def _load_audio(self, path):
        """Load audio file as tensor."""
        # Placeholder - replace with librosa or torchaudio
        return torch.randn(self.seq_len)
    
    def _apply_patches(self, audio):
        """Apply patching to audio."""
        n_patches = self.seq_len // self.patch_size
        return audio.view(-1, self.patch_size)
    
    def inverse_apply_patches(self, patches):
        """Reconstruct audio from patches."""
        return patches.view(-1)


def collate_fn(batch):
    """
    Collate function for audio batches.
    
    Args:
        batch: List of audio samples
        
    Returns:
        batched_tensors: Collated batch tensors
    """
    audio_list = [item["audio"] for item in batch]
    
    # Pad to max length
    max_len = max(len(a) for a in audio_list)
    audio_padded = [
        torch.nn.functional.pad(a, (0, max_len - len(a)))
        for a in audio_list
    ]
    
    return torch.stack(audio_padded)


__all__ = ["AudioDataset", "collate_fn"]
