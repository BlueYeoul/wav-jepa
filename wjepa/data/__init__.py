"""
Data initialization utilities for W-JEPA pre-training.
Supports dynamic input lengths via DynamicMaskCollator1D as collate_fn.
"""

from torch.utils.data import DataLoader

from .dataset import LibriSpeechDatasetFactory, collate_fn
from .sampler import DistributedSampler


def init_data(
    data,
    root_path,
    batch_size,
    training,
    rank,
    world_size,
    datasets_weights=None,
    collator=None,
    num_workers=1,
    pin_mem=False,
    dynamic_config=None,
    **kwargs,   # absorb unused video-era args (fps, dataset_fpcs, etc.)
):
    """
    Initialize data loader with dynamic audio input support.

    Args:
        data          : dataset type string ("AudioDataset" / "DynamicAudioDataset")
        root_path     : str or list[str] – dataset root directory
        batch_size    : samples per batch
        training      : shuffle flag
        rank          : process rank (distributed)
        world_size    : total processes
        collator      : DynamicMaskCollator1D instance used as collate_fn
        num_workers   : DataLoader workers
        pin_mem       : pin_memory flag
        dynamic_config: dict with "dynamic_seq_len" sub-dict for min/max_sec
    """
    dynamic_config = dynamic_config or {}
    dynamic_seq = dynamic_config.get("dynamic_seq_len", {})
    min_sec = dynamic_seq.get("min_seq_len_sec", 1.5)
    max_sec = dynamic_seq.get("max_seq_len_sec", 20.0)

    root = root_path[0] if isinstance(root_path, list) else root_path

    dataset = LibriSpeechDatasetFactory.create(
        root_path=root,
        mode=500,  # or whatever default mode you want
        min_sec=min_sec,
        max_sec=max_sec,
    )

    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler if training else None,
        shuffle=(sampler is None and training),
        collate_fn=collator if collator is not None else collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return loader, sampler


__all__ = ["init_data", "AudioDataset", "collate_fn", "DistributedSampler"]
