"""
Sampling utilities for W-JEPA pre-training.
"""

import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedSampler(Sampler):
    """
    Distributed sampler for multi-GPU training.
    
    Ensures each GPU gets a unique subset of data.
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__()
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        # Duplicate for even split
        if len(indices) < self.num_replicas:
            indices += indices
        indices = indices[:self.num_replicas * (len(self.dataset) // self.num_replicas)]
        
        if self.shuffle:
            import torch
            g = torch.Generator()
            g.manual_seed(42)  # Placeholder
            indices = torch.randperm(len(indices), generator=g).tolist()
        
        # Sub-sample per rank
        step = len(self.dataset) // self.num_replicas
        start = self.rank * step
        end = start + step
        
        return iter(indices[start:end])
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) // self.num_replicas


__all__ = ["DistributedSampler"]
