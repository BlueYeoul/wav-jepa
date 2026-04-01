"""
Run configuration and argument parsing for W-JEPA.
"""

import argparse
from datetime import datetime
from typing import Any, Dict


def parse_args() -> Dict[str, Any]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="W-JEPA Pre-training")
    
    # ---------------------------------------------------------------------- #
    # Meta Configuration
    # ---------------------------------------------------------------------- #
    parser.add_argument("--folder", type=str, required=True, 
                        help="Experiment folder path")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--read-checkpoint", type=str, default=None,
                        help="Path to checkpoint to read")
    parser.add_argument("--write-checkpoint", type=str, default=None,
                        help="Path to save checkpoints")
    
    # ---------------------------------------------------------------------- #
    # Dataset Configuration
    # ---------------------------------------------------------------------- #
    parser.add_argument("--dataset-type", type=str, default="audio",
                        help="Dataset type (audio, video, etc.)")
    parser.add_argument("--seq-len", type=int, default=16000,
                        help="Sequence length")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Patch size")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loading workers")
    
    # ---------------------------------------------------------------------- #
    # Model Configuration
    # ---------------------------------------------------------------------- #
    parser.add_argument("--model-name", type=str, default="audio_transformer_large",
                        help="Model type")
    parser.add_argument("--in-chans", type=int, default=1,
                        help="Input channels")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length for transformer")
    parser.add_argument("--embed-dim", type=int, default=768,
                        help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=12,
                        help="Transformer depth")
    parser.add_argument("--num-heads", type=int, default=12,
                        help="Number of attention heads")
    
    # ---------------------------------------------------------------------- #
    # Training Configuration
    # ---------------------------------------------------------------------- #
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Warmup epochs")
    parser.add_argument("--clip-grad", type=float, default=0.0,
                        help="Clip gradient norm (0 = no clipping)")
    
    # ---------------------------------------------------------------------- #
    # Mask Configuration (V-JEPA style)
    # ---------------------------------------------------------------------- #
    parser.add_argument("--pred-mask-scale", type=str, default="0.15,0.5",
                        help="Mask scale range (min,max)")
    parser.add_argument("--num-blocks", type=int, default=1,
                        help="Number of mask blocks")
    parser.add_argument("--max-context-ratio", type=float, default=1.0,
                        help="Maximum context ratio")
    parser.add_argument("--max-keep", type=int, default=None,
                        help="Maximum tokens to keep")
    
    # ---------------------------------------------------------------------- #
    # Distributed Training
    # ---------------------------------------------------------------------- #
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA visible devices")
    parser.add_argument("--nodes", type=int, default=1,
                        help="Number of nodes")
    parser.add_argument("--node-rank", type=int, default=0,
                        help="Node rank (for multi-node)")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Total number of nodes")
    
    args = parser.parse_args()
    
    # Convert to dict
    return vars(args)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary."""
    return {
        # Meta
        "folder": "experiments/debug",
        "seed": 0,
        "resume": False,
        # Dataset
        "dataset_type": "audio",
        "seq_len": 16000,
        "patch_size": 16,
        "batch_size": 32,
        "num_workers": 4,
        # Model
        "model_name": "audio_transformer_large",
        "in_chans": 1,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        # Training
        "epochs": 1000,
        "lr": 1e-4,
        "warmup_epochs": 10,
        "clip_grad": 0.0,
        # Mask
        "pred_mask_scale": "0.15,0.5",
        "num_blocks": 1,
        "max_context_ratio": 1.0,
        # Distributed
        "gpu": "0",
        "nodes": 1,
    }


if __name__ == "__main__":
    # Test argument parsing
    config = get_default_config()
    print("Default config:", config)
