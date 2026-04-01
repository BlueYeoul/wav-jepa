"""
W-JEPA entry point.

Usage:
    python -m wjepa --folder experiments/debug --epochs 100
"""

import sys
import argparse
from wjepa.train import main as train_main


def main():
    """Entry point for the package."""
    parser = argparse.ArgumentParser(description="W-JEPA Pre-training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file (YAML)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA device")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    if args.config:
        print(f"Config file: {args.config}")
    else:
        print("Usage: python -m wjepa --config <config.yaml>")
        sys.exit(1)
    
    # Import and run training
    try:
        from wjepa.train import main as train_main
        
        # Load YAML config
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Add command line args
        config["gpu"] = args.gpu
        config["resume"] = args.resume
        
        # Run training
        train_main(config)
        
    except ImportError:
        print("Please install required packages: pip install -e .")
        sys.exit(1)


if __name__ == "__main__":
    main()
