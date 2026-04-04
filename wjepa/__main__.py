"""
W-JEPA entry point.

Usage:
    # Normal training
    python -m wjepa --config config/config_base.yaml

    # Smoke-test train.py logic with dummy data (no GPU / no dataset needed)
    python -m wjepa --test
    python -m wjepa --test --config config/config_base.yaml   # override model settings
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="W-JEPA Pre-training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--test", action="store_true",
                        help="Run smoke-test of train.py logic with dummy data")
    parser.add_argument("--gpu", type=str, default="0",
                        help="CUDA device index")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        if config:
            config.setdefault("meta", {})["gpu"] = args.gpu
            if args.resume:
                config.setdefault("meta", {})["load_checkpoint"] = True

    if args.test:
        # --test: exercise the real train.py code path with dummy data
        from wjepa.train import main_test
        main_test(config)
    else:
        if config is None:
            print("Error: --config is required for training.")
            print("  Training:  python -m wjepa --config config/config_base.yaml")
            print("  Smoke test: python -m wjepa --test")
            sys.exit(1)
        from wjepa.train import main as train_main
        train_main(config, resume_preempt=args.resume)


if __name__ == "__main__":
    main()
