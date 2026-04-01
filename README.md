# W-JEPA (Waveformer JEPA)

V-JEPA 2.1 implementation for audio sequences.

## Overview

This is a pre-training framework based on JEMPA (Joint-Embodied Predictive Architecture) adapted for audio processing. It includes:

- **Encoder**: Hierarchical audio transformer backbone
- **Predictor**: Transformer-based predictor for masked modeling
- **Masking**: Advanced mask generation with distance-weighted losses
- **Training**: Distributed training with EMA targets

## Project Structure

```
wjepa/
├── __main__.py      # Entry point
├── data/            # Data preprocessing
├── loss.py          # Loss functions
├── masks/           # Mask collator & distance
├── models/          # Encoder, Predictor, Wrappers
├── run.py           # Run configuration
├── src/             # Additional source
├── test/            # Test utilities
├── train.py         # Training loop
└── utils.py         # Training utilities
```

## Requirements

- Python >= 3.12
- PyTorch >= 2.10.0

## Installation

```bash
pip install -e .
```

## Usage

```bash
python -m wjepa --config config.yaml
```

Or directly:

```python
from wjepa.train import main

# Configure arguments
args = {
    "folder": "experiments/debug",
    "meta": {...},
    # ...
}

main(args)
```

## License

MIT License
