# TinyRB: Efficient Implementation of GPT-2 (127M)

## Overview
TinyRB is an optimized implementation of the GPT-2 (127M) model, developed under the EigenCore research initiative. This project focuses on reducing computational time and resource requirements while maintaining competitive performance.

This repository contains the source code and documentation related to TinyRB, including architectural improvements, training optimizations, and performance analysis.

## Features
- **Architectural Optimization**: Based on GPT-2 (124M) with adjustments in pre-training and hyperparameter configurations.
- **Reduced Computational Costs**: Utilizes techniques such as Flash Attention, mixed-precision training, and distributed data parallel (DDP).
- **Curated Pre-Training Dataset**: Uses the edu_fineweb10B dataset, optimized for reasoning and language comprehension tasks.
- **Scalability**: Supports multi-GPU training and configurations optimized for resource-limited hardware.

## Installation
### Prerequisites
To run this project, ensure you have:
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional for GPU acceleration)
- Additional dependencies (installable via `pip`):

```sh
pip install -r requirements.txt
```

## Usage
### Training
To train TinyRB from scratch, run:
```sh
python train.py --config config/tinyrb.yaml
```
<!-- 
### Evaluation
To evaluate the model on benchmarks such as HellaSwag:
```sh
python evaluate.py --model-checkpoint path/to/checkpoint
``` -->

## Optimization Techniques
TinyRB incorporates various optimizations to enhance efficiency:
- **Parameter Sharing**: Reuses weights between the embedding and output layers.
- **Flash Attention**: Reduces the computational cost of the self-attention mechanism.
- **Mixed-Precision Training (bfloat16/float16)**: Speeds up computations and reduces memory consumption.
- **Gradient Clipping**: Prevents training instability.
- **torch.compile Usage**: Improves execution performance by fusing model operations.

## Results
The TinyRB model outperforms GPT-2 (124M) on the HellaSwag benchmark, achieving better performance in text prediction tasks. Full performance details and analysis can be found in `pending..`.

## Future Work
- Implementation of a scaled version of TinyRB following Scaling Laws.
- Evaluation on additional natural language understanding benchmarks.
- Further training optimizations to enhance efficiency and performance.

## Authors
This work has been developed by:
- **M. Galindo** (EigenCore) - max@eigencore.org
- **A. León** (EigenCore) - aleon@eigencore.org

## License
This project is distributed under the APACHE 2.0 License. See `LICENSE` for more information.

---
For more information about this project, refer to the full report in `pending..`.

