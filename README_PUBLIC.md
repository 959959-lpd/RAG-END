# AI Literary Creation Engine

This repository contains the source code for the paper "Poetry Generation Framework Based on RAG and Fact-Enhanced Decoding".

## Directory Structure

- `AI_Literary_Creation_Engine/`: Main package code.
  - `src/`: Source code for model, data processing, and training.
  - `scripts/`: Helper scripts.
- `run_benchmark.py`: Main script to reproduce evaluation results.
- `run_comparison_experiments.sh`: Bash script for running experiments.
- `generate_tsne.py`: Script for visualization.

## Setup

1. Install dependencies:
   ```bash
   pip install -r AI_Literary_Creation_Engine/requirements.txt
   ```
2. Configure Paths:
   Modify `config.py` or environment variables in `run_benchmark.py` to point to your data and model paths.

## Usage

To run the benchmark:
```bash
python run_benchmark.py
```

## Note on Data

Large datasets and model checkpoints are not included in this code-only release. Please refer to the paper for data acquisition details or contact the authors.
