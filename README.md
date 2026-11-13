# EaMRD-CRCV

Research code for evidence-aware multimodal rumor detection and claim/reasoning conflict verification. This is a source-only snapshot for inspection and reproduction; datasets, feature caches, model weights, API credentials, and experiment outputs are not included.

## Contents

- `model/`: multimodal detector, evidence/claim fusion, graph reasoning, and contrastive modules.
- `preprocess/`: frame, OCR/ASR, claim, evidence, reasoning, and feature encoding stages.
- `data_utils/`: dataset adapters and cached-feature loading.
- `train.py`, `evaluate.py`, `run_protocol.py`: training, evaluation, and protocol entry points.
- `tools/`: feature audits and baseline evaluation utilities.

## Setup

Install dependencies with `pip install -r requirements.txt`. Copy `.env.example` to `.env` and supply only the API credentials required by the preprocessing services you use. Download FakeSV/FakeTT from their original providers and prepare the metadata, split files, and feature caches expected by `data_utils/`. The original research workspace uses local absolute paths in some preprocessing and dataset scripts; adjust those paths to your environment before running. The full configurations from the archived E1 runs are in `configs/release/`; their cache names refer to feature caches that are not redistributed here.

Example training commands (after data preparation):

```bash
python train.py --config configs/release/fakesv_full.yaml --seed 123
python train.py --config configs/release/fakett_full.yaml --seed 123
```

## Experimental scope

The archived configurations document selected runs; they do not imply a new or unbiased hyperparameter selection. Reproduction requires the matching splits, preprocessing pipeline, external model/API versions, and cached features. Cross-dataset generalization was weak in exploratory tests and is not claimed as a strength of this code.

