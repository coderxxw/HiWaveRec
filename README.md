# HiWaveRec: Hierarchical Wavelet Decoupling Framework with Adaptive Gating for Sequential Recommendation

Accepted by **ECML-PKDD 2026**.

## Overview

HiWaveRec is a sequential recommendation model built on top of the RecBole framework, which employs multi-level discrete wavelet transform to decouple long-term and short-term user interests, with a heterogeneous dual-branch architecture (LFSA + CDG) for differentiated modeling.

## Requirements

Python 3.9 or later is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

The project follows the standard RecBole data layout. Datasets are stored in the `dataset/` directory.
The project adopts the RecBole standard data organization method. Datasets is automatically downloaded to the `dataset/` when training.
## Quick Start

### Train

```bash
python run_recbole.py --model HiWave --dataset amazon-beauty
```

```bash
python run_recbole.py --model HiWave --dataset ml-1m
```

If needed, adjust settings in `recbole/properties/model/HiWave.yaml` and `recbole/properties/overall.yaml`.

