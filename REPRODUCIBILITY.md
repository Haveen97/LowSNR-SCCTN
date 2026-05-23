# Reproducibility Guide

This document provides step-by-step instructions for reproducing the reported values in Tables 2 and 3 of the manuscript.

---

## Table 3: Overall Accuracy Comparison

### Full LowSNR-SCCTN Model (I/Q + SNR dB)

| Dataset | Reported Accuracy | Command |
|---------|------------------|---------|
| RadioML2016.10a | 68.59% ± 0.28% | `python train.py --dataset RML2016.10a --use_snr True --seeds 42,123,456,789,1024` |
| RadioML2016.10b | 67.99% ± 0.33% | `python train.py --dataset RML2016.10b --use_snr True --seeds 42,123,456,789,1024` |

**Steps:**
1. Download `RML2016.10a_dict.pkl` and `RML2016.10b_dict.pkl` from [DeepSig](https://www.deepsig.ai/datasets)
2. Place the files in the `data/` directory
3. Run the training script for each dataset with all five seeds
4. The script outputs the mean accuracy and standard deviation across seeds

### I/Q-Only Variant

| Dataset | Reported Accuracy | Command |
|---------|------------------|---------|
| RadioML2016.10a | 66.88% ± 0.31% | `python train.py --dataset RML2016.10a --use_snr False --seeds 42,123,456,789,1024` |
| RadioML2016.10b | 65.67% ± 0.35% | `python train.py --dataset RML2016.10b --use_snr False --seeds 42,123,456,789,1024` |

**Steps:**
1. Same as above, but set `--use_snr False` to disable SNR conditioning

### Baseline Methods

The baseline results in Table 2 are reproduced directly from the values reported in the respective original publications:
- TADCNN [31]: 66.64% (10a)
- RLITNN [32]: 63.84% (10a), 65.32% (10b)
- ICRNNA [11]: 63.24% (10a), 65.39% (10b)
- CNN-BiLSTM-DNN [8]: 62.73% (10a), 64.76% (10b)
- MCLDNN [41]: 61.91% (10a)
- (etc.)

Refer to the original papers for their experimental protocols.

---

## Table 2: Per-SNR Accuracy Comparison

### Our Model (Full and I/Q-Only)

| Output File | Content |
|-------------|---------|
| `results/per_snr_10a_full.csv` | Per-SNR accuracy for full model on 10a |
| `results/per_snr_10a_iqonly.csv` | Per-SNR accuracy for I/Q-only variant on 10a |
| `results/per_snr_10b_full.csv` | Per-SNR accuracy for full model on 10b |
| `results/per_snr_10b_iqonly.csv` | Per-SNR accuracy for I/Q-only variant on 10b |

**Steps:**
1. After training, run evaluation: `python evaluate.py --checkpoint best.keras --dataset RML2016.10a`
2. The script outputs per-SNR accuracy and saves it to CSV
3. CSV files contain two columns: `SNR` (dB) and `Accuracy` (float)

### Baseline Methods (Literature-Reported)

The per-SNR accuracy values for baseline methods in Table 3 are reproduced from the respective original publications:
- ICRNNA [11]: Per-SNR values for 10b from the original paper
- TADCNN [31]: Per-SNR values for 10a from the original paper
- CNN-BiLSTM-DNN [8]: Per-SNR values for both datasets from the original paper
- MCLDNN [41]: Per-SNR values for both datasets from the original paper

Empty entries in Table 3 indicate that the corresponding method did not report accuracy at that SNR level.

---

## Data Split Reproduction

The exact train/validation/test split can be reproduced using:

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Step 1: 85% train+val, 15% test
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, te_idx = next(sss1.split(X, Yint))

# Step 2: 80% train, 20% validation (from the 85% portion)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr2_idx, va_idx = next(sss2.split(X[tr_idx], Yint[tr_idx]))
