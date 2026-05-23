# Copyright (c) 2026 Haveen Yaseen Hussein AL-Zahawi et al.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Training script for LowSNR-SCCTN model.
Supports both full SNR-conditioned model and I/Q-only ablation.
"""

import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

from model import build_low_snr_scctn, build_iq_only_scctn
from data_loader import load_rml2016_any_format
from balanced_sampler import make_balanced_class_snr_dataset
from lr_schedule import WarmupCosine


# ---------------------------
# Parse Arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Train LowSNR-SCCTN model")
parser.add_argument("--dataset", type=str, default="RML2016.10a",
                    help="Dataset to use: RML2016.10a or RML2016.10b")
parser.add_argument("--data_path", type=str, default="/kaggle/input/RML2016.10a_dict.pkl",
                    help="Path to dataset pickle file")
parser.add_argument("--use_snr", type=lambda x: x.lower() == "true", default=True,
                    help="Whether to use SNR conditioning (True for full model, False for I/Q-only)")
parser.add_argument("--batch_size", type=int, default=512,
                    help="Batch size for training")
parser.add_argument("--epochs", type=int, default=120,
                    help="Number of training epochs")
parser.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                    help="Comma-separated list of random seeds")
parser.add_argument("--output_dir", type=str, default="./checkpoints",
                    help="Directory to save model checkpoints")
args = parser.parse_args()

# ---------------------------
# Configuration
# ---------------------------
DATA_PATH = args.data_path
BATCH = args.batch_size
EPOCHS = args.epochs
USE_SNR = args.use_snr
RANDOM_SEEDS = [int(s) for s in args.seeds.split(",")]

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------
# Load Dataset
# ---------------------------
print(f"Loading dataset from {DATA_PATH}...")
with open(DATA_PATH, "rb") as f:
    raw = pickle.load(f, encoding="latin1")

X, Y, SNR = load_rml2016_any_format(raw)
X = np.transpose(X, (0, 2, 1)).astype("float32")  # (N, 128, 2)
NUM_CLASSES = int(np.max(Y) + 1)
Yint = Y.astype("int32")
SNRdB = SNR.astype("float32").reshape(-1, 1)

print(f"Shapes -> X: {X.shape}, Y: {Yint.shape}, SNR: {SNRdB.shape}")
print(f"Number of classes: {NUM_CLASSES}")

# ---------------------------
# Train/Val/Test Split
# ---------------------------
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, te_idx = next(sss1.split(X, Yint))

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr2_idx, va_idx = next(sss2.split(X[tr_idx], Yint[tr_idx]))

X_tr, Y_tr, S_tr = X[tr_idx][tr2_idx], Y[tr_idx][tr2_idx], SNRdB[tr_idx][tr2_idx]
X_va, Y_va, S_va = X[tr_idx][va_idx], Y[tr_idx][va_idx], SNRdB[tr_idx][va_idx]
X_te, Y_te, S_te = X[te_idx], Y[te_idx], SNRdB[te_idx]

print(f"Train: {X_tr.shape[0]}, Val: {X_va.shape[0]}, Test: {X_te.shape[0]}")

# ---------------------------
# Training Loop Over Seeds
# ---------------------------
test_accuracies = []

for seed in RANDOM_SEEDS:
    print(f"\n{'='*60}")
    print(f"Training with seed {seed} (use_snr={USE_SNR})")
    print(f"{'='*60}")
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Create datasets
    train_ds = make_balanced_class_snr_dataset(
        X_tr, Y_tr, S_tr, batch=BATCH, num_classes=NUM_CLASSES, augment=True
    )
    val_ds = make_balanced_class_snr_dataset(
        X_va, Y_va, S_va, batch=BATCH, num_classes=NUM_CLASSES, augment=False
    )
    
    # Build model
    if USE_SNR:
        model = build_low_snr_scctn(input_len=128, num_classes=NUM_CLASSES, cond_dim=32)
    else:
        model = build_iq_only_scctn(input_len=128, num_classes=NUM_CLASSES, cond_dim=32)
    
    model.summary()
    
    # Training configuration
    train_steps = max(1, len(X_tr) // BATCH)
    val_steps = max(1, len(X_va) // BATCH)
    total_steps = train_steps * EPOCHS
    warmup_steps = train_steps * 3
    
    sched = WarmupCosine(base_lr=1e-3, warmup_steps=warmup_steps, total_steps=total_steps)
    opt = tf.keras.optimizers.Adam(learning_rate=sched, clipnorm=1.0)
    
    model.compile(
        optimizer=opt,
        loss={
            "mod_out": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            "snr_out": tf.keras.losses.MeanAbsoluteError()
        },
        loss_weights={"mod_out": 1.0, "snr_out": 0.05},
        metrics={"mod_out": ["accuracy"]},
    )
    
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mod_out_accuracy", mode="max",
            factor=0.5, patience=6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mod_out_accuracy", mode="max",
            patience=30, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, f"best_seed{seed}.keras"),
            monitor="val_mod_out_accuracy", mode="max", save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set (seed {seed})...")
    pred = model.predict({"iq": X_te, "snr": S_te}, batch_size=BATCH, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred_mod = pred[0]
    else:
        pred_mod = pred
    
    y_pred = np.argmax(pred_mod, axis=1)
    if Y_te.ndim == 1:
        y_true = Y_te.astype(int)
    elif Y_te.ndim == 2:
        y_true = np.argmax(Y_te, axis=1)
    else:
        y_true = Y_te.squeeze().astype(int)
    
    test_acc = (y_pred == y_true).mean() * 100
    test_accuracies.append(test_acc)
    print(f"TEST Accuracy (seed {seed}): {test_acc:.2f}%")

# ---------------------------
# Summary
# ---------------------------
print(f"\n{'='*60}")
print(f"SUMMARY: use_snr={USE_SNR}")
print(f"{'='*60}")
for s, a in zip(RANDOM_SEEDS, test_accuracies):
    print(f"Seed {s}: {a:.2f}%")
print(f"Mean: {np.mean(test_accuracies):.2f}%")
print(f"Std:  {np.std(test_accuracies):.2f}%")
