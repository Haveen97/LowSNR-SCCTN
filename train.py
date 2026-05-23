import os, pickle, numpy as np, tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from model import build_low_snr_scctn
from data_loader import load_rml2016_any_format
from balanced_sampler import make_balanced_class_snr_dataset
from lr_schedule import WarmupCosine

# ---------------------------
# Configuration
# ---------------------------
DATA_PATH = "/kaggle/input/RML2016.10a_dict.pkl"
BATCH = 512
EPOCHS = 120
RANDOM_SEEDS = [42, 123, 456, 789, 1024]

# ---------------------------
# Load dataset
# ---------------------------
with open(DATA_PATH, "rb") as f:
    raw = pickle.load(f, encoding="latin1")

X, Y, SNR = load_rml2016_any_format(raw)
X = np.transpose(X, (0,2,1)).astype("float32")
NUM_CLASSES = int(np.max(Y) + 1)
Yint = Y.astype("int32")
SNRdB = SNR.astype("float32").reshape(-1,1)

# ---------------------------
# Train/Val/Test split
# ---------------------------
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, te_idx = next(sss1.split(X, Yint))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr2_idx, va_idx = next(sss2.split(X[tr_idx], Yint[tr_idx]))

X_tr, Y_tr, S_tr = X[tr_idx][tr2_idx], Y[tr_idx][tr2_idx], SNRdB[tr_idx][tr2_idx]
X_va, Y_va, S_va = X[tr_idx][va_idx],   Y[tr_idx][va_idx],   SNRdB[tr_idx][va_idx]
X_te, Y_te, S_te = X[te_idx], Y[te_idx], SNRdB[te_idx]

# ---------------------------
# Training loop over seeds
# ---------------------------
for seed in RANDOM_SEEDS:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    train_ds = make_balanced_class_snr_dataset(X_tr, Y_tr, S_tr, batch=BATCH, num_classes=NUM_CLASSES, augment=True)
    val_ds   = make_balanced_class_snr_dataset(X_va, Y_va, S_va, batch=BATCH, num_classes=NUM_CLASSES, augment=False)
    
    model = build_low_snr_scctn(input_len=128, num_classes=NUM_CLASSES, cond_dim=32)
    
    train_steps = max(1, len(X_tr) // BATCH)
    val_steps   = max(1, len(X_va) // BATCH)
    total_steps  = train_steps * EPOCHS
    warmup_steps = train_steps * 3
    
    sched = WarmupCosine(base_lr=1e-3, warmup_steps=warmup_steps, total_steps=total_steps)
    opt = tf.keras.optimizers.Adam(learning_rate=sched, clipnorm=1.0)
    
    model.compile(
        optimizer=opt,
        loss={"mod_out": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
              "snr_out": tf.keras.losses.MeanAbsoluteError()},
        loss_weights={"mod_out": 1.0, "snr_out": 0.05},
        metrics={"mod_out": ["accuracy"]},
    )
    
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mod_out_accuracy", mode="max",
                                              factor=0.5, patience=6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_mod_out_accuracy", mode="max",
                                          patience=30, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f"best_seed{seed}.keras", monitor="val_mod_out_accuracy",
                                            mode="max", save_best_only=True)
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1
    )
    
    print(f"Training completed for seed {seed}")