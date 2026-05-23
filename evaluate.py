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
Evaluation script for LowSNR-SCCTN model.
Computes test accuracy, classification report, confusion matrix, and per-SNR accuracy.
"""

import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_te, Y_te, S_te, batch_size=512, output_csv="per_snr_accuracy.csv"):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained keras model
        X_te: Test I/Q samples (N, 128, 2)
        Y_te: Test labels (N,) or (N, K)
        S_te: Test SNR values (N, 1)
        batch_size: Batch size for prediction
        output_csv: Path to save per-SNR accuracy CSV
    
    Returns:
        test_acc: Overall test accuracy (float)
        snr_acc: List of (SNR, accuracy) tuples
        cm: Confusion matrix (numpy array)
    """
    # Predict
    pred = model.predict({"iq": X_te, "snr": S_te}, batch_size=batch_size, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred_mod = pred[0]
    else:
        pred_mod = pred

    y_pred = np.argmax(pred_mod, axis=1)

    # Handle label format
    if Y_te.ndim == 1:
        y_true = Y_te.astype(int)
    elif Y_te.ndim == 2 and Y_te.shape[1] == 1:
        y_true = Y_te.squeeze().astype(int)
    elif Y_te.ndim == 2:
        y_true = np.argmax(Y_te, axis=1)
    else:
        y_true = Y_te.reshape((Y_te.shape[0], -1))
        y_true = np.argmax(Y_te, axis=-1)

    # Overall accuracy
    test_acc = (y_pred == y_true).mean() * 100
    print(f"\nTEST Accuracy: {test_acc:.2f}%\n")

    # Classification report
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    # Per-SNR accuracy
    snr_levels = np.unique(S_te.squeeze())
    snr_acc = []
    print("\nAccuracy vs SNR:")
    for s in snr_levels:
        m = np.isclose(S_te.squeeze(), s)
        acc = (y_pred[m] == y_true[m]).mean() * 100
        snr_acc.append((int(s), float(acc)))
        print(f"SNR {int(s):>4} dB : {acc:6.2f}%")

    # Sort and save
    snr_acc = sorted(snr_acc, key=lambda t: t[0])
    np.savetxt(output_csv, snr_acc, delimiter=",", fmt="%d,%.6f",
               header="SNR,Accuracy")
    print(f"\nPer-SNR accuracy saved to {output_csv}")

    return test_acc, snr_acc, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LowSNR-SCCTN model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.keras file)")
    parser.add_argument("--dataset", type=str, default="RML2016.10a",
                        help="Dataset name for output labeling")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for evaluation")
    parser.add_argument("--output_csv", type=str, default="per_snr_accuracy.csv",
                        help="Path to save per-SNR accuracy CSV")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = tf.keras.models.load_model(args.checkpoint)

    # Load test data (modify paths as needed)
    import pickle
    from data_loader import load_rml2016_any_format
    from sklearn.model_selection import StratifiedShuffleSplit

    DATA_PATH = "/kaggle/input/RML2016.10a_dict.pkl"
    with open(DATA_PATH, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    X, Y, SNR = load_rml2016_any_format(raw)
    X = np.transpose(X, (0, 2, 1)).astype("float32")
    Yint = Y.astype("int32")
    SNRdB = SNR.astype("float32").reshape(-1, 1)

    # Reproduce the same test split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, te_idx = next(sss1.split(X, Yint))
    X_te, Y_te, S_te = X[te_idx], Yint[te_idx], SNRdB[te_idx]

    # Evaluate
    evaluate_model(model, X_te, Y_te, S_te, batch_size=args.batch_size, output_csv=args.output_csv)
