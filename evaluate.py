import numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_te, Y_te, S_te, BATCH=512):
    pred = model.predict({"iq": X_te, "snr": S_te}, batch_size=BATCH, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred_mod = pred[0]
    else:
        pred_mod = pred
    
    y_pred = np.argmax(pred_mod, axis=1)
    
    if Y_te.ndim == 1:
        y_true = Y_te.astype(int)
    elif Y_te.ndim == 2 and Y_te.shape[1] == 1:
        y_true = Y_te.squeeze().astype(int)
    elif Y_te.ndim == 2:
        y_true = np.argmax(Y_te, axis=1)
    else:
        y_true = Y_te.reshape((Y_te.shape[0], -1))
        y_true = np.argmax(Y_te, axis=-1)
    
    acc = (y_pred == y_true).mean()
    print(f"\nTEST Accuracy: {acc*100:.2f}%\n")
    print("Classification report:\n", classification_report(y_true, y_pred, digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    
    # Per-SNR accuracy
    snr_levels = np.unique(S_te.squeeze())
    snr_acc = []
    for s in snr_levels:
        m = np.isclose(S_te.squeeze(), s)
        snr_acc.append((int(s), float((y_pred[m] == y_true[m]).mean())))
    snr_acc = sorted(snr_acc, key=lambda t: t[0])
    print("\nAccuracy vs SNR:")
    for s, a in snr_acc:
        print(f"SNR {s:>3} dB : {a*100:5.2f}%")
    
    # Save per-SNR CSV
    np.savetxt("per_snr_accuracy.csv", snr_acc, delimiter=",", fmt="%d,%.6f", header="SNR,Accuracy")
    
    return acc, snr_acc, cm