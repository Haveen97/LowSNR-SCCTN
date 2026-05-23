import os, pickle, numpy as np

def load_rml2016_any_format(obj):
    """Load RadioML2016 dataset from various pickle formats."""
    if isinstance(obj, dict) and all(k in obj for k in ["X","Y","SNR"]):
        X = obj["X"]; Y = obj["Y"]; SNR = obj["SNR"]
        return np.asarray(X), np.asarray(Y), np.asarray(SNR)
    if isinstance(obj, dict):
        mods = sorted({k[0] for k in obj.keys()})
        snrs = sorted({k[1] for k in obj.keys()})
        X_list, Y_list, SNR_list = [], [], []
        for m in mods:
            for s in snrs:
                block = obj.get((m, s), None)
                if block is None: continue
                X_list.append(block)
                Y_list.extend([mods.index(m)] * block.shape[0])
                SNR_list.extend([s] * block.shape[0])
        X = np.vstack(X_list).astype("float32")
        Y = np.array(Y_list, dtype="int32")
        SNR = np.array(SNR_list, dtype="float32")
        return X, Y, SNR
    raise ValueError("Unrecognized pickle structure")