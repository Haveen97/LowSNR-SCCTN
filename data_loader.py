# Copyright (c) 2026 Haveen Yaseen Hussein AL-Zahawi  et al.
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
Data loader for RadioML2016.10a and RadioML2016.10b datasets.
Handles various pickle formats.
"""

import numpy as np


def load_rml2016_any_format(obj):
    """
    Load RadioML2016 dataset from various pickle formats.
    
    Supports:
    - Dict with keys X, Y, SNR
    - Dict keyed by (modulation, snr) tuples
    
    Args:
        obj: Unpickled object from RadioML2016 dataset file
    
    Returns:
        X: I/Q samples (N, 2, 128)
        Y: Modulation labels (N,)
        SNR: SNR values (N,)
    """
    # Case A: dict with X, Y, SNR
    if isinstance(obj, dict) and all(k in obj for k in ["X", "Y", "SNR"]):
        X = obj["X"]
        Y = obj["Y"]
        SNR = obj["SNR"]
        return np.asarray(X), np.asarray(Y), np.asarray(SNR)
    
    # Case B: dict keyed by (mod, snr)
    if isinstance(obj, dict):
        mods = sorted({k[0] for k in obj.keys()})
        snrs = sorted({k[1] for k in obj.keys()})
        X_list, Y_list, SNR_list = [], [], []
        for m in mods:
            for s in snrs:
                block = obj.get((m, s), None)
                if block is None:
                    continue
                X_list.append(block)
                Y_list.extend([mods.index(m)] * block.shape[0])
                SNR_list.extend([s] * block.shape[0])
        X = np.vstack(X_list).astype("float32")
        Y = np.array(Y_list, dtype="int32")
        SNR = np.array(SNR_list, dtype="float32")
        return X, Y, SNR
    
    raise ValueError("Unrecognized pickle structure")
