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
Class- and SNR-balanced sampling with phase rotation augmentation.
"""

import numpy as np
import tensorflow as tf


def make_balanced_class_snr_dataset(
    X, Y, SNR_vec, batch=512, num_classes=None, augment=True, shuffle=True
):
    """
    Creates a balanced dataset where each batch contains an equal number
    of samples from each (modulation class, SNR bucket) pair.
    
    SNR buckets: <= -10 dB, (-10, 0] dB, (0, 6] dB, > 6 dB
    
    Args:
        X: I/Q samples (N, T, 2)
        Y: Labels as integer (N,) or one-hot (N, K)
        SNR_vec: SNR values (N,) or (N, 1)
        batch: Batch size
        num_classes: Number of modulation classes
        augment: Whether to apply phase rotation augmentation
        shuffle: Whether to shuffle indices within each batch
    
    Returns:
        tf.data.Dataset yielding ((iq, snr), (mod_out, snr_out)) tuples
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y)
    SNR_vec = np.asarray(SNR_vec, dtype=np.float32)

    # Unify labels
    if Y.ndim == 1:
        y_int = Y.astype(np.int64)
        if num_classes is None:
            num_classes = int(np.max(y_int)) + 1
        Y_oh = tf.keras.utils.to_categorical(y_int, num_classes).astype(np.float32)
    elif Y.ndim == 2:
        Y_oh = Y.astype(np.float32)
        if num_classes is None:
            num_classes = Y_oh.shape[1]
        y_int = np.argmax(Y_oh, axis=1).astype(np.int64)
    else:
        raise ValueError("Y must be shape (N,) or (N,K)")

    # Unify SNR shape
    if SNR_vec.ndim == 1:
        SNR_vec = SNR_vec[:, None]
    elif SNR_vec.ndim == 2 and SNR_vec.shape[1] == 1:
        pass
    else:
        raise ValueError("SNR_vec must be shape (N,) or (N,1)")
    SNR_vec = np.clip(SNR_vec, -40.0, 40.0).astype(np.float32)

    # SNR bucketing
    def snr_bucket(v):
        if v <= -10:
            return 0
        elif v <= 0:
            return 1
        elif v <= 6:
            return 2
        else:
            return 3

    buckets = {(c, b): [] for c in range(num_classes) for b in range(4)}
    for i, (c, s) in enumerate(zip(y_int, SNR_vec.squeeze())):
        buckets[(int(c), snr_bucket(float(s)))].append(i)

    per_cell = max(1, batch // (num_classes * 4))
    cells = list(buckets.keys())
    rng = np.random.RandomState(123)

    def gen():
        while True:
            idxs = []
            for cell in cells:
                pool = buckets[cell]
                if not pool:
                    pool = rng.randint(0, len(X), size=1).tolist()
                idxs += list(rng.choice(pool, size=per_cell, replace=True))
            if shuffle:
                rng.shuffle(idxs)
            idxs = np.asarray(idxs[:batch], dtype=np.int64)
            xb = X[idxs]
            yb = Y_oh[idxs]
            sb = SNR_vec[idxs]
            yield xb, yb, sb

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((None, X.shape[1], X.shape[2]), tf.float32),
            tf.TensorSpec((None, num_classes), tf.float32),
            tf.TensorSpec((None, 1), tf.float32),
        ),
    )

    # Phase rotation augmentation
    def phase_rotate(x):
        th = tf.random.uniform((), -np.pi, np.pi, dtype=x.dtype)
        c, s = tf.cos(th), tf.sin(th)
        I, Q = x[..., 0], x[..., 1]
        return tf.stack([c*I - s*Q, s*I + c*Q], axis=-1)

    def map_aug(x, y, s):
        if augment:
            mask = tf.random.uniform((tf.shape(x)[0],), 0, 1) < 0.5
            x = tf.where(mask[:, None, None], tf.map_fn(phase_rotate, x), x)
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        s = tf.clip_by_value(s, -40.0, 40.0)
        return ({"iq": x, "snr": s}, {"mod_out": y, "snr_out": s})

    return ds.map(map_aug, num_parallel_calls=tf.data.AUTOTUNE)\
             .prefetch(tf.data.AUTOTUNE)\
             .repeat()
