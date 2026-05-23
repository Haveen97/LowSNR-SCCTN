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
Full Model Architecture for LowSNR-SCCTN.
Contains all custom layers, stems, blocks, and model builder functions.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- Precision Policy ----------
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32")

# ---------- Utility Functions ----------

def _safe_eps(x, f16_eps=1e-4, f32_eps=1e-12):
    """Return a dtype-safe epsilon for numerically stable divisions/rsqrt."""
    return tf.constant(f16_eps if x.dtype == tf.float16 else f32_eps, dtype=x.dtype)


def snr_embed(snr, d=32, name="snr_mlp"):
    """Map scalar SNR value to a learnable embedding vector."""
    x = layers.Dense(32, activation="relu", name=name+"_d1")(snr)
    x = layers.Dense(d, activation="relu", name=name+"_d2")(x)
    return x


def bn32(name):
    """BatchNorm that computes in float32 for numerical stability."""
    return layers.BatchNormalization(momentum=0.99, epsilon=1e-4, dtype="float32", name=name)


def cast_back(x):
    """Cast BN (float32) output back to policy compute dtype."""
    return tf.cast(x, tf.keras.mixed_precision.global_policy().compute_dtype)


# ---------- Custom Layers ----------

@keras.utils.register_keras_serializable(package="rf")
class ComplexConv1DNR(layers.Layer):
    """
    Complex 1D convolution on IQ + SNR-conditioned soft-threshold shrinkage on magnitude.
    Input:  x_iq: (B, T, 2)   cond: (B, D) or None
    Output: (B, T, 2*filters)
    """
    def __init__(self, filters, kernel_size=7, strides=1, padding="same", cond_dim=32, **kw):
        super().__init__(**kw)
        self.filters = int(filters)
        self.conv_r = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
        self.conv_i = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
        self.bn = bn32(self.name + "_bn")
        self.th_mlp = keras.Sequential([
            layers.Dense(filters, activation="relu"),
            layers.Dense(filters, activation="softplus")
        ], name=self.name+"_thr")

    def call(self, x_iq, snr_cond=None, training=None):
        I, Q = x_iq[..., 0:1], x_iq[..., 1:2]
        real = self.conv_r(I) - self.conv_i(Q)
        imag = self.conv_r(Q) + self.conv_i(I)
        x = tf.concat([real, imag], axis=-1)

        x = self.bn(x, training=training)
        x = cast_back(x)
        x = keras.activations.relu(x)

        C = x.shape[-1] // 2
        xr, xi = x[..., :C], x[..., C:]
        mag = tf.sqrt(tf.maximum(xr*xr + xi*xi, _safe_eps(xr)))

        if snr_cond is None:
            thr = tf.reshape(tf.cast(0.05, x.dtype), [1, 1, 1])
            scale = tf.nn.relu(mag - thr) / (mag + _safe_eps(mag))
        else:
            t = self.th_mlp(snr_cond)
            t = tf.expand_dims(tf.cast(t, x.dtype), 1)
            scale = tf.nn.relu(mag - t) / (mag + _safe_eps(mag))

        xr = xr * scale
        xi = xi * scale
        return tf.concat([xr, xi], axis=-1)


@keras.utils.register_keras_serializable(package="rf")
class NoiseAdaptiveSE1D(layers.Layer):
    """SE with trimmed-mean squeeze and SNR-temperature."""
    def __init__(self, channels, reduction=8, cond_dim=32, trim=0.1, **kw):
        super().__init__(**kw)
        self.channels = int(channels)
        self.reduction = int(reduction)
        self.trim = float(trim)
        self.fc1 = layers.Dense(self.channels // self.reduction, activation="relu")
        self.fc2 = layers.Dense(self.channels, activation=None)
        self.temp_mlp = keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="softplus")
        ], name=self.name+"_temp")

    @staticmethod
    def trimmed_mean(x, trim=0.1, axis=1):
        k = tf.shape(x)[axis]
        lo = tf.cast(tf.round(trim * tf.cast(k, tf.float32)), tf.int32)
        hi = k - lo
        x_sorted = tf.sort(x, axis=axis)
        idx = tf.range(lo, hi)
        x_trim = tf.gather(x_sorted, idx, axis=axis)
        return tf.reduce_mean(x_trim, axis=axis)

    def call(self, x, snr_cond=None):
        xm = self.trimmed_mean(x, trim=self.trim, axis=1)
        z = self.fc1(xm)
        z = self.fc2(z)
        if snr_cond is not None:
            tau = self.temp_mlp(snr_cond) + tf.cast(0.5, z.dtype)
            tau = tf.maximum(tau, _safe_eps(z))
            z = z / tau
        g = tf.nn.sigmoid(z)
        return x * g[:, None, :]


@keras.utils.register_keras_serializable(package="rf")
class NoiseConditionedFiLM(layers.Layer):
    """FiLM with bounded gain for stability."""
    def __init__(self, channels, cond_dim, max_gain=0.5, **kw):
        super().__init__(**kw)
        self.channels = int(channels)
        self.max_gain = float(max_gain)
        self.gamma = layers.Dense(self.channels)
        self.beta  = layers.Dense(self.channels)

    def call(self, x, cond):
        g = tf.tanh(self.gamma(cond)) * self.max_gain
        b = self.beta(cond)
        g = g[:, None, :]
        b = b[:, None, :]
        return x * (1.0 + g) + b


@keras.utils.register_keras_serializable(package="rf")
class NoiseAwareTransformer(layers.Layer):
    """Transformer block preceded by a depthwise denoise conv; temp scaled by SNR."""
    def __init__(self, num_heads=4, key_dim=32, ff_dim=256, rate=0.1, **kw):
        super().__init__(**kw)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=rate, name=self.name+"_mha")
        self.ln1 = layers.LayerNormalization(epsilon=1e-5, name=self.name+"_ln1")
        self.ff1 = layers.Dense(ff_dim, activation="relu", name=self.name+"_ff1")
        self.drop = layers.Dropout(rate)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5, name=self.name+"_ln2")
        self._ff2 = None
        self._dw  = None
        self.temp_mlp = keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="softplus")
        ], name=self.name+"_temp")

    def build(self, input_shape):
        C = input_shape[-1]
        self._ff2 = layers.Dense(C, name=self.name+"_ff2")
        self._dw  = layers.SeparableConv1D(filters=C, kernel_size=5, padding="same", name=self.name+"_dw")

    def call(self, x, snr_cond=None, training=None):
        y = self._dw(x)
        if snr_cond is None:
            tau = tf.cast(1.0, y.dtype)
        else:
            tau = self.temp_mlp(snr_cond) + tf.cast(0.5, y.dtype)
            tau = tf.maximum(tau, _safe_eps(y))
        scale = tf.math.rsqrt(tau)
        y = y * scale[:, None, :]

        attn = self.mha(y, y, training=training)
        x = self.ln1(x + attn)
        ff = self.ff1(x)
        ff = self.drop(ff, training=training)
        ff = self._ff2(ff)
        x = self.ln2(x + ff)
        return x


# ---------- Stems / Blocks ----------

@keras.utils.register_keras_serializable(package="rf")
class LowSNRStem(layers.Layer):
    def __init__(self, out_channels=64, cond_dim=32, **kw):
        super().__init__(**kw)
        self.cc   = ComplexConv1DNR(filters=out_channels//2, kernel_size=7, cond_dim=cond_dim, name=self.name+"_cc")
        self.bn   = bn32(self.name+"_bn")
        self.se   = NoiseAdaptiveSE1D(out_channels, reduction=8, cond_dim=cond_dim, trim=0.1, name=self.name+"_se")
        self.film = NoiseConditionedFiLM(out_channels, cond_dim=cond_dim, max_gain=0.5, name=self.name+"_film")
        self.tr   = NoiseAwareTransformer(num_heads=4, key_dim=32, ff_dim=2*out_channels, rate=0.1, name=self.name+"_tr")

    def call(self, iq, cond, training=None):
        x = self.cc(iq, snr_cond=cond, training=training)
        x = self.bn(x, training=training)
        x = cast_back(x)
        x = keras.activations.relu(x)
        x = self.se(x, cond)
        x = self.film(x, cond)
        x = self.tr(x, cond, training=training)
        return x


@keras.utils.register_keras_serializable(package="rf")
class LowSNRBlockFeat(layers.Layer):
    def __init__(self, out_channels=128, cond_dim=32, **kw):
        super().__init__(**kw)
        self.proj = layers.Conv1D(out_channels, 1, padding="same", name=self.name+"_proj")
        self.bn   = bn32(self.name+"_bn")
        self.se   = NoiseAdaptiveSE1D(out_channels, reduction=8, cond_dim=cond_dim, trim=0.1, name=self.name+"_se")
        self.film = NoiseConditionedFiLM(out_channels, cond_dim=cond_dim, max_gain=0.5, name=self.name+"_film")
        self.dw1  = layers.DepthwiseConv1D(kernel_size=5, dilation_rate=2, padding="same", name=self.name+"_dw1")
        self.pw1  = layers.Conv1D(out_channels, 1, padding="same", name=self.name+"_pw1")
        self.tr   = NoiseAwareTransformer(num_heads=4, key_dim=32, ff_dim=2*out_channels, rate=0.1, name=self.name+"_tr")

    def call(self, x, cond, training=None):
        y = self.proj(x)
        y = self.bn(y, training=training)
        y = cast_back(y)
        y = keras.activations.relu(y)
        y = self.se(y, cond)
        y = self.film(y, cond)
        r = self.dw1(y)
        r = self.pw1(r)
        y = y + r
        y = self.tr(y, cond, training=training)
        return y


# ---------- Model Builders ----------

def build_low_snr_scctn(input_len=128, num_classes=11, cond_dim=32):
    """
    Build the full LowSNR-SCCTN model with SNR conditioning.
    
    Args:
        input_len: Number of time steps (default 128)
        num_classes: Number of modulation classes (11 for 10a, 10 for 10b)
        cond_dim: SNR embedding dimension (default 32)
    
    Returns:
        keras.Model with inputs [iq, snr] and outputs [mod_out, snr_out]
    """
    iq_in  = keras.Input(shape=(input_len, 2), name="iq")
    snr_in = keras.Input(shape=(1,),           name="snr")
    cond   = snr_embed(snr_in, d=cond_dim, name="cond")

    x = LowSNRStem(out_channels=64,  cond_dim=cond_dim, name="stem")(iq_in, cond)
    x = LowSNRBlockFeat(out_channels=128, cond_dim=cond_dim, name="blk1")(x, cond)
    x = LowSNRBlockFeat(out_channels=128, cond_dim=cond_dim, name="blk2")(x, cond)

    gap = layers.GlobalAveragePooling1D(name="gap")(x)

    h = layers.Dense(256, activation="relu")(gap)
    h = layers.Dropout(0.2)(h)
    mod_out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="mod_out")(h)

    s = layers.Dense(64, activation="relu")(gap)
    snr_out = layers.Dense(1, dtype="float32", name="snr_out")(s)

    return keras.Model([iq_in, snr_in], [mod_out, snr_out], name="LowSNR_SCCTN")


def build_iq_only_scctn(input_len=128, num_classes=11, cond_dim=32):
    """
    Build the I/Q-only variant of LowSNR-SCCTN without SNR conditioning.
    All noise-adaptive layers use default thresholds instead of SNR-conditioned ones.
    
    Args:
        input_len: Number of time steps (default 128)
        num_classes: Number of modulation classes (11 for 10a, 10 for 10b)
        cond_dim: Dummy cond_dim for layer compatibility
    
    Returns:
        keras.Model with inputs [iq, snr] and outputs [mod_out, snr_out]
    """
    iq_in  = keras.Input(shape=(input_len, 2), name="iq")
    snr_in = keras.Input(shape=(1,),           name="snr")

    x = LowSNRStem(out_channels=64,  cond_dim=cond_dim, name="stem")(iq_in, cond=None)
    x = LowSNRBlockFeat(out_channels=128, cond_dim=cond_dim, name="blk1")(x, cond=None)
    x = LowSNRBlockFeat(out_channels=128, cond_dim=cond_dim, name="blk2")(x, cond=None)

    gap = layers.GlobalAveragePooling1D(name="gap")(x)

    h = layers.Dense(256, activation="relu")(gap)
    h = layers.Dropout(0.2)(h)
    mod_out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="mod_out")(h)

    s = layers.Dense(64, activation="relu")(gap)
    snr_out = layers.Dense(1, dtype="float32", name="snr_out")(s)

    return keras.Model([iq_in, snr_in], [mod_out, snr_out], name="LowSNR_SCCTN_IQonly")
