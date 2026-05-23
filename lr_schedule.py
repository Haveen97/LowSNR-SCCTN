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
Warmup cosine learning rate schedule.
"""

import math
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="sched")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup followed by cosine decay.
    
    Args:
        base_lr: Peak learning rate after warmup
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
    """
    def __init__(self, base_lr, warmup_steps, total_steps, name=None):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.name = name or "WarmupCosine"

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        base_lr = tf.cast(self.base_lr, tf.float32)
        # Warmup phase
        warm = base_lr * step / tf.maximum(1.0, warmup_steps)
        # Cosine decay phase
        progress = tf.clip_by_value(
            (step - warmup_steps) / tf.maximum(1.0, (total_steps - warmup_steps)),
            0.0, 1.0
        )
        cosv = 0.5 * base_lr * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < warmup_steps, warm, cosv)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "name": self.name,
        }
