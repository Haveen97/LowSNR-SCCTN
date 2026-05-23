import math, tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="sched")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, name=None):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps  = int(total_steps)
        self.name = name or "WarmupCosine"

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps  = tf.cast(self.total_steps, tf.float32)
        base_lr      = tf.cast(self.base_lr, tf.float32)
        warm = base_lr * step / tf.maximum(1.0, warmup_steps)
        progress = tf.clip_by_value((step - warmup_steps) / tf.maximum(1.0, (total_steps - warmup_steps)), 0.0, 1.0)
        cosv = 0.5 * base_lr * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < warmup_steps, warm, cosv)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "name": self.name,
        }