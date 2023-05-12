import tensorflow as tf


class ExponentialDecayCustom(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            initial_learning_rate,
            decay_steps,
            decay_rate,
            staircase=False,
            name=None):
        super(ExponentialDecayCustom, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "ExponentialDecayCustom") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_rate = tf.cast(self.decay_rate, dtype)

            global_step_recomp = tf.cast(step, dtype)
            p = global_step_recomp / decay_steps
            print("ExponentialDecayCustom CALLED")
            if self.staircase:
                p = tf.floor(p)
            return tf.multiply(
                initial_learning_rate, tf.pow(decay_rate, p), name=name)
