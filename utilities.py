import os
import logging
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import newton_cg as es
from schedulers import ExponentialDecayCustom


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def set_optimizers(cfg_optimizers, d_model):
    optimizers = []
    if "Adam" in cfg_optimizers and cfg_optimizers["Adam"]:
        for adam_kwargs in cfg_optimizers["Adam"]:
            if adam_kwargs['learning_rate'] == 'CustomSchedule':
                adam_kwargs.pop("learning_rate")
                opt = tf.keras.optimizers.Adam(learning_rate=CustomSchedule(d_model), **adam_kwargs)
                optimizers.append((opt, "Adam", 'CustomSchedule'))
            else:
                opt = tf.keras.optimizers.Adam(**adam_kwargs)
                optimizers.append((opt, "Adam", adam_kwargs['learning_rate']))

    if "SGD" in cfg_optimizers and cfg_optimizers["SGD"]:
        for sgd_kwargs in cfg_optimizers["SGD"]:
            opt = tf.keras.optimizers.SGD(**sgd_kwargs)
            optimizers.append((opt, "SGD", sgd_kwargs['learning_rate']))

    if "Newton_CG" in cfg_optimizers and cfg_optimizers["Newton_CG"]:
        for cg_kwargs in cfg_optimizers["Newton_CG"]:
            if cg_kwargs['learning_rate'] == 'ExponentialDecayCustom':
                cg_kwargs.pop('learning_rate')
                opt = es.EHNewtonOptimizer(learning_rate=ExponentialDecayCustom(0.001, 4000, 0.96), **cg_kwargs)
                optimizers.append((opt, "Newton_CG", 'ExponentialDecayCustom', cg_kwargs['tau']))
            else:
                opt = es.EHNewtonOptimizer(**cg_kwargs)
                optimizers.append((opt, "Newton_CG", cg_kwargs['learning_rate'], cg_kwargs['tau']))

    print("-" * 30, "Optimizers Information", "-" * 30)
    for optimizer in optimizers:
        if optimizer[1] == "Newton_CG":
            _, name, lr, tau = optimizer
            print(f'opt= {name}, lr={lr}, tau={tau}')
        else:
            _, name, lr = optimizer
            print(f'opt= {name}, lr={lr}')
    return optimizers
