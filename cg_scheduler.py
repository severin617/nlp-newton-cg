from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np
import math
import pandas as pd
import os


class NewtonLRScheduler(Callback):
    """Learning rate scheduler.

  Arguments:
      schedule: a function that takes an epoch index as input
          (integer, indexed from 0) and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages.

  ```python
  # This function keeps the learning rate at 0.001 for the first ten epochs
  # and decreases it exponentially after that.
  def scheduler(epoch):
    if epoch < 10:
      return 0.001
    else:
      return 0.001 * tf.math.exp(0.1 * (10 - epoch))

  callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
  model.fit(data, labels, epochs=100, callbacks=[callback],
            validation_data=(val_data, val_labels))
  ```
  """

    def __init__(self, schedule=None, verbose=0, start_lr=1.0, k=0.1, mode="exponential",
                 epochs_drop=10.0, drop=0.5, constant_epochs=0, log_name=None):
        super(NewtonLRScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.start_lr = start_lr
        self.k = k
        self.mode = mode
        self.epochs_drop = epochs_drop
        self.drop = drop
        self.constant_epochs = constant_epochs
        self.log_name = log_name
        self.history = {}

    def lr_scheduler(self, epoch):
        if epoch < self.constant_epochs:
            return self.start_lr

        actual_epoch = epoch - self.constant_epochs

        if self.schedule is not None:
            try:  # Support for new API
                lr = float(K.get_value(self.model.optimizer.lr))
                lr = self.schedule(actual_epoch, lr)
                return lr
            except TypeError:  # Support for old API for backward compatibility
                return self.schedule(actual_epoch)

        elif self.mode == "step":
            return self.step_decay(actual_epoch)
        elif self.mode == "time":
            return self.time_decay(actual_epoch)
        elif self.mode == "exponential":
            return self.exp_decay(actual_epoch)
        else:
            return self.start_lr

    # Time-based scheduler
    def time_decay(self, epoch):
        lr = self.start_lr
        for i in range(0, epoch + 1):
            lr *= 1. / (1. + self.k * i)
        return lr

    # Stepwise scheduler
    def step_decay(self, epoch):
        return self.start_lr * math.pow(self.drop, math.floor(epoch / self.epochs_drop))

    # Exponential scheduler
    def exp_decay(self, epoch):
        return self.start_lr * np.exp(-self.k * epoch)

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        lr = self.lr_scheduler(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('epoch', []).append(epoch)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_train_end(self, logs=None):
        if self.log_name is not None:
            df = pd.DataFrame(self.history)
            df.to_csv(f'{self.log_name}', index=False)
            print(f"Saved logs to {os.getcwd()}/{self.log_name}")
