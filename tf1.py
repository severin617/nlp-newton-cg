import argparse
import math

import numpy as np
import pandas as pd

# Make NumPy printouts easier to read.
from cg_scheduler import NewtonLRScheduler
from clr_callback import CyclicLR, SGDRScheduler

np.set_printoptions(precision=3, suppress=True)
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # Enable tf v1 behavior as in v2 a lot have changed
# import dl_utils as utils
import newton_cg as es

print(tf.__version__)

parser = argparse.ArgumentParser(description='Keras  regression case',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--optimizer', type=int, default=0, help='0 for ncg, 1 for adam, 2 for sgd')
parser.add_argument('--scheduler', type=int, default=0, help='1 for exponential, 2 for time-based, 3 for step-based, '
                                                             '0 for constant')
parser.add_argument('--lr', type=float, default=0.01, help='')
parser.add_argument('--epoch', type=int, default=50, help='')

args = parser.parse_args()

data = pd.read_csv("birth_rate.csv")
data.head()


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


# Split data/labels
data_X = np.array(data['Birth rate'])
# data_X = np.array(range(10)) * 0.1 - 0.5
# np.array(data['Birth rate'])
# data_Y = sigmoid(2 * data_X + 0.2)
# np.array(data['Life expectancy'])
data_Y = np.array(data['Life expectancy'])

# print(data_X)
# print(data_Y)

# print(len(data))

# X = tf.placeholder(tf.float32, name='X')
# Y = tf.placeholder(tf.float32, name='Y')

# w = tf.get_variable('weight', initializer=tf.constant(0.1))
# b = tf.get_variable('bias', initializer=tf.constant(0.1))

# Y_hat = tf.add(tf.multiply(X,w), b)

# loss = (Y-Y_hat)*(Y-Y_hat) # tf.keras.losses.mean_squared_error(Y,Y_hat)
# loss = huber_loss(Y,Y_hat)


# model = keras.Model(inputs=[X], outputs=Y, name=f'model{i}')
# model = tf.keras.Sequential([
#    X,
#    layers.Dense(units=1)
# ])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.build((None, 1))
print(model.get_weights())

if args.optimizer < 1:
    model.compile(optimizer=es.EHNewtonOptimizer(0.001), loss='mse')
elif args.optimizer < 2:
    model.compile(optimizer=tf.train.AdamOptimizer(1), loss='mse')
else:
    model.compile(optimizer=tf.train.GradientDescentOptimizer(1), loss='mse')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

model.summary()
model.evaluate(data_X, data_Y)

# for manually changing lr
# for i in range(10):
#     print(model.get_weights())
#     # adapt lr
#     model.fit(data_X, data_Y,  epochs=1, callbacks=[lr_exp])

callbacks_array = []

if args.scheduler == 1:
    callbacks_array.append(es.exp_lr)
elif args.scheduler == 2:
    callbacks_array.append(es.time_lr)
elif args.scheduler == 3:
    callbacks_array.append(es.step_lr)
else:
    callbacks_array.append(es.const_lr)
    print("Picked constant lr")


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr - 0.01


nlr = NewtonLRScheduler(start_lr=1.0, drop=0.5, epochs_drop=10.0,
                        verbose=1, constant_epochs=5, schedule=scheduler)

clr_triangular = CyclicLR(mode='triangular2', base_lr=0.01, max_lr=1.0, step_size=26.)

lrc = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)

warm_clr = SGDRScheduler(min_lr=0.001, max_lr=1.0, steps_per_epoch=26, lr_decay=0.8, cycle_length=5, mult_factor=1.5)

# 190 training samples, 19 batch size => 10 Iteration per epoch, 30 epochs => 300 total iterations

# history = model.fit(data_X, data_Y, callbacks=[warm_clr], epochs=args.epoch, batch_size=19, verbose=1)

# h = clr_triangular.history
# lr = h['lr']
# iters = h['iterations']
#
# dict = {'lr': lr, 'iterations': iters}
#
df = pd.DataFrame(warm_clr.history)

# saving the dataframe
df.to_csv('warm.csv', index=False)

# losses = history.history['loss']
#
# print(lr)
# print(iters)
# print(losses)
# print(len(losses))

print(model.trainable_variables)

print("Weigths:")
print(model.get_weights())
