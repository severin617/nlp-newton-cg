import collections
import logging
import math
import os
import pathlib
import re
import string
import sys
import time
import pickle
import numpy as np
import pandas as pd
#from IPython.display import clear_output

import tensorflow as tf

from clr_callback import CyclicLR
from schedulers import ExponentialDecayCustom

print(tf.__version__)
# from tensorflow import keras
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.text import Tokenizer

import transformer as tr
import data_load
import data_generator as dg
from utilities import set_tf_loglevel, set_optimizers, CustomSchedule
import newton_cg as es

import json
from pprint import pprint

set_tf_loglevel(logging.FATAL)


def loss_function(tar_real_onehot, pred):
    """ TODO
  Args:
    tar_real_onehot: target indices sequence. (batch_size, len_seq)
    pred: output of transformer. (batch_size, len_seq, target_vocab_size)
  Returns:
    output: TODO
  """
    real = tf.argmax(tar_real_onehot, axis=-1)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(tar_real_onehot, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(tar_real_onehot, pred):
    # Exclude the <pad>
    real = tf.argmax(tar_real_onehot, axis=-1)  # reverse one hot
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    pred_argmax = tf.cast(tf.argmax(pred, axis=2), dtype=real.dtype)
    accuracies = tf.equal(real, pred_argmax)
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


if __name__ == '__main__':
    # Step1. Data Preparation
    with open('config.json') as j:
        cfg_data = json.load(j)['DATA_PATH']
    data = data_load.DataLoad(cfg_data)
    print("-" * 30, "data information", "-" * 30)
    tokenizer_enc, tokenizer_dec = data.load_tokenizer()
    embedding_matrix_enc, embedding_matrix_dec, size_of_vocabulary_enc, size_of_vocabulary_dec = data.load_weight_matrix()
    idices_tr_enc, idices_tr_dec, idices_val_enc, idices_val_dec = data.load_samples()

    num_tr = idices_tr_enc.shape[0]  # num of dataset used in training
    num_val = idices_val_enc.shape[0]  # num of dataset used in validation
    tr_inp, tr_tar, tr_tar_inp, tr_tar_real = data.input_generator(idices_tr_enc, idices_tr_dec, num_tr,
                                                                   data_type="training")
    val_inp, val_tar, val_tar_inp, val_tar_real = data.input_generator(idices_val_enc, idices_val_dec, num_val,
                                                                       data_type="validation")

    # Step2. Build the Model
    with open('config.json') as j:
        cfg_model = json.load(j)["MODEL_HYPERPARAMETERS"]
    model_params = []
    for i, hyparams in enumerate(cfg_model[0:1]):
        # kill previous model
        tf.keras.backend.clear_session()
        tf.reset_default_graph()

        num_layers = hyparams["num_layers"]
        num_heads = hyparams["num_heads"]
        rate = hyparams["dropout_rate"]
        dff = hyparams["dff"]
        pe_inp = hyparams["pe_inp"]
        pe_tar = hyparams["pe_tar"]
        d_model = embedding_matrix_enc.shape[1]

        # Step3. Set optimizers
        with open('config.json') as j:
            cfg_optimizers = json.load(j)["OPTIMIZERS"]
        optimizers = set_optimizers(cfg_optimizers, d_model)
        # Step4. Set checkpoints
        with open('config.json') as j:
            cfg_checkpts = json.load(j)['PRE_TRAINED']

        for optimizer_info in optimizers:
            tf.keras.backend.clear_session()
            tf.reset_default_graph()

            transformer = tr.Transformer(num_layers,
                                         d_model,
                                         num_heads,
                                         dff,
                                         size_of_vocabulary_enc,
                                         size_of_vocabulary_dec,
                                         pe_inp,
                                         pe_tar,
                                         embedding_matrix_enc,
                                         tr_inp.shape[1],
                                         embedding_matrix_dec,
                                         tr_tar_inp.shape[1],
                                         rate=rate)

            # Create a Encoder
            en_inps = keras.Input(shape=(None,), name="encoder_inps")
            # Create a Decoder
            de_inps = keras.Input(shape=(None,), name="decoder_inps")
            # noinspection PyCallingNonCallable
            out, _ = transformer(en_inps, de_inps, True)

            model = keras.Model(inputs=[en_inps, de_inps], outputs=out, name=f'model{i}')
            print("-" * 30, "Model Summary", "-" * 30)
            model.summary()

            print('-' * 30, "Start the new optimizer", '-' * 30)
            if optimizer_info[1] == "Newton_CG":
                optimizer, name, lr, tau = optimizer_info
                solver_name = name + '_lr' + str(lr) + '_tau' + str(tau)
                print(f'opt= {name}, lr={lr}, tau={tau}')
            else:
                optimizer, name, lr = optimizer_info
                solver_name = name + '_lr' + str(lr)
                print(f'opt= {name}, lr={lr}')

            # Set checkpoint directory
            model_name = f"samples{num_tr}_{num_layers}layers_{num_heads}heads_{dff}dff"
            checkpoint_dir = os.path.join("checkpoints2", model_name, solver_name)

            print('-' * 30, 'checkpoint file information', '-' * 30)
            if not os.path.exists(checkpoint_dir):
                print(checkpoint_dir, " not found, build a directory......")
                os.makedirs(checkpoint_dir)
                print(checkpoint_dir, 'is built.')
            else:
                print(checkpoint_dir, ' directory exists!')

            if cfg_checkpts["use_pretrained"] and model_name in cfg_checkpts:
                print('Use Adam pretrained model')
                pre_trained_checkpoint_dir = cfg_checkpts[model_name]["pre_trained_checkpoint_dir"]
                if os.path.isfile(pre_trained_checkpoint_dir + '/cp.ckpt.index'):
                    pre_trained_checkpoint_path = os.path.join(pre_trained_checkpoint_dir, 'cp.ckpt')
                    model.load_weights(pre_trained_checkpoint_path)
                    print(pre_trained_checkpoint_path, 'is found!')
                    print('Latest 1st pretrained model checkpoint restored!!')
            elif cfg_checkpts["use_pretrained"] and model_name not in cfg_checkpts:
                print("Use Use Adam pretrained model, but there's no pretrained model, please check it")
                sys.exit()
            else:
                print('Without Adam pretrained model')
                checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')
                if os.path.isfile(checkpoint_dir + '/cp.ckpt.index'):
                    model.load_weights(checkpoint_path)
                    print(checkpoint_path, 'is found!')
                    print('Latest checkpoint restored!!')
                else:
                    last_epoch = 0
                    print("checkpoint not found, training from scratch。")

            checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')
            print("checkpoint_path:", checkpoint_path)
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

            # Create a CSVLogger that saves the logs
            print('-' * 30, '.log file information', '-' * 30)
            if os.path.isfile(os.path.join(checkpoint_dir, 'training_0.log')):
                log_lsts = [_ for _ in os.listdir(checkpoint_dir) if _.endswith('.log')]
                print('.log files exist!')
                print('.log files:', log_lsts)
                # create new filename
                idx = str(len(log_lsts))
                filename = f"training_{idx}.log"
                print('Create next .log file:', filename)
            else:
                print(".log file not found, training from scratch。")
                filename = 'training_0.log'

            logger_path = os.path.join(checkpoint_dir, filename)
            print(logger_path, 'is created!')
            csv_logger = CSVLogger(logger_path)

            # Extract hyper-parameters
            batch_size = cfg_optimizers["batch_size"]
            epochs = cfg_optimizers["epochs"]
            start_lr = optimizer_info[2]
            # decay = start_lr / epochs

            # setup CLR callback
            # clr_fun = lambda x: np.exp(-0.1 * 4 * (x-1))
            #clr_triangular2 = CyclicLR(mode='exp_range', base_lr=0.001, max_lr=1.0, step_size=2224., gamma=0.99994)
            clr_triangular2 = CyclicLR(mode='exp_range', base_lr=lr, max_lr=0.1, step_size=22., gamma=0.99)

            # Setup callbacks
            # es.const_lr
            final_callbacks = [cp_callback, csv_logger]#, clr_triangular2]

            # Step5. Training
            training_generator = dg.DataGenerator(tr_inp, tr_tar_inp, tr_tar_real,
                                                  batch_size=batch_size, n_classes=size_of_vocabulary_enc)
            validation_generator = dg.DataGenerator(val_inp, val_tar_inp, val_tar_real,
                                                    batch_size=96, n_classes=size_of_vocabulary_enc)

            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                  reduction='none')
            model.compile(optimizer=optimizer,
                          loss=loss_function,
                          metrics=[accuracy_function])
            print("FINISHED MODEL COMPILE")
            hist = model.fit_generator(generator=training_generator,
                                       validation_data=validation_generator,
                                       epochs=epochs,
                                       verbose=2,
                                       callbacks=final_callbacks
                                       )

            h = clr_triangular2.history
            lr = h['lr']
            iters = h['iterations']
            acc = h['accuracy_function']
            loss = h['loss']

            dict = {'lr': lr, 'iterations': iters, 'accuracy': acc, 'loss': loss}

            df = pd.DataFrame(dict)

            # saving the dataframe
            df.to_csv('exp_range_second_run.csv', index=False)
