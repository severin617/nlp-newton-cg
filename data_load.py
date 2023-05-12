import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


class DataLoad():
    def __init__(self, cfg):
        self.tokenizer_enc = os.path.join(cfg['data_folder'], cfg['tokenizer_enc'])
        self.tokenizer_dec = os.path.join(cfg['data_folder'], cfg['tokenizer_dec'])
        self.embedding_matrix_enc = os.path.join(cfg['data_folder'], cfg['embedding_matrix_enc'])
        self.embedding_matrix_dec = os.path.join(cfg['data_folder'], cfg['embedding_matrix_dec'])
        self.indices_tr_enc = os.path.join(cfg['data_folder'], cfg['indices_tr_enc'])
        self.indices_tr_dec = os.path.join(cfg['data_folder'], cfg['indices_tr_dec'])
        self.indices_val_enc = os.path.join(cfg['data_folder'], cfg['indices_val_enc'])
        self.indices_val_dec = os.path.join(cfg['data_folder'], cfg['indices_val_dec'])

    def load_tokenizer(self):
        """
        Args:
            tokenizer_en (str): tokenizer of encoder
            tokenizer_de (str): tokenizer of decoder
        Returns:
            A tuple containing two tf.keras.preprocessing.text.Tokenizer
                1st: tokenizer of encoder
                2nd: tokenizer of decoder
        """
        with open(self.tokenizer_enc, 'rb') as handle:
            tokenizer_encoder = pickle.load(handle)
        with open(self.tokenizer_dec, 'rb') as handle:
            tokenizer_decoder = pickle.load(handle)
        print('num of index in encoder =', len(tokenizer_encoder.word_index) + 1)
        print('num of index in decoder =', len(tokenizer_decoder.word_index) + 1)
        return tokenizer_encoder, tokenizer_decoder

    def load_weight_matrix(self):
        """Load pre-trained embedding_matrix for tf.keras.layers.Embedding
        Args:
            embedding_matrix_en (str): .npy file, Pre-trained embedding_matrix of encoder
            embedding_matrix_de (str): .npy file, Pre-trained embedding_matrix of decoder
        Returns:
            A tuple containing two np.array
                1st: embedding_matrix of encoder
                2nd: embedding_matrix of decoder

        """
        embedding_matrix_encoder = np.load(self.embedding_matrix_enc)
        embedding_matrix_decoder = np.load(self.embedding_matrix_dec)
        size_of_vocabulary_encoder = embedding_matrix_encoder.shape[0]
        size_of_vocabulary_decoder = embedding_matrix_decoder.shape[0]
        print('embedding_matrix_encoder.shape =', embedding_matrix_encoder.shape)
        print('embedding_matrix_decoder.shape =', embedding_matrix_decoder.shape)
        print('size_of_vocabulary_encoder =', size_of_vocabulary_encoder)
        print('size_of_vocabulary_decoder =', size_of_vocabulary_decoder)

        return embedding_matrix_encoder, embedding_matrix_decoder, \
               size_of_vocabulary_encoder, size_of_vocabulary_decoder

    def load_samples(self):
        """
        Args:
            indices_tr_en (str): .npy file, training samples in indices form of encoder, input language
            indices_tr_de (str): .npy file, training samples in indices form of decoder, target language
            indices_val_en (str): .npy file, validation samples in indices form of encoder, input language
            indices_val_de (str): .npy file, validation samples in indices form of encoder, target language

        Returns:
            A tuple containing four np.array

        """
        indices_tr_enc = np.load(self.indices_tr_enc)
        indices_tr_dec = np.load(self.indices_tr_dec)
        indices_val_enc = np.load(self.indices_val_enc)
        indices_val_dec = np.load(self.indices_val_dec)
        print('idices_tr_encoder.shape =', indices_tr_enc.shape)
        print('idices_tr_decoder.shape =', indices_tr_dec.shape)
        print('idices_val_encoder.shape =', indices_val_enc.shape)
        print('idices_val_decoder.shape =', indices_val_dec.shape)
        return indices_tr_enc, indices_tr_dec, indices_val_enc, indices_val_dec

    def input_generator(self, idices_enc, idices_dec, num_samples, data_type="training"):
        """

        Args:
            idices_en:
            idices_de:
            num_samples:
            data_type: "training" or "validation"

        Returns:

        """
        num_inp = num_samples  # num of dataset used in training
        inp = idices_enc[:num_inp]
        tar = idices_dec[:num_inp]  # tar source
        tar_inp = tar[:, :-1]  # decoder's input of tar
        tar_real = tar[:, 1:]  # decoder's output of tar
        print(f'{"-" * 30} {data_type} data information {"-" * 30}')
        print("inp.shape:", inp.shape)
        print("-" * 20)
        print("tar.shape:", tar.shape)
        print("-" * 20)
        print("tar_inp.shape:", tar_inp.shape)
        print("-" * 20)
        print("tar_real.shape:", tar_real.shape)
        print("-" * 20)

        return inp, tar, tar_inp, tar_real
