import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.regularizers import L1L2
from keras.models import Sequential


class Model:
    def __init__(self,
                 vocab_size,
                 embed_size,
                 maxlen,
                 lstm_units=250,
                 regularizers=False,
                 l1=1e-5,
                 l2=1e-4):
        self.model = Sequential()
        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.regularizers = regularizers
        self.maxlen = maxlen
        self.l1 = l1
        self.l2 = l2

    def get_model(self):
        self.model.add(Embedding(self.vocab_size,
                       self.embed_size, input_shape=(self.maxlen,)))
        if self.regularizers == True:
            self.model.add(LSTM(self.lstm_units, activation="tanh",
                                kernel_regularizer=L1L2(l1=self.l1, l2=self.l2)))
        else:
            self.model.add(LSTM(self.lstm_units, activation="tanh"))
        self.model.add(Dense(1))
        return self.model

    def model_summary(self):
        return self.model.summary()
