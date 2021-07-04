from keras.optimizer_v2 import adam
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import adam_v2
from Vocab import Vocab
from sklearn.model_selection import train_test_split
from Model import Model
import matplotlib.pyplot as plt


# Vocab Object
vocab = Vocab(frequency=2)


# Reading the data
df_list = pd.read_csv("Data/archive/IMDB Dataset.csv").values.tolist()
vocab.clean_sentences(df_list)
vocab.build_vocab(df_list)
(x, y) = vocab.numericalize_sentences(df_list)
x = vocab.pad_num_sentences(x)
# Now X has padded sentences list and Y has sentiments

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Model parameters
EPOCHS = 10
BATCH_SIZE = 32
vocab_size = len(vocab)
embed_size = 128
lstm_units = 100
lr = 3e-4
maxlen = 100
l1 = 0.0001
l2 = 0.0002


model = Model(vocab_size=vocab_size,
              embed_size=embed_size,
              maxlen=250,
              regularizers=True,
              l1=l1,
              l2=l2).get_model()

model.compile(optimizer=adam_v2.Adam(learning_rate=lr),
              loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                    epochs=EPOCHS, validation_data=(x_test, y_test))

# Visualizing the loss
plt.plot(history.history["accuracy"])
plt.show()
plt.plot(history.history["loss"])
plt.show()

test_sent = "I really liked the movie it was so awesome it was so good"
test_sent = vocab.test_sentence(test_sent)
print(model.predict(test_sent))
