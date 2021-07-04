import pandas as pd
import spacy
import re
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np

spacy_eng = spacy.load("en_core_web_sm")


class Vocab:
    def __init__(self, frequency=2):
        '''
        Order of execution of functions:
        1.) First Initialize class Vocab
        2.) Now call clean_sentences(data_list)
        3.) Now call build_vocab(data_list)
        4.) Now call numericalize_sentences(data_list)
        4.) Now call pad_num_sentences(data_list,maxlen)
        '''
        self.frequency = frequency
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.tokenizer = spacy_eng.tokenizer
        self.REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    def clean_sentences(self, data_list, inplace=True):
        ret_lst = []
        for i in range(len(data_list)):
            review = data_list[i][0]
            if inplace == True:
                data_list[i][0] = self.REPLACE_NO_SPACE.sub("", review.lower())
                data_list[i][0] = self.REPLACE_WITH_SPACE.sub(
                    " ", data_list[i][0])
            else:
                t_lst = []
                data = self.REPLACE_NO_SPACE.sub("", review.lower())
                data = self.REPLACE_WITH_SPACE.sub(" ", data)
                t_lst.append(data)
                t_lst.append(data_list[i][1])
                ret_lst.append(t_lst)

        if inplace == False:
            return ret_lst

    def build_vocab(self, data_list):
        '''
        Datalist must of type:
        ex: [["review","sentiment"],["review","sentiment"],["review","sentiment"]....]
        '''
        freq = {}
        idx = 4
        print("> Started building vocab for english")
        for data_item in data_list:
            movie_review = data_item[0]
            for word in self.tokenizer(movie_review):
                word = word.text
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1

                if freq[word] == self.frequency:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        print("> Vocab building Finished")
        print("> Vocab_len:", len(self.stoi))

    def numericalize_sentences(self, data_list):
        print("> Numericalizing sentences")
        num_captions = []
        sentiment = []
        for data in data_list:
            review = data[0]
            sentiment_data = data[1]
            if sentiment_data == "positive":
                sentiment.append(1)
            else:
                sentiment.append(0)

            t_lst = []
            for word in self.tokenizer(review):
                word = word.text
                if word in self.stoi:
                    t_lst.append(self.stoi[word])
                else:
                    t_lst.append(self.stoi["<UNK>"])

            num_captions.append(t_lst)

        return num_captions, sentiment

    def pad_num_sentences(self, df_list, maxlen=250):
        print("> Padding sentences")
        num_sent = pad_sequences(df_list, maxlen=maxlen, padding="post")
        return num_sent

    def test_sentence(self, sent):
        ret_lst = []
        for word in self.tokenizer(sent):
            word = word.text
            if word in self.stoi:
                ret_lst.append(self.stoi[word])
            else:
                ret_lst.append(self.stoi["<UNK>"])

        x = []
        x.append(ret_lst)
        x = self.pad_num_sentences(x, maxlen=100)

        # ret_lst = np.array(self.pad_num_sentences(list(ret_lst)))
        return x

    def __len__(self):
        return len(self.stoi)


# if __name__ == "__main__":
#     df = pd.read_csv("Data/archive/IMDB Dataset.csv").values.tolist()
#     vocab = Vocab(frequency=2)
#     vocab.clean_sentences(df)
#     vocab.build_vocab(df)
#     (num_sent, y) = vocab.numericalize_sentences(df)
#     padded = vocab.pad_num_sentences(num_sent)
#     print(len(padded))
    # print(padded[0])
    # print(df[0])
