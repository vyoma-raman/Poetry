import requests
import re
import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from nltk.tokenize import sent_tokenize

from callback import Callback

class existingGenerator(object):
    """Generator object that produces poetry given seed text. Object has already been created and saved."""
    def __init__(self, filename, model, tokenizer, max_sequence_len, accuracy):
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.callback = Callback(accuracy)
        self.model = model

    def generatePoem(self, seed, length):
        """
        Creates a poem starting from a seed text.
        :str seed: seed text from which poem is generated
        :int length: number of words to generate
        :str return: generated poem
        """
        full_text = seed
        for _ in range(length):
            token_list = self.tokenizer.texts_to_sequences([full_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len - 1, padding='pre')
            predicted = np.argmax(self.model.predict(token_list), axis=-1)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            full_text += " " + output_word
        return full_text

class newGenerator(existingGenerator):
    """New poetry generator object to be trained."""
    def __init__(self, name, num_pages, accuracy=1):
        self.corpus = sent_tokenize(self.getCorpus("https://dailycal.org/author/" + name + "/", num_pages))
        self.tokenizer = Tokenizer()
        self.max_sequence_len, X_train, y_train = self.processCorpus()
        self.callback = Callback(accuracy)
        self.model = self.createModel(X_train, y_train)
        self.filename = name + "-" + str(accuracy)

    def getCorpus(self, link, p):
        """
        Gets corpus of texts by a given Daily Cal author.
        :str link: link to an author page
        :int p: number of pages of articles by this author
        :str return: concatenated texts of articles by this author
        """
        # Initialize list to hold links to articles
        article_links = []
        # Get links to all articles
        for page in range(p):
            articles = requests.get(link + "page/" + str(page + 1) + "/").text
            article_links += re.findall('<a href="(.+)"><img width', articles)
        # Instantiate string to hold all text
        all_text = ""
        # Get each article
        for link in article_links:
            article = requests.get(link).text
            # Get article text
            article_text = re.findall('<div class="entry-content">\n([\s\S]+)' + "\n<p id='tagline'>", article)[0]
            # Clean article text
            article_text = re.sub("<.+?>", "", article_text)
            article_text = re.sub("â..", "'", article_text)
            article_text = re.sub("Â", "", article_text)
            article_text = re.sub("&#8211;", "", article_text)
            article_text = re.sub("\xa0", "", article_text)
            # Append article text to string
            all_text += "\n\n" + article_text
        return all_text.strip()

    def processCorpus(self):
        """
        Modifies corpus into text appropriate for training.
        :param return max_sequence_len: length of longest sentence
        :param return X_train: features of training set
        :param return y_train: labels of training set
        """
        # Fit tokenizer on corpus
        self.tokenizer.fit_on_texts(self.corpus)
        input_sequences = []
        for line in self.corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        # Pad shorter sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        # Assemble training data
        X_train, labels = input_sequences[:,:-1], input_sequences[:,-1]
        y_train = tf.keras.utils.to_categorical(labels, num_classes=len(self.tokenizer.word_index) + 1)
        return max_sequence_len, X_train, y_train

    def createModel(self, X_train, y_train):
        """
        Creates and trains model on processed corpus.
        :param X_train: features of training set
        :param y_train: labels of training set
        :param return: model used to generate poetry
        """
        total_words = len(self.tokenizer.word_index) + 1
        # Create model and layers
        model = Sequential()
        model.add(Embedding(total_words, 100, input_length=self.max_sequence_len-1))
        model.add(Bidirectional(LSTM(150)))
        model.add(Dense(total_words, activation="softmax"))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
        # Fit model to training data
        fitting = model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=[self.callback])
        return model
