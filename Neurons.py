import numpy as np  # installed numpy
import nltk  # natural language tool kit
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')

stemmer = PorterStemmer()


def token(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_word = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)

    for idx, w in enumerate(words):
        if w in sentence_word:
            bag[idx] = 1
    return bag
