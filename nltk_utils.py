import nltk
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt')
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    # It's not working check it out when back here!!!!!
    return nltk.word_tokenize(sentence)

# Stemming is reducing a word/token to a limited number of characters that represents it.
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [0,     1,      0,    1,     0,     0,       0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

