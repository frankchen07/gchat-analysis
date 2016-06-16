import pandas as pd
import nltk
from nltk.collocations import *
from nltk.util import ngrams



def gen_trigrams(sentences):

    """
    @input: sentences by author and day
    @action: grabs trigram combinations
    @output: outputs a list of trigrams

    """

    day_trigrams = []
    for sentence in sentences:
        day_trigrams.append([trigram for trigram in ngrams(sentence, 3)])
    day_trigrams = [trigram for sentence in day_trigrams for trigram in sentence]

    return day_trigrams


def gen_bigrams(sentences):

    """
    @input: sentences by author and day
    @action: grabs bigram combinations
    @output: outputs a list of bigrams

    """

    day_bigrams = []
    for sentence in sentences:
        day_bigrams.append([bigram for bigram in nltk.bigrams(sentence)])
    day_bigrams = [bigram for sentence in day_bigrams for bigram in sentence]

    return day_bigrams



if __name__ == "__main__":

    print 'Support script for ngram creation. Load me up.'











