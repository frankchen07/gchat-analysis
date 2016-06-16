def get_idf(all_words, list_docs):

    """
    """

    return {word: len(list_docs) / float(len([doc for doc in list_docs if word in doc])) for word in all_words}


def get_tf(word, words):

    """
    """

    return (words.count(word) / float(len(words)))



if __name__ == "__main__":

    print 'TF-IDF support.'

