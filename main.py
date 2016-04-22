import numpy as np
import os

from gensim import corpora, models
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import logging
from Word2VecVectors import Word2VecProcessing
from DataManager import DataManager
from ConcatenatedLDA import ConcatenatedLDA


# Set up data directories
word_to_vec_trained_model_data_directory = '/home/dmacjam/Word2Vec/'

# Set up gensim logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def train_lda(dictionary):
    print("a")



def trainLda():
    '''
    LDA training by gensim.
    :param corpus:
    :return:
    '''
    dictionary = corpora.Dictionary(examples)
    print(len(dictionary), dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in examples]
    print(corpus[0])
    return models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)

def representAsBow():
    print("CBOW representation")
    vectorizer = CountVectorizer(analyzer = "word",   \
                                tokenizer = None,    \
                                preprocessor = None, \
                                stop_words = None,   \
                                max_features = 5000)
    flattened_examples = [val for sublist in examples for val in sublist]
    print("All words count:", len(flattened_examples))
    train_data_features = vectorizer.fit_transform(list(unique_words))
    train_data_features = train_data_features.toarray()
    print(train_data_features.shape, train_data_features[0])
    vocab = vectorizer.get_feature_names()
    print(vocab[:100])




# Main Function
if __name__ == '__main__':
    data_manager = DataManager()
    data_manager.createVocabulary(500)
    data_manager.concatenateVisualFeatures()

    lda = ConcatenatedLDA(data_manager)
    lda.trainLDA()


    #readTextualFeatures(websites_data_directory, 'train_data.txt', 5000)
    #print(images)
    #print(len(examples[0]), examples[0])
    #pprint(examples[:10])

    #print(len(examples_weights), examples_weights[0])

    #lda_model = trainLda()
    #print(lda_model.print_topics(num_topics=10))

    #representAsBow()
    #word2vec = Word2VecProcessing()
    #word2vec.create_word2vec_model(word_to_vec_trained_model_data_directory)

