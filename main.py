import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import logging
from Word2VecVectors import Word2VecProcessing
from DataManager import DataManager

import time

# Set up gensim logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

images = []
examples = []
examples_weights = []
id2word = dict()
unique_words = set()

# Set up data directories
word_to_vec_trained_model_data_directory = '/home/dmacjam/Word2Vec/'
#websites_data_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Textual/'
websites_data_directory = '../data/TBIRDataSet/Features/Textual/'


def readTextualFeatures(directory, filename, number_of_examples_to_load):
    '''
    Read and preprocess textual features. We cannot load it into numpy, because number of features is not same for
    every example.
    :param directory:
    :param filename:
    :return:
    '''
    global unique_words, examples, examples_weights
    p_stemmer = PorterStemmer()
    stop = stopwords.words('english')
    number_of_lines = 0
    number_of_words = 0
    unique_words_counter = 0
    lines = DataManager.read_visual_file(directory, filename, number_of_examples_to_load)
    for line in lines:
        words = []
        weights = []
        number_of_lines += 1
        line_words = line.split()
        images.append(line_words[0])
        number_of_features = int(line_words[1])
        for j in range(0, number_of_features, 2):
            number_of_words += 1
            word = line_words[j + 2].decode('utf-8')
            word = p_stemmer.stem(word)
            weight = float(line_words[j + 3]) / 100000
            # TODO filter 2 char words, better filter digits e.g. 12th
            if (word not in stop) and (word.isdigit() == False):  # (weight > 2000)
                # word = str(word)        # Because it is unicode - special chars cannot be converted to ascii
                if (word not in unique_words):
                    words.append(word)
                    unique_words.add(word)
                    id2word[unique_words_counter] = word
                    weights.append(weight)
        if ((number_of_lines + 1) % 4000 == 0):
            print("Read %d of %d\n" % (number_of_lines + 1, number_of_examples_to_load))
        examples.append(words)
        examples_weights.append(weights)
    print("Number of text examples read:", number_of_lines, len(examples))
    print("Number of unique words:", len(unique_words), "from ", number_of_words)



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



def loadImageFeatures(directory, filename):
    #with open(os.path.join(directory, filename)) as textual_file:
    print("A")

def loadImageIds(directory, filename):
    # TODO load file with image ids
    print("A")



# Main Functio
if __name__ == '__main__':


    readTextualFeatures(websites_data_directory, 'train_data.txt', 10000)
    #print(images)
    #print(len(examples[0]), examples[0])
    #pprint(examples[:10])

    #print(len(examples_weights), examples_weights[0])

    #lda_model = trainLda()
    #print(lda_model.print_topics(num_topics=10))

    #representAsBow()
    #word2vec = Word2VecProcessing()
    #word2vec.create_word2vec_model(word_to_vec_trained_model_data_directory)

