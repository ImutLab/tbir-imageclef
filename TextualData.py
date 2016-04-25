import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from gensim.models.ldamodel import LdaModel

from Word2VecVectors import Word2VecProcessing
from DataManager import DataManager

import time

class TextualData():
    #imgIDs = []
    #id2word = {}
    #unique_words = set()

    def __init__(self, directory, filename, number_of_examples_to_load):
        self.imgIDs = []
        self.id2word = {}
        self.unique_words = set()
        self.examples_words = []
        self.examples_weights = []
        self.w2v_Matrix = None
        self.readTextualFeatures(directory, filename, number_of_examples_to_load)

    def readTextualFeatures(self, directory, filename, number_of_examples_to_load):
        '''
        Read and preprocess textual features.
        :param directory
        :param filename
        :return
        '''
        p_stemmer = PorterStemmer()
        stop = stopwords.words('english')
        number_of_lines = 0
        number_of_words = 0
        unique_words_counter = 0
        lines = DataManager.read_visual_file(directory, filename, number_of_examples_to_load)
        for line in lines:
            words = []
            weights = []
            line_words = line.split()
            self.imgIDs.append(line_words[0])
            number_of_features = int(line_words[1])
            for j in range(2, len(line_words), 2):
                number_of_words += 1
                word, weight = line_words[j].decode('utf-8'), float(line_words[j+1])/100000
                word = p_stemmer.stem(word)
                if (word not in stop) and (word.isdigit()==False):
                    if word not in self.unique_words:
                        words.append(word)
                        self.unique_words.add(word)
                        self.id2word[unique_words_counter] = word
                        weights.append(weight)
                        unique_words_counter += 1
            number_of_lines += 1
            if number_of_lines%5000 == 0:
                print("Read %d of %d" % (number_of_lines, number_of_examples_to_load))
            self.examples_words.append(words)
            self.examples_weights.append(weights)
        print("Number of text examples read: %s of %s" % (number_of_lines, len(self.examples_words)))
        print("Number of unique words: %s from %s" % (len(self.unique_words), number_of_words))

    def construct_matrix_by_word2vec(self, word2vec_trained_model_data_directory):
        process = Word2VecProcessing()
        process.create_word2vec_model(word2vec_trained_model_data_directory)
        print type(process.word2vec_model.syn0)
        print len(process.word2vec_model.syn0[0])
        process.create_word2index()
        process.check_words_in_word2vec_vocabulary(self.unique_words)

        #self.w2v_Matrix = process.create_average_vectors(self.examples_words, self.examples_weights)
        #print("Dimensions of matrix: %s, %s" % (len(self.w2v_Matrix), len(self.w2v_Matrix[0])))


if __name__ == '__main__':
    texData = TextualData('../data/TBIRDataSet/Features/Textual', 'train_data.txt', 10000)
    texData.construct_matrix_by_word2vec('../data/')
    #corpus = generate_corpus('./data/TBIRDataSet/Features/Textual/', 'train_data.txt', 1000)

