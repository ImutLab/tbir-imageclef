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



# Main Function
if __name__ == '__main__':
    data_manager = DataManager()

    #data_manager.createDictionary()

    data_manager.createTextualVocabulary()
    #data_manager.divideTrainAndTestData()

    # Training
    #data_manager.prepareVisualFeatures()



    #lda = ConcatenatedLDA(data_manager)
    #lda.trainLDA()
    #lda.loadLDAModel()
    #lda.inferNewDocuments()

