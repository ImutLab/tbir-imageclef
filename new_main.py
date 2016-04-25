import numpy as np
import os, sys

from gensim import corpora, models
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import logging
from Word2VecVectors import Word2VecProcessing
from DataManager import *
from ConcatenatedLDA import ConcatenatedLDA
import cPickle as pickle

import time


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
    #print(corpus[0])
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
    # =================================Training Data==================================
    websites_data_directory = '/Users/enzo/Projects/TBIR/test/data/data_6stdpt/Features/Textual'
    websites_filename = 'train_data.txt'
    images_features_directory = '/Users/enzo/Projects/TBIR/test/data/data_6stdpt/Features/Visual/Visual/'
    images_features_filename = 'scaleconcept16_data_visual_vgg16-relu7.dfeat'
    image_ids_directory = '/Users/enzo/Projects/TBIR/test/data/data_6stdpt/Features/'
    image_ids_filename = 'scaleconcept16_ImgID.txt_Mod'
    textual_image_ids_directory = '/Users/enzo/Projects/TBIR/test/data/data_6stdpt/Features/'

    trainData = TrainData(websites_data_directory, images_features_directory, image_ids_directory,
                          websites_filename, images_features_filename, image_ids_filename)

    visualMatrix_filename = 'visualMatrix_train.npy'
    #number_of_examples_to_load = 1000
    number_of_examples_to_load = len(trainData.textual_image_ids)
    #trainData.create_id2word(number_of_examples_to_load)
    #trainData.save_visualMatrix_to_file('../visualMatrix_train/', visualMatrix_filename)

    #'''
    #number_of_examples_to_load = 40000
    #number_of_examples_to_load = len(trainData.textual_image_ids)
    # check if there is corpus file
    #corpus_filename = '%scorpus.npy' % number_of_examples_to_load
    #if corpus_filename not in os.listdir('../corpus/'):
    #    trainData.createVocabulary(number_of_examples_to_load)
    #    trainData.concatenateVisualFeatures('../visualMatrix_train', visualMatrix_filename)
        #trainData.save_corpus_to_file('../', corpus_filename)
    #trainData.load_corpus_from_file('../', corpus_filename)
    #print trainData.corpus[0]

    lda = ConcatenatedLDA(trainData)
    '''
    # load corpus from corpus file
    t0 = time.time()
    print("Loading corpus from corpus file")
    np_corpus = np.load(os.path.join('../corpus/', '0corpus.npy'))
    print("Done: load corpus: %s size (%.3fs)" % (np_corpus.shape[0], (time.time()-t0)))
    corpus_chunk = []
    index = 0
    for line in np_corpus:
        index += 1
        if index%3000 == 0: print("Transform %s" % index)
        corpus_chunk.append([(int(wordID), float(weight)) for (wordID, weight) in line])
    lda.trainLDA_from_corpus(corpus_chunk)
    #lda.updateLDA(new_corpus)
    lda.save_ldaModel_to_file('../ldaModel/', 'lda.model')
    print lda.lda_model.print_topics(num_topics=5)
    '''


    # ===================================Teaser Data==================================
    websites_data_directory = '/Users/enzo/Projects/TBIR/test/data/data_6stdpt/DevData/Teaser/'
    websites_filename = 'scaleconcept16.teaser_dev_data_textual.scofeat.v20160212'
    images_features_directory = '/Users/enzo/Projects/TBIR/test/data/data_6stdpt/DevData/Teaser/'
    images_features_filename = 'scaleconcept16.teaser_dev_data_visual_vgg16-relu7.dfeat'
    image_ids_directory = '/Users/enzo/Projects/TBIR/test/data/data_6stdpt/DevData/Teaser/scaleconcept16.teaser_dev_id.v20160212/'
    image_ids_filename = 'scaleconcept16.teaser_dev.ImgID.txt'

    lda.load_ldaModel_from_file('../ldaModel/', 'lda.model')
    id2word = pickle.load(open('../corpus/id2word.dat', 'rb'))

    teaserData = TeaserData(websites_data_directory, images_features_directory, image_ids_directory,
                            websites_filename, images_features_filename, image_ids_filename, id2word)
    teaserData.create_TopicImage_Matrix(None, None, lda.lda_model)

    # ================================================================================

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

