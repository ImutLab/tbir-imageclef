from gensim.models import Word2Vec
import time
import numpy as np
import os

class Word2VecProcessing(object):
    word2vec_model = None
    word2vec_word2index = {}


    def create_word2vec_model(self, word_to_vec_trained_model_data_directory):
        self.word2vec_model = self._loadWord2vecModel(word_to_vec_trained_model_data_directory,'vectors.bin', True)


    def create_average_vectors(self, examples, examples_weights):
        examples_average_vectors = self._getAvgFeatureVectors(examples, self.word2vec_model, examples_weights)
        print(examples_average_vectors.shape)
        # print(examples_average_vectors[0])

    def create_word2index(self):
        if self.word2vec_model:
            index2word = self.word2vec_model.index2word
            for i in range(len(index2word)):
                index, word = i, index2word[i]
                self.word2vec_word2index[word] = index
        else: print("Word2Vec model has not been constructed!")

    def check_words_in_word2vec_vocabulary(self, unique_words):
        count = 0
        size = len(unique_words)
        reduced_unique_words = set()
        for word in unique_words:
            if word in self.word2vec_model.vocab: reduced_unique_words.add(word)
            else:
                if count < 100: print word
                count += 1
        print("There is %s words (of %s) in Word2Vec vocabulary!" % (count, len(unique_words)))
        return reduced_unique_words

    def _loadWord2vecModel(self, directory, filename, binary=False):
        '''
        Load Word2Vec model from file.
        :param directory:
        :param filename:
        :return:
        '''
        model = Word2Vec.load_word2vec_format(os.path.join(directory, filename), binary=binary)
        # Print a wordsXfeatures matrrix shape

        print(model.syn0.shape)
        # print(model["car"])
        # print(model["000qUQAfomr0QAm4"])
        # accuracy = model.accuracy('/home/dmacjam/Word2Vec/questions-words.txt')
        return model

    def _makeFeatureVector(self, words, word_vector_model, weights, num_features):

        '''
        Function to average all words vectors from a website.
        :param words:
        :param num_features:
        :return:
        '''

        # TODO add weights based on score
        feature_vector = np.zeros((num_features,), dtype="float32")
        nwords = 0
        assert len(words) == len(weights)
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        index2word_set = set(word_vector_model.index2word)
        # create a hash table for word and index
        for i in range(0, len(words)):
            word = words[i]
            weight = weights[i]
            if word in index2word_set:
                nwords += 1
                # Sum feature vector
                feature_vector = np.add(feature_vector, weight * word_vector_model[word])
        # Divide the result by the number of words to get the average
        if (nwords != 0):
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector

    def _getAvgFeatureVectors(self, websites, word_vector_model, weights):
        '''
        Given a set of website (each one a list of words) calculate the average feature vector
        for each one and return  a 2D numpy array.
        :param websites:
        :param word_vector_model:
        :return:
        '''
        print("Getting average feature vectors")
        start = time.time()  # Start time
        # Get number of features in word2vec model.
        num_features = word_vector_model.syn0.shape[1]
        # Inicialize a counter.
        counter = 0
        # Preallocate a 2D numpy array, for speed
        website_feature_vecs = np.zeros((len(websites), num_features), dtype="float32")
        for i in range(0, len(websites)):
            website = websites[i]
            # Print a status message every 1000th review
            if counter % 100. == 0 and counter != 0:
                print "Get average vector for %d of %d" % (counter, len(websites))
            website_feature_vecs[counter] = self._makeFeatureVector(website, word_vector_model, weights[i], num_features)
            counter += 1
        end = time.time()
        print "Time taken for averaging feature vectors: ", end - start, "seconds."
        return website_feature_vecs
