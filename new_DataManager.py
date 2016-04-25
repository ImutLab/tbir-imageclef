import os, sys
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import cPickle as pickle
from visualMatrix import *

# websites_data_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Textual/'
#websites_data_directory = '/Users/enzo/Projects/TBIR/test/data/TBIRDataSet/Features/Textual/'
# images_features_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Visual/Visual/'
#images_features_directory = '/Users/enzo/Projects/TBIR/test/data/TBIRDataSet/Features//Visual/Visual/'
# image_ids_directory= '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/'
#image_ids_directory= '/Users/enzo/Projects/TBIR/test/data/TBIRDataSet/Features/'
#websites_filename = 'train_data.txt'
#images_features_filename = 'scaleconcept16_data_visual_vgg16-relu7.dfeat'
#image_ids_filename = 'scaleconcept16_ImgID.txt_Mod'


class DataManager(object):
    text_examples = []
    text_examples_weights = []

    id2word = dict()
    word2id = dict()

    unique_words = set()
    unique_words_counter = 0

    text_examples_img_ids = []
    textual_examples_loaded = 0

    corpus = []
    image_ids = []

    def __init__(self, websites_data_directory, images_features_directory, image_ids_directory,
                       websites_filename, images_features_filename, image_ids_filename):
        self._load_image_ids(image_ids_directory, image_ids_filename)

        self.websites_data_directory = websites_data_directory
        self.websites_filename = websites_filename
        self.images_features_directory = images_features_directory
        self.images_features_filename = images_features_filename

        self.generateVisualFeatures()
        pass

    @staticmethod
    def _readVisualFile(websites_data_directory, websites_filename, number_of_examples_to_load):
        with open(os.path.join(websites_data_directory, websites_filename)) as textual_file:
             for i in range(0, number_of_examples_to_load):
                 yield textual_file.readline()

    @staticmethod
    def _readImageFeaturesFile(images_features_directory, images_features_filename):
        with open(os.path.join(images_features_directory, images_features_filename), 'r') as fVisual:
            description = fVisual.readline()
            numImgs, numFeatures = description.split()
            print "%s images, %s features" % (numImgs, numFeatures)
            for line in fVisual:
                yield line.split()

    def generateVisualFeatures(self):
        '''
        Generate word names and ids for image features.
        :return:
        '''
        for i in range(0, 4096):
            feature_name = "feature"+`(i+1)`
            self.id2word[self.unique_words_counter] = feature_name
            self.word2id[feature_name] = self.unique_words_counter
            self.unique_words_counter += 1

    def _generateCorpusForImage(self, features):
        image_corpus_line = []
        vector_sum = sum(float(feature_value) for feature_value in features)
        for i in range(0, 4096):
            feature_value = float(features[i]) / vector_sum
            if (feature_value != 0):
                feature_name = "feature"+`(i+1)`
                feature_id = self.word2id[feature_name]
                image_corpus_line.append( (feature_id, feature_value) )
        return image_corpus_line

    def _load_image_ids(self, image_ids_directory, image_ids_filename):
        print("===============================================================")
        print("Loading image ids (from image visual feature file)")
        with open(os.path.join(image_ids_directory, image_ids_filename)) as image_id_file:
            self.image_ids = image_id_file.read().splitlines()
        print("Done loading image ids: %s" % len(self.image_ids))


class TrainData(DataManager):
    def __init__(self, websites_data_directory, images_features_directory, image_ids_directory,
                       websites_filename, images_features_filename, image_ids_filename, textual_image_ids_directory=None, textual_image_ids_filename=None):
        DataManager.__init__(self, websites_data_directory, images_features_directory, image_ids_directory,
                                   websites_filename, images_features_filename, image_ids_filename)

        self.textual_image_ids = []
        #self._load_textual_image_ids(websites_data_directory, websites_filename)
        self._load_textual_image_ids()
        self.np_corpus = None
        #self.generateVisualFeatures()

    def generateVisualFeatures(self):
        '''
        Generate word names and ids for image features.
        :return:
        '''
        for i in range(0, 4096):
            feature_name = "feature"+`(i+1)`
            self.id2word[self.unique_words_counter] = feature_name
            self.word2id[feature_name] = self.unique_words_counter
            self.unique_words_counter += 1

    def concatenateVisualFeatures_Old(self):
        print("===============================================================")
        print("Concatenating visual features to textual.")

        visual_lines = DataManager._readImageFeaturesFile(self.images_features_directory, self.images_features_filename)
        textual_index = 0
        images_index = 0
        for visual_line in visual_lines:
            image_id = self.image_ids[images_index]
            if image_id == self.text_examples_img_ids[textual_index]:
                image_corpus_line = self._generateCorpusForImage(visual_line[1:])
                # Concatenate textual and image vectors
                self.corpus[textual_index] = self.corpus[textual_index] + image_corpus_line
                textual_index += 1
            images_index += 1
            # Debug print
            if ((images_index + 1) % 4000 == 0):
                print("Read visual features %d" % (images_index + 1))
                print("The size of corpus with %s lines: %s bytes" % (len(self.corpus), sys.getsizeof(self.corpus)))
            # Stop when all textual examples were found.
            if ( textual_index >= self.textual_examples_loaded ):
                break
        print("The size of corpus with %s lines: %s bytes" % (len(self.corpus), sys.getsizeof(self.corpus)))
        print("Done: concatenating visual features to textual")

    def concatenateVisualFeatures(self, visualMatrix_directory, visualMatrix_filename):
        '''
        Visual Matrix was fetched chunk by chunk
        30000 images per chunk (per file)
        :param: visualMatrix_filename: 'visualMatrix_train.npy'
        '''
        print("===============================================================")
        print("Concatenating visual features to textual.\n")
        # assert(len(self.corpus) == visualMatrix.shape[0])
        self.np_corpus = np.zeros((len(self.textual_image_ids),), dtype=object)

        #index = 0   # index for saving corpus
        for i in range(len(self.corpus)/30000 + 1):
            visualMatrix = np.load(os.path.join(visualMatrix_directory, (('%s' % (i*30000)) + visualMatrix_filename)))
            for j in range(visualMatrix.shape[0]):
                if (i*30000 + j) >= len(self.corpus): break
                image_corpus_line = self._generateCorpusForImage(visualMatrix[j])
                self.np_corpus[i*30000 + j] = np.array(self.corpus[i*30000 + j] + image_corpus_line)

                #index += 1
                #if index >= 10000:
                #    self.save_corpus_to_file('../corpus/', ( '%scorpus.dat' % (((i*30000+j)/10000)*10000) ))
                #    index = 0

                if (i*30000 + j) % 4000 == 0:
                    print("Read visual features %s" % (i*30000 + j))

        print("The size of corpus with %s lines: %s bytes\n" % (len(self.corpus), sys.getsizeof(self.corpus)))

        print("Saving corpus to file\n")
        for i in range(0, self.np_corpus.shape[0]/30000):
            corpus_chunk = self.np_corpus[i*30000: (i+1)*30000]
            np.save(os.path.join('../corpus/', ('%s'%(i*30000))+'corpus.npy'), corpus_chunk)
            print("Save %s corpus.\n" % (i*30000))
        np.save(os.path.join('../corpus/', ('300000'+'corpus.npy')), self.np_corpus[300000:])
        print("Save all corpus chunk\n")
        np.save(os.path.join('../corpus/', ('Corpus.npy')), self.np_corpus)
        print("Done: save corpus to file\n")
        print("Done: concatenating visual features to textual\n")

    def save_visualMatrix_to_file(self, visualMatrix_directory, visualMatrix_filename):
        if visualMatrix_filename not in os.listdir(visualMatrix_directory):
            visMatrix = readVisMatrix(self.textual_image_ids, self.image_ids, self.images_features_directory, self.images_features_filename)
            matrix2File(visMatrix, visualMatrix_directory, visualMatrix_filename)

    def _load_textual_image_ids(self):
        print("===============================================================\n")
        print("Loading textual image ids (image ids in train_data.txt)")
        with open(os.path.join(self.websites_data_directory, self.websites_filename)) as textual_image_id_file:
            for line in textual_image_id_file:
                self.textual_image_ids.append(line.split()[0])
        print("Done: load %s textual image ids.\n" % len(self.textual_image_ids))

    def createVocabulary(self, number_of_examples_to_load):
        '''
        Read and preprocess textual features. Create vocabulary and corpus.
        :param directory:
        :param filename:
        :param number_of_examples_to_load:
        :return:
        '''
        print("===============================================================")
        print("Loading textual features.\n")
        p_stemmer = PorterStemmer()
        stop = stopwords.words('english')
        number_of_lines = 0
        number_of_words = 0
        lines = DataManager._readVisualFile(self.websites_data_directory, self.websites_filename, number_of_examples_to_load)
        for line in lines:
            # For every line in document.
            words = []
            weights = []
            corpus_line_dict = dict()
            number_of_lines += 1
            line_words = line.split()
            self.text_examples_img_ids.append(line_words[0])
            number_of_features = int(line_words[1])
            # for j in range(0, number_of_features, 2):
            for j in range(0, number_of_features*2, 2):
                number_of_words += 1
                word = line_words[j + 2].decode('utf-8')
                # Tokenize word to delete apostrophe.
                word = word_tokenize(word)[0]
                # Stem the word.
                word = p_stemmer.stem(word)
                # Normalize weight
                weight = float(line_words[j + 3]) / 100000
                # TODO filter 2 char words, better filter digits e.g. 12th
                if (word not in stop) and (word.isdigit() == False):
                    if (word not in self.unique_words):
                        # words.append(word)
                        self.unique_words.add(word)
                        self.id2word[self.unique_words_counter] = word
                        self.word2id[word] = self.unique_words_counter
                        corpus_line_dict[self.unique_words_counter] = weight
                        self.unique_words_counter += 1
                        # weights.append(weight)
                    else:
                        word_id = self.word2id[word]
                        if word_id in corpus_line_dict:
                            corpus_line_dict[word_id] += weight
                        else:
                            corpus_line_dict[word_id] = weight
            # Create array of tuples (word_id, weight)
            corpus_line = []
            for key, value in corpus_line_dict.iteritems():
                corpus_line.append( (key, value) )
            self.corpus.append(corpus_line)
            # Debug print
            if ((number_of_lines + 1) % 4000 == 0):
                print("Read %d of %d\n" % (number_of_lines + 1, number_of_examples_to_load))
            #self.text_examples.append(words)
            #self.text_examples_weights.append(weights)

        print("Number of text examples read: %s, %s" % (number_of_lines, len(self.text_examples_img_ids)))
        print("Number of unique words: %s from %s" % (len(self.unique_words), number_of_words))

        assert len(self.corpus) == len(self.text_examples_img_ids)
        self.textual_examples_loaded = len(self.text_examples_img_ids)

        #print self.corpus[0]
        #print self.id2word[1]

    def create_id2word(self, number_of_examples_to_load):
        '''
        Read and preprocess textual features. Create vocabulary.
        :param directory:
        :param filename:
        :param number_of_examples_to_load:
        :return:
        '''
        print("===============================================================")
        print("Loading textual features.\n")
        p_stemmer = PorterStemmer()
        stop = stopwords.words('english')
        number_of_lines = 0
        number_of_words = 0
        lines = DataManager._readVisualFile(self.websites_data_directory, self.websites_filename, number_of_examples_to_load)
        for line in lines:
            # For every line in document.
            words = []
            number_of_lines += 1
            line_words = line.split()
            self.text_examples_img_ids.append(line_words[0])
            number_of_features = int(line_words[1])
            # for j in range(0, number_of_features, 2):
            for j in range(0, number_of_features*2, 2):
                number_of_words += 1
                word = line_words[j + 2].decode('utf-8')
                word = word_tokenize(word)[0]
                word = p_stemmer.stem(word)
                if (word not in stop) and (word.isdigit() == False):
                    if (word not in self.unique_words):
                        self.unique_words.add(word)
                        self.id2word[self.unique_words_counter] = word
                        self.word2id[word] = self.unique_words_counter
                        self.unique_words_counter += 1
            # Debug print
            if ((number_of_lines + 1) % 4000 == 0):
                print("Read %d of %d\n" % (number_of_lines + 1, number_of_examples_to_load))

        print("Number of unique words: %s from %s" % (len(self.unique_words), number_of_words))
        # save id2word to file
        print("\nSaving id2word to file.")
        t0 = time.time()
        pickle.dump(self.id2word, open('../corpus/id2word.dat', 'wb'), True)
        print("Done: %.3fs\n" % (time.time()-t0))

    def save_corpus_to_file(self, corpus_directory, corpus_filename):
        print("\n=================================")
        print("Save corpus to file.")
        #self.np_corpus = np.array(self.corpus, dtype='int32, float32')
        t0 = time.time()
        pickle.dump(self.corpus, open(os.path.join(corpus_directory, corpus_filename), 'wb'), True)
        print("The corpus file size is: %s bytes" % os.path.getsize(os.path.join(corpus_directory, corpus_filename)))
        #np.save(os.path.join(corpus_directory, corpus_filename), self.np_corpus)
        print("Done: %.3fs\n" % (time.time()-t0))

    def load_corpus_from_file(self, corpus_directory, corpus_filename):
        print("\n=================================")
        print("Load corpus from file.")
        t0 = time.time()
        self.corpus = pickle.load(open(os.path.join(corpus_directory, corpus_filename), 'rb'))
        print("Done: %.3fs\n" % (time.time()-t0))


class TeaserData(DataManager):

    def __init__(self, websites_data_directory, images_features_directory, image_ids_directory,
                       websites_filename, images_features_filename, image_ids_filename, id2word):
        DataManager.__init__(self, websites_data_directory, images_features_directory, image_ids_directory,
                                   websites_filename, images_features_filename, image_ids_filename)
        self.id2word = id2word
        #self.generateVisualFeatures()

    def _readTextualFile(websites_data_directory, websites_filename):
        with open(os.path.join(websites_data_directory, websites_filename)) as textual_file:
            while textual_file:
                yield textual_file.readline()

    def create_WordTopic_Matrix(self, word2id, text, ldaModel):
        '''
        Read and preprocess text.
        Create word-topic distribution matrix.
        :param text: text needed predict corresponding image
        :param ldaModel:
        :return:
        '''
        print("\n===============================================================")
        print("")
        p_stemmer = PorterStemmer()
        stop = stopwords.words('english')

        words = []
        text_words = text.split()
        textID, numWords = text_words[0], text_words[1]
        for j in range(0, numWords*2, 2):
            word = text_words[j+2].decode('utf-8')
            word = word_tokenize(word)[0]
            word = p_stemmer.stem(word)
            if (word not in stop) and (word.isdigit()==False):
                if (word not in words):
                    words.append(word)

        numTopics = 100
        wordTopic_Matrix = np.zeros((numWords, numTopics), dtype='float32')
        for i in range(numWords):
            for j in range(numTopics):
                word_distr = ldaModel.get_topic_terms(j)
                for wordID, p in word_distr:
                    if word2id[words[i]] == wordID:
                        wordTopic_Matrix[i][j] = p

        return wordTopic_Matrix

    def create_TopicImage_Matrix(self, word2id, visualMatrix, ldaModel):
        '''
        Load visual Matrix for Teaser images.
        Create topic-image distribution matrix
        '''
        print("\n===============================================================")
        print("\nCreating topic-image distribution matrix\n")
        if not visualMatrix: visualMatrix = self.create_visualMatrix()
        numTopics = 100
        numImages = len(self.image_ids)
        assert(numImages==visualMatrix.shape[0])
        topicImage_Matrix = np.zeros((numTopics, numImages), dtype='float32')
        flag = 0
        for i in range(numImages):
            # img = self._generateCorpusForImage(visualMatrix[i])
            img = [(j, visualMatrix[i][j]) for j in range(visualMatrix.shape[1])]
            img_lda = ldaModel[img]     # topic-image distribution
            for topicID, p in img_lda:
                if p!=0.: topicImage_Matrix[topicID][i] = p

        print("Done: create topic-image distribution matrix")
        return topicImage_Matrix

    def create_visualMatrix(self):
        '''
        Create visual Matrix for Teaser task
        '''
        print("================================================================")
        print("Creating visual feature Matrix from image features file")
        vMatrix = np.zeros((len(self.image_ids), 4096), dtype='float32')
        with open(os.path.join(self.images_features_directory, self.images_features_filename)) as fVisual:
            description = fVisual.readline()
            numImgs, numFeatures = description.split()
            print("Reading %s teaser images, %s features" % (numImgs, numFeatures))
            assert(len(self.image_ids) == int(numImgs))
            index = 0
            for line in fVisual:
                vector = line.split()
                vector = [float(vector[i]) for i in range(len(vector))]
                vMatrix[index] = vector
                index += 1
                if index%1000 == 0:
                    print("Read %d visual feature vectors from %s images" % (index, numImgs))
        # checked: visual matrix generated rightly
        print("Done: read visual feature Matrix")
        return vMatrix

    def predict(self, wordTopic_Matrix, topicImage_Matrix, k=100):
        simMatrix = np.dot(wordTopic_Matrix, topicImage_Matrix)
        simMatrix = np.transpose(simMatrix)
        sim = []
        for i in range(simMatrix.shape[0]):
            product = 1.0
            for j in range(simMatrix.shape[1]):
                if simMatrix[i][j] != 0.:
                    product *= simMatrix[i][j]
            sim.append(product)
        sim.sort(reverse=True)

        return sim[:k]

