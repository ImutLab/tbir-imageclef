import os
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import pickle

from gensim import corpora

# websites_data_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Textual/'
websites_data_directory = '/Users/enzo/Projects/TBIR/test/data/TBIRDataSet/Features/Textual/'
# images_features_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Visual/Visual/'
images_features_directory = '/Users/enzo/Projects/TBIR/test/data/TBIRDataSet/Features//Visual/Visual/'
# image_ids_directory= '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/'
image_ids_directory= '/Users/enzo/Projects/TBIR/test/data/TBIRDataSet/Features/'
websites_filename = 'train_data.txt'
images_features_filename = 'scaleconcept16_data_visual_vgg16-relu7.dfeat'
image_ids_filename = 'scaleconcept16_ImgID.txt_Mod'

p_stemmer = SnowballStemmer("english")
stop = stopwords.words('english')

class DataManager(object):
    # Dictionaries for creating sparse matrix.
    id2word = dict()
    word2id = dict()

    # Unique words
    unique_words = set()
    unique_words_counter = 0

    # Text corpus
    corpus = []
    test_text_corpus = []

    # Image ids from training data
    textual_img_ids = []
    # Image ids for testing.
    test_text_img_id_gt = []

    # Number of textual data loaded.
    textual_examples_loaded = 0

    # All image ids.
    image_ids = []

    # Testing image vector - for all except training data.
    test_img_vectors = []
    test_img_vectors_ids = []

    def __init__(self):
        self._load_image_ids()
        self.generateVisualFeatures()

    @staticmethod
    def _readVisualFile():
        with open(os.path.join(websites_data_directory, websites_filename)) as textual_file:
             for line in textual_file:
                 yield line

    @staticmethod
    def _readImageFeaturesFile():
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


    def prepareVisualFeatures(self):
        print("Preparing data for testing - Concatenating visual features to textual.")
        visual_lines = DataManager._readImageFeaturesFile()
        train_index = 0
        images_index = 0
        for visual_line in visual_lines:
            image_id = self.image_ids[images_index]
            if image_id == self.textual_img_ids[train_index]:
                #Concatenate textual and image vectors to training corpus
                image_corpus_line = self._generateCorpusForImage(visual_line[1:])
                self.corpus[train_index] = self.corpus[train_index] + image_corpus_line
                train_index += 1
            #else:
                #self.test_img_vectors.append(image_corpus_line)
                #self.test_img_vectors_ids.append(image_id)
            images_index += 1
            # Debug print
            if ((images_index + 1) % 4000 == 0):
                print("Read visual features %d" % (images_index + 1))
            if train_index >= self.textual_examples_loaded:
                break
        print("Done: concatenating visual features to textual")

    def _generateCorpusForImage(self, features):
        image_corpus_line = []
        vector_sum = sum(map(float, features))
        for i in range(0, 4096):
            feature_value = float(features[i]) / vector_sum
            if feature_value != 0:
                feature_name = "feature"+`(i+1)`
                feature_id = self.word2id[feature_name]
                image_corpus_line.append( (feature_id, feature_value) )
        return image_corpus_line


    def _load_image_ids(self):
        print("Loading image ids")
        with open(os.path.join(image_ids_directory, image_ids_filename)) as image_id_file:
            self.image_ids = image_id_file.read().splitlines()
        print("Done loading image ids.")

    def divideTrainAndTestData(self, portion=1000):
        divide_index = portion      #int(float(1-portion) * self.textual_examples_loaded)
        #self.test_text_img_id_gt = self.textual_img_ids[:divide_index]
        #self.test_text_corpus[:] = self.corpus[:divide_index]
        self.corpus = self.corpus[:divide_index]

    def createTextualVocabulary(self, lineLimit=float("inf")):
        '''
        Read and preprocess textual features. Create vocabulary and corpus.
        :param directory:
        :param filename:
        :return:
        '''
        print("Loading textual features and creating textual vocabulary.")
        number_of_lines = 0
        number_of_words = 0
        lines = DataManager._readVisualFile()
        for line in lines:
            # For every line in document.
            corpus_line_dict = dict()
            number_of_lines += 1
            line_words = line.split()
            self.textual_img_ids.append(line_words[0])
            number_of_features = int(line_words[1])
            # for j in range(0, number_of_features, 2):
            for j in range(0, number_of_features*2, 2):
                number_of_words += 1
                word = line_words[j + 2].decode('utf-8')
                self.processWord(word)
                # Normalize weight
                weight = float(line_words[j + 3]) / 100000
                # TODO filter 2 char words, better filter digits e.g. 12th
                if (word not in stop) and (word.isdigit() == False):
                    if word not in self.unique_words:
                        self.unique_words.add(word)
                        self.id2word[self.unique_words_counter] = word
                        self.word2id[word] = self.unique_words_counter
                        corpus_line_dict[self.unique_words_counter] = weight
                        self.unique_words_counter += 1
                    else:
                        # Add weight if exists for current website.
                        word_id = self.word2id[word]
                        if word_id in corpus_line_dict:
                            corpus_line_dict[word_id] += weight
                        else:
                            corpus_line_dict[word_id] = weight
            # Create array of tuples (word_id, weight) from dictionary
            corpus_line = []
            for key, value in corpus_line_dict.iteritems():
                corpus_line.append( (key, value) )
            self.corpus.append(corpus_line)
            # Debug output
            if ((number_of_lines + 1) % 4000 == 0):
                print("Read %d" % (number_of_lines + 1))
            # Limit number of lines
            if number_of_lines >= lineLimit:
                break

        print("Number of text examples read:", number_of_lines)
        print("Number of unique words:", len(self.unique_words), "from ", number_of_words)
        assert len(self.corpus) == len(self.textual_img_ids)
        self.textual_examples_loaded = len(self.textual_img_ids)

        # Delete unnecessary variables
        self.unique_words = None

        self.saveTraining()


    def processWord(self, word):
        # Tokenize word to delete apostrophe.
        word = word_tokenize(word)[0]
        # Stem the word.
        return p_stemmer.stem(word)

    def _readTextualFile(self):
        with open(os.path.join(websites_data_directory, websites_filename)) as textual_file:
             for line in textual_file:
                 words = line.split()[2::2]
                 words = [self.processWord(word.decode('utf-8')) for word in words]
                 yield words

    def createDictionary(self):
        corpus = []
        lines = self._readTextualFile()
        for line in lines:
            corpus.append(line)
        dictionary = corpora.Dictionary(corpus)
        print(dictionary)
        # remove stop words and words that appear only once
        stop_ids = [dictionary.token2id[stopword] for stopword in stop if stopword in dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        dictionary.compactify() # remove gaps in id sequence after words that were removed
        print(dictionary)
        dictionary.save('/media/dmacjam/Data disc/Open data/TBIR/dictionary')

    def saveTraining(self):
        with open('/media/dmacjam/Data disc/Open data/TBIR/id2word', 'wb') as file:
            pickle.dump(self.id2word, file, pickle.HIGHEST_PROTOCOL)
        with open('/media/dmacjam/Data disc/Open data/TBIR/word2id', 'wb') as file:
            pickle.dump(self.word2id, file, pickle.HIGHEST_PROTOCOL)
        with open('/media/dmacjam/Data disc/Open data/TBIR/custom-dict', 'wb') as file:
            pickle.dump(self.corpus, file, pickle.HIGHEST_PROTOCOL)

    def load_training(self):
        with open('/media/dmacjam/Data disc/Open data/TBIR/id2word', 'rb') as file:
            self.id2word = pickle.load(file)
        with open('/media/dmacjam/Data disc/Open data/TBIR/word2id', 'rb') as file:
            self.word2id = pickle.load(file)
        with open('/media/dmacjam/Data disc/Open data/TBIR/custom-dict', 'rb') as file:
            self.corpus = pickle.load(file)





