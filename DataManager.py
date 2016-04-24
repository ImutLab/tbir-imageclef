import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

websites_data_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Textual/'
images_features_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Visual/Visual/'
image_ids_directory= '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/'
websites_filename = 'train_data.txt'
images_features_filename = 'scaleconcept16_data_visual_vgg16-relu7.dfeat'
image_ids_filename = 'scaleconcept16_ImgID.txt_Mod'


class DataManager(object):
    text_examples = []
    text_examples_weights = []
    id2word = dict()
    word2id = dict()
    unique_words = set()
    text_examples_img_ids = []
    train_text_img_ids = []
    test_text_img_ids = []
    textual_examples_loaded = 0
    corpus = []
    unique_words_counter = 0
    image_ids = []
    # train_text_corpus = []
    test_text_corpus = []
    test_img_vectors = []
    divide_index = 0

    def __init__(self):
        self._load_image_ids()
        self.generateVisualFeatures()
        pass

    @staticmethod
    def _readVisualFile(number_of_examples_to_load):
        with open(os.path.join(websites_data_directory, websites_filename)) as textual_file:
             for i in range(0, number_of_examples_to_load):
                 yield textual_file.readline()

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

    def divideTrainAndTestData(self, portion):
        self.divide_index = int(float(portion) * self.textual_examples_loaded)
        self.test_text_corpus = self.corpus[:(self.textual_examples_loaded - self.divide_index)]
        self.corpus = self.corpus[(self.textual_examples_loaded - self.divide_index):]
        self.train_text_img_ids = self.text_examples_img_ids[(self.textual_examples_loaded - self.divide_index):]
        self.test_text_img_ids = self.text_examples_img_ids[:(self.textual_examples_loaded - self.divide_index)]
        print("Train corpus length", len(self.corpus))
        print("Test corpus length", len(self.test_text_corpus))


    def concatenateVisualFeatures(self):
        print("Concatenating visual features to textual.")
        visual_lines = DataManager._readImageFeaturesFile()
        train_index = 0
        test_index = 0
        images_index = 0
        for visual_line in visual_lines:
            image_id = self.image_ids[images_index]
            image_corpus_line = self._generateCorpusForImage(visual_line[1:])
            if train_index < self.divide_index and image_id == self.train_text_img_ids[train_index]:
                # Concatenate textual and image vectors to training corpus
                self.corpus[train_index] = self.corpus[train_index] + image_corpus_line
                train_index += 1
            elif test_index < (self.textual_examples_loaded - self.divide_index) and image_id == self.test_text_img_ids[test_index]:
                # Create only image vector for testing
                self.test_img_vectors.append(image_corpus_line)
                test_index += 1
            elif train_index >= self.divide_index and test_index >= (self.textual_examples_loaded - self.divide_index):
                break
            images_index += 1
            # Debug print
            if ((images_index + 1) % 4000 == 0):
                print("Read visual features %d" % (images_index + 1))
            # Stop when all textual examples were found.
        print("Done: concatenating visual features to textual")

    def _generateCorpusForImage(self, features):
        image_corpus_line = []
        vector_sum = sum(map(float, features))
        for i in range(0, 4096):
            feature_value = float(features[i]) / vector_sum
            if (feature_value != 0):
                feature_name = "feature"+`(i+1)`
                feature_id = self.word2id[feature_name]
                image_corpus_line.append( (feature_id, feature_value) )
        return image_corpus_line


    def _load_image_ids(self):
        print("Loading image ids")
        with open(os.path.join(image_ids_directory, image_ids_filename)) as image_id_file:
            self.image_ids = image_id_file.read().splitlines()
        print("Done loading image ids.")


    def createVocabulary(self, number_of_examples_to_load, countSkip=100):
        '''
        Read and preprocess textual features. Create vocabulary and corpus.
        :param directory:
        :param filename:
        :param number_of_examples_to_load:
        :return:
        '''
        print("Loading textual features.")
        p_stemmer = PorterStemmer()
        stop = stopwords.words('english')
        number_of_lines = 0
        number_of_words = 0
        lines = DataManager._readVisualFile(number_of_examples_to_load)
        for line in lines:
            # For every line in document.
            words = []
            weights = []
            corpus_line_dict = dict()
            number_of_lines += 1
            line_words = line.split()
            self.text_examples_img_ids.append(line_words[0])
            number_of_features = int(line_words[1])
            for j in range(0, number_of_features, 2):
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

        print("Number of text examples read:", number_of_lines, len(self.text_examples))
        print("Number of unique words:", len(self.unique_words), "from ", number_of_words)

        assert len(self.corpus) == len(self.text_examples_img_ids)
        self.textual_examples_loaded = len(self.text_examples_img_ids)

        #print self.corpus[0]
        #print self.id2word[1]

