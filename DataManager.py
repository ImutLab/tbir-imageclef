import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

websites_data_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Textual/'
websites_filename = 'train_data.txt'


class DataManager(object):
    text_examples = []
    text_examples_weights = []
    id2word = dict()
    word2id = dict()
    unique_words = set()
    text_examples_img_ids = []
    corpus = []

    def __init__(self):
        pass

    @staticmethod
    def _readVisualFile(number_of_examples_to_load):
        with open(os.path.join(websites_data_directory, websites_filename)) as textual_file:
             for i in range(0, number_of_examples_to_load):
                 yield textual_file.readline()

    def createVocabulary(self, number_of_examples_to_load):
        '''
        Read and preprocess textual features. Create vocabulary and corpus.
        :param directory:
        :param filename:
        :param number_of_examples_to_load:
        :return:
        '''
        p_stemmer = PorterStemmer()
        stop = stopwords.words('english')
        number_of_lines = 0
        number_of_words = 0
        unique_words_counter = 0
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
                if (word not in stop) and (word.isdigit() == False):  # (weight > 2000)
                    if (word not in self.unique_words):
                        # words.append(word)
                        self.unique_words.add(word)
                        self.id2word[unique_words_counter] = word
                        self.word2id[word] = unique_words_counter
                        corpus_line_dict[unique_words_counter] = weight
                        unique_words_counter += 1
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

        print self.corpus[0]
        print self.id2word[1]

