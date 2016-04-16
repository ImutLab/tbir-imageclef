import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from pprint import pprint


images = []
examples = []
unique_words = set()

def read_textual_features(directory, filename, number_of_examples_to_load):
    '''
    Read and preprocess textual features.
    :param directory:
    :param filename:
    :return:
    '''
    global unique_words, examples
    p_stemmer = PorterStemmer()
    stop = stopwords.words('english')
    number_of_lines = 0
    with open(os.path.join(directory, filename)) as textual_file:
        for i in range(0,number_of_examples_to_load):
        #for line in textual_file:
            line = textual_file.readline()
            words = []
            number_of_lines += 1
            line_words = line.split()
            images.append(line_words[0])
            number_of_features = int(line_words[1])
            for i in range(0,number_of_features,2):
                word = line_words[i+2].decode('utf-8')
                word = p_stemmer.stem(word)
                weight = int(line_words[i+3])
                if (word not in stop) and (word.isdigit() == False):        # (weight > 2000)
                    #word = str(word)        # Because it is unicode - special chars cannot be converted to ascii
                    words.append(word)      # TODO add only unique words
                    unique_words.add(word)
            examples.append(words)
    print(number_of_lines)
    #words.shape = (number_of_lines, words.shape[0]/number_of_lines)

def train_lda():
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


# Main Functio
if __name__ == '__main__':
    # Data directory of train file
    data_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Textual/'
    read_textual_features(data_directory, 'train_data.txt', 100)
    #print(images)
    print(len(examples))
    print(len(examples[0]), examples[0])
    #pprint(examples[:10])
    print(len(unique_words))

    lda_model = train_lda()
    print(lda_model.print_topics(num_topics=10))