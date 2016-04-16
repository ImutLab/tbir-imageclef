import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import time


images = []
examples = []
examples_weights = []
unique_words = set()

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
    with open(os.path.join(directory, filename)) as textual_file:
        for i in range(0,number_of_examples_to_load):
        #for line in textual_file:
            line = textual_file.readline()
            words = []
            weights = []
            number_of_lines += 1
            line_words = line.split()
            images.append(line_words[0])
            number_of_features = int(line_words[1])
            for j in range(0,number_of_features,2):
                word = line_words[j+2].decode('utf-8')
                word = p_stemmer.stem(word)
                weight = float(line_words[j+3]) / 100000
                # TODO filter 2 char words, better filter digits e.g. 12th
                if (word not in stop) and (word.isdigit() == False):        # (weight > 2000)
                    #word = str(word)        # Because it is unicode - special chars cannot be converted to ascii
                    words.append(word)      # TODO add only unique words
                    unique_words.add(word)
                    weights.append(weight)
            if( (i+1)%1000 == 0 ):
                print("Read %d of %d\n" % ( i+1, number_of_examples_to_load))
            examples.append(words)
            examples_weights.append(weights)
    print(number_of_lines)

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

def loadWord2vecModel(directory, filename):
    '''
    Load Word2Vec model from file.
    :param directory:
    :param filename:
    :return:
    '''
    model = Word2Vec.load_word2vec_format(os.path.join(directory, filename), binary=True)
    # Print a wordsXfeatures matrrix shape
    print(model.syn0.shape)
    print(model["car"])
    return model

def loadImageFeatures(directory, filename):
    #with open(os.path.join(directory, filename)) as textual_file:
    print("A")

def loadImageIds(directory, filename):
    # TODO load file with image ids
    print("A")

def makeFeatureVector(words, word_vector_model, weights, num_features):
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
    for i in range(0, len(words)):
        word = words[i]
        weight = weights[i]
        if word in index2word_set:
            nwords += 1
            # Sum feature vector
            feature_vector = np.add(feature_vector, weight*word_vector_model[word])
    # Divide the result by the number of words to get the average
    if(nwords != 0):
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

def getAvgFeatureVectors(websites, word_vector_model, weights):
    '''
    Given a set of website (each one a list of words) calculate the average feature vector
    for each one and return  a 2D numpy array.
    :param websites:
    :param word_vector_model:
    :return:
    '''
    print("Getting average feature vectors")
    start = time.time() # Start time
    # Get number of features in word2vec model.
    num_features = word_vector_model.syn0.shape[1]
    # Inicialize a counter.
    counter = 0
    # Preallocate a 2D numpy array, for speed
    website_feature_vecs = np.zeros((len(websites), num_features), dtype="float32")
    for i in range(0, len(websites)):
        website = websites[i]
        # Print a status message every 1000th review
        if counter % 1000. == 0 and counter != 0:
           print "Get average vector for %d of %d" % (counter, len(websites))
        website_feature_vecs[counter] = makeFeatureVector(website, word_vector_model, weights[i], num_features)
        counter += 1
    end = time.time()
    print "Time taken for averaging feature vectors: ", end-start, "seconds."
    return website_feature_vecs




# Main Functio
if __name__ == '__main__':
    # Set up data directories
    websites_data_directory = '/media/dmacjam/Data disc/Open data/TBIR/data_6stdpt/Features/Textual/'
    word_to_vec_trained_model_data_directory = '/home/dmacjam/Word2Vec/'

    readTextualFeatures(websites_data_directory, 'train_data.txt', 1000)
    #print(images)
    print(len(examples))
    print(len(examples[0]), examples[0])
    #pprint(examples[:10])
    print(len(unique_words))
    print(len(examples_weights), examples_weights[0])

    #lda_model = trainLda()
    #print(lda_model.print_topics(num_topics=10))

    representAsBow()
    word_vector_model = loadWord2vecModel(word_to_vec_trained_model_data_directory,'vectors.bin')

    examples_average_vectors = getAvgFeatureVectors(examples, word_vector_model, examples_weights)
    print(examples_average_vectors.shape)
    print(examples_average_vectors[0])