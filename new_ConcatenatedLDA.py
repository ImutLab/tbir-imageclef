import gensim
from DataManager import DataManager
import time, os

class ConcatenatedLDA(object):
    data_manager = None
    lda_model = None

    def __init__(self, data_manager):
        self.data_manager = data_manager

    def trainLDA(self):
        assert isinstance(self.data_manager, DataManager)
        print("\nThe size of corpus: %s\n" % len(self.data_manager.corpus))
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.data_manager.corpus[:-2], id2word=self.data_manager.id2word,
                                                         num_topics= 100, update_every=1, passes=1, chunksize=1000)

        print("Five topic of model:\n%s" % self.lda_model.print_topics(num_topics=5))
        for doc in self.data_manager.corpus[-2:]:
            doc_lda = self.lda_model[doc]
            print("Length of lda of this doc: %s\n%s" % (len(doc_lda), doc_lda))
            for topic in doc_lda:
                word_distr = self.lda_model.get_topic_terms(topic[0])
                print("Length of word distribution: %s\n%s" % (len(word_distr), word_distr))
        # print(self.lda_model.get_topic_terms(0))
        # print(self.lda_model.get_document_topics(self.data_manager.corpus[0]))

    def trainLDA_from_corpus(self, corpus):
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=self.data_manager.id2word, num_topics=100, update_every=1, passes=1, chunksize=1000)

    def updateLDA(self, corpus):
        self.lda_model.update(corpus)

    def save_ldaModel_to_file(self, ldaModel_directory, ldaModel_filename):
        print("=========================================================")
        print("Save lda model to file.")
        self.lda_model.save(os.path.join(ldaModel_directory, ldaModel_filename))
        print("Done")

    def load_ldaModel_from_file(self, ldaModel_directory, ldaModel_filename):
        print("=========================================================")
        print("Load lda model From file.")
        self.lda_model = gensim.models.LdaModel.load(os.path.join(ldaModel_directory, ldaModel_filename))
        print("Done")

