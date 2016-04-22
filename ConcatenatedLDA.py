import gensim
from DataManager import DataManager

class ConcatenatedLDA(object):
    data_manager = None
    lda_model = None

    def __init__(self, data_manager):
        self.data_manager = data_manager

    def trainLDA(self):
        assert isinstance(self.data_manager, DataManager)
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.data_manager.corpus[:-2], id2word=self.data_manager.id2word,
                                                         num_topics= 100, update_every=1, passes=1, chunksize=1000)

        print(self.lda_model.print_topics(num_topics=5))
        for doc in self.data_manager.corpus[-2:]:
            doc_lda = self.lda_model[doc]
            print (len(doc_lda), doc_lda)
            for topic in doc_lda:
                word_distr = self.lda_model.get_topic_terms(topic[0])
                print len(word_distr), word_distr
        # print(self.lda_model.get_topic_terms(0))
        # print(self.lda_model.get_document_topics(self.data_manager.corpus[0]))

