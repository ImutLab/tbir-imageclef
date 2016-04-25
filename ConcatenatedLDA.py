import gensim
from DataManager import DataManager
import numpy as np
from scipy import linalg

lda_model_filename = '/media/dmacjam/Data disc/Open data/lda-model-X'

class ConcatenatedLDA(object):
    data_manager = None
    lda_model = None

    def __init__(self, data_manager):
        self.data_manager = data_manager

    def trainLDA(self):
        assert isinstance(self.data_manager, DataManager)
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.data_manager.corpus, id2word=self.data_manager.id2word,
                                                         num_topics= 10, update_every=1, passes=1, chunksize=2000)

        # Save the model.
        self.lda_model.save(lda_model_filename)

    def loadLDAModel(self):
        self.lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_filename)


    def inferNewDocuments(self):
        print(self.lda_model.print_topics(num_topics=5))
        (gamma_img, sstats) = self.lda_model.inference(self.data_manager.test_img_vectors)
        print gamma_img.shape, gamma_img[0]
        (gamma_txt, sstats) = self.lda_model.inference(self.data_manager.test_text_corpus)
        print gamma_txt.shape, gamma_txt[0]

        gamma_img = np.array(gamma_img)
        gamma_txt = np.array(gamma_txt)

        results = np.dot(gamma_txt, np.transpose(gamma_img)) /linalg.norm(gamma_txt)/linalg.norm(gamma_img)
        print results.shape, results[0]

        for i, result in enumerate(results):
            max_indices = np.argsort(result)[::-1]
            predicted_img_ids = self.data_manager.test_img_vectors_ids[max_indices]
            print("Example #",i,"with top similarities:", predicted_img_ids[:10])


        #for doc in self.data_manager.corpus[-2:]:
        #    doc_lda = self.lda_model[doc]
        #    print (len(doc_lda), doc_lda)
        #    for topic in doc_lda:
        #        word_distr = self.lda_model.get_topic_terms(topic[0], topn=self.data_manager.unique_words_counter)
        #        print len(word_distr)
        # print(self.lda_model.get_topic_terms(0))
        # print(self.lda_model.get_document_topics(self.data_manager.corpus[0]))

        # TODO create dataset of other images
        # TODO apply lda for topic distribution on that images - inference method
        # TODO sum over first equation

