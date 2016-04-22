import os

class DataManager(object):

    @staticmethod
    def read_visual_file(directory, filename, number_of_examples_to_load):
        with open(os.path.join(directory, filename)) as textual_file:
             for i in range(0,number_of_examples_to_load):
                 yield textual_file.readline()

    @staticmethod
    def create_vocabulary(directory, filename, number_of_examples_to_load):
        # TODO presun sem tu metodu na nacitanie vocabulary
        print("Abc")