'''
Usage:
    use numpy matrix directly by method: readVisMatrix
    if you want to put the matrix into file
    use method: matrix2File
    you can use the filename you want
    but the filename has to end with '.npy'
    load matrix from file by numpy method: numpy.load()
'''

import numpy as np
import os
import time


def readVisMatrix(directory, filename, numObjects):
    '''
    Construct visual feature vectors
    '''
    vMatrix = np.zeros(shape=(numObjects, 4096), dtype='f')
    with open(os.path.join(directory, filename), 'r') as fVisual:
        description = fVisual.readline()
        numImgs, numFeatures = description.split()
        print "%s images, %s features" % (numImgs, numFeatures)

        index = 0
        for line in fVisual:
            iv = line.split()
            imgID, vector = iv[0], iv[1:]
            vector = [float(vector[i]) for i in range(len(vector))]
            vMatrix[index] = vector

            index += 1
            if index%5000==0: print("Read %d of %d Images" % (index, numObjects))
            if index >= numObjects: break

    return vMatrix

def matrix2File(npArray, directory='', filename='imgVisualMatrix.npy'):
    '''
    Read numpy matrix into file
    '''
    #npArray.tofile(os.path.join(directory, filename))
    np.save(os.path.join(directory, filename), npArray)


if __name__ == '__main__':
    #matrix2File(readVisMatrix('/Users/enzo/Projects/TBIR/dataset/Features/Visual/', 'scaleconcept16_data_visual_vgg16-relu7.dfeat', 300000))
    t1 = time.time()
    c = np.load("imgVisualMatrix.npy")
    t2 = time.time()
    print("Time to load: %s" % str(t2-t1))
    print("The dimensions are: %s X %s" % (c.shape[0], c.shape[1]))
    print("First line of the matrix:\n%s" % c[0])
