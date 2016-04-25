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


def readVisMatrix(textual_img_ids, image_ids, directory, filename):
    '''
    Construct visual feature vectors
    '''
    print("=================================================================")
    print("Reading visual feature Matrix from images features file")
    numObjects = len(textual_img_ids)
    vMatrix = np.zeros(shape=(numObjects, 4096), dtype='float32')
    with open(os.path.join(directory, filename), 'r') as fVisual:
        description = fVisual.readline()
        numImgs, numFeatures = description.split()
        print "Reading %s training images from %s images, %s features" % (numObjects, numImgs, numFeatures)

        textual_index = 0
        images_index = 0
        for line in fVisual:
            image_id = image_ids[images_index]
            if image_id == textual_img_ids[textual_index]:
                iv = line.split()
                imgID, vector = iv[0], iv[1:]
                assert(imgID == image_id)
                vector = [float(vector[i]) for i in range(len(vector))]
                vMatrix[textual_index] = vector
                textual_index += 1
            images_index += 1

            if (images_index+1)%4000==0:
                print("Read %d visual features from %d images of %d Images to read." % (textual_index, images_index, numObjects))
            if textual_index >= numObjects: break
    print("Done: read visual feature Matrix")

    return vMatrix

def matrix2File(npArray, directory='', filename='imgVisualMatrix.npy'):
    '''
    Read numpy matrix into file
    '''
    print("=================================================================")
    print("Saving visual feature Matrix numpy file")
    t0 = time.time()
    #npArray.tofile(os.path.join(directory, filename))
    for i in range(0, npArray.shape[0]/30000):
        new_npArray = npArray[i*30000:(i+1)*30000]
        np.save(os.path.join(directory, ('%s'%(i*30000))+filename), new_npArray)
    np.save(os.path.join(directory, ('300000'+filename)), npArray[300000:])
    print("Done: save visual feature Matrix (%.3fs)" % (time.time()-t0))


if __name__ == '__main__':
    #matrix2File(readVisMatrix('/Users/enzo/Projects/TBIR/dataset/Features/Visual/', 'scaleconcept16_data_visual_vgg16-relu7.dfeat', 300000))
    t1 = time.time()
    c = np.load("imgVisualMatrix.npy")
    t2 = time.time()
    print("Time to load: %s" % str(t2-t1))
    print("The dimensions are: %s X %s" % (c.shape[0], c.shape[1]))
    print("First line of the matrix:\n%s" % c[0])
