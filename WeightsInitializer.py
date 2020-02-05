from keras import backend as K
import tensorflow as tf
import numpy as np

class WeightsInitializer():

    def __init__(self, KernelInputMat, KernelRecurrentMat, numberFeatures, neurons):
        self.inputMat = KernelInputMat
        self.recMat = KernelRecurrentMat
        self.features = numberFeatures
        self.neurons = neurons

    #REF--> EXPLANATION TENSOR'S --> https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/refs/heads/0.6.0/tensorflow/g3doc/how_tos/variables/index.md

    def initInputMat(self):

        '''

        :return: a tensor of Kernel Input Matrix weights
        '''

        #REFS: http://www.machinelearningtutorial.net/fullpage/an-introduction-to-tensorflow-variables/
        #https://stackoverflow.com/questions/38805599/why-tf-constant-initializer-does-not-take-a-constant-tensor

        shapeInputMat = [self.features, (self.neurons*4)] #DEFINITION OF THE SHAPE OF INPUT MATRIX, NEEDED TO PASS TO TENSOR CREATION

        #RESHAPE UNIDIMENSIONAL MATRIX TO BIDIMENSIONAL
        reshapedInputMat = self.initInputMat().reshape(shapeInputMat[0], shapeInputMat[1])

        weights = tf.get_variable(name= 'inputWeights',
                                  shape = shapeInputMat,
                                  initializer = tf.constant(reshapedInputMat)) #I NEED TO CONFIRM AFTER THE INITIALIZATION OF THIS VALUES ON TRAINING CALLBACK --> I THINK THE ORDER TO DISTRIBUTE THE VALUES ARE CORRECT
        return weights

    def initRecMat(self):

        '''

        :return: a tensor of Kernel Input Matrix weights
        '''

        shapeRecurrentMatrix = [self.neurons, (self.neurons * 4)]

        #RESHAPE UNIDIMENSIONAL MATRIX TO BIDIMENSIONAL
        reshapedRecurrentMatrix = self.initRecMat().reshape(shapeRecurrentMatrix[0], shapeRecurrentMatrix[1])

        weights = tf.get_variable(name= 'recWeights',
                                  shape = shapeRecurrentMatrix,
                                  initializer = tf.constant(reshapedRecurrentMatrix)) #I NEED TO CONFIRM AFTER THE INITIALIZATION OF THIS VALUES ON TRAINING CALLBACK --> I THINK THE ORDER TO DISTRIBUTE THE VALUES ARE CORRECT
        return weights