import tensorflow as tf

class WeightsInitializer():

    def __init__(self, inputMatrix, recurrentMatrix):
        self.inputMat = inputMatrix
        self.recurrentMat = recurrentMatrix

    #REF--> EXPLANATION TENSOR'S --> https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/refs/heads/0.6.0/tensorflow/g3doc/how_tos/variables/index.md

    def initInputMat(self, shape, dtype=None):

        '''
        :param inputMat: numpy array --> kernel input matrix
        :return: a tensor of Kernel Input Matrix weights
        '''

        #REFS: http://www.machinelearningtutorial.net/fullpage/an-introduction-to-tensorflow-variables/
        #https://stackoverflow.com/questions/38805599/why-tf-constant-initializer-does-not-take-a-constant-tensor
        #https://github.com/tensorlayer/tensorlayer/issues/373 --> DIFFERENCE BETWEEN CONSTANT AND CONSTANT_INITIALIZER
        #IN TENSORFLOW 2.0 GET_VARIABLE IS DEPRECATED, I NEED TO USE A COMPATIBLE APPROACH --> https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable

        init = tf.compat.v1.constant_initializer(self.inputMat) #https://www.tensorflow.org/api_docs/python/tf/constant_initializer
        weights = tf.compat.v1.get_variable(name= 'inputWeights',
                                  shape = [self.inputMat.shape[0], self.inputMat.shape[1]],
                                  initializer = init) #I NEED TO CONFIRM AFTER THE INITIALIZATION OF THIS VALUES ON TRAINING CALLBACK --> I THINK THE ORDER TO DISTRIBUTE THE VALUES ARE CORRECT
        return weights

    def initRecMat(self, shape, dtype=None):

        #EXPLANATION OF INITIALIZER FUNCTIONS TEMPLATE --> https://stackoverflow.com/questions/44663625/how-to-initialize-a-convolution-layer-with-an-arbitrary-kernel-in-keras
        '''

        :return: a tensor of Kernel Input Matrix weights
        '''

        init = tf.compat.v1.constant_initializer(self.recurrentMat)
        weights = tf.compat.v1.get_variable(name= 'recWeights',
                                  shape = [self.recurrentMat.shape[0], self.recurrentMat.shape[1]],
                                  initializer = init) #I NEED TO CONFIRM AFTER THE INITIALIZATION OF THIS VALUES ON TRAINING CALLBACK --> I THINK THE ORDER TO DISTRIBUTE THE VALUES ARE CORRECT
        return weights