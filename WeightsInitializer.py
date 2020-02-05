import tensorflow as tf

class WeightsInitializer():

    def __init__(self):
        pass

    #REF--> EXPLANATION TENSOR'S --> https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/refs/heads/0.6.0/tensorflow/g3doc/how_tos/variables/index.md

    def initInputMat(self, inputMat):

        '''
        :param inputMat: numpy array --> kernel input matrix
        :return: a tensor of Kernel Input Matrix weights
        '''

        #REFS: http://www.machinelearningtutorial.net/fullpage/an-introduction-to-tensorflow-variables/
        #https://stackoverflow.com/questions/38805599/why-tf-constant-initializer-does-not-take-a-constant-tensor

        weights = tf.get_variable(name= 'inputWeights',
                                  shape = inputMat.shape,
                                  initializer = tf.constant(inputMat)) #I NEED TO CONFIRM AFTER THE INITIALIZATION OF THIS VALUES ON TRAINING CALLBACK --> I THINK THE ORDER TO DISTRIBUTE THE VALUES ARE CORRECT
        return weights

    def initRecMat(self, recurrentMat):

        '''

        :return: a tensor of Kernel Input Matrix weights
        '''

        weights = tf.get_variable(name= 'recWeights',
                                  shape = recurrentMat.shape,
                                  initializer = tf.constant(recurrentMat)) #I NEED TO CONFIRM AFTER THE INITIALIZATION OF THIS VALUES ON TRAINING CALLBACK --> I THINK THE ORDER TO DISTRIBUTE THE VALUES ARE CORRECT
        return weights