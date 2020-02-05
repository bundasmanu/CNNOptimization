from keras import backend as K

class WeightsInitializer():

    def __init__(self, KernelInputMat, KernelRecurrentMat):
        self.inputMat = KernelInputMat
        self.recMat = KernelRecurrentMat

    def initInputMat(self):

        '''

        :return: a tensor of Kernel Input Matrix weights
        '''

    def initRecMat(self):

        '''

        :return: a tensor of Kernel Input Matrix weights
        '''