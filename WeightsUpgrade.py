from keras.callbacks import Callback
import numpy

class WeightsUpgrade(Callback):

    def __init__(self, particleWeights, numberOfNeurons):
        self.PWeights = particleWeights
        self.nNeurons = numberOfNeurons

    def on_train_begin(self, logs={}):

        #Ref: https://stackoverflow.com/questions/42861460/how-to-interpret-weights-in-a-lstm-layer-in-keras

        numberLayer = 0 #VARIABLE THAT CONTROLS ITERATIVE LAYERS --> TO UPGRADE THE SPECIFIC WEIGHTS
        newStartPositionOfLayers = 0

        for layer in self.model.layers: #UPDATING THE WEIGHTS OF EACH MODEL LAYER

            W = layer.get_weights[0]
            U = layer.get_weights[1]

            #EXTRACT ALL MATRICES OF U AND V
            W_i = W[:, :self.nNeurons]
            W_f = W[:, self.nNeurons: self.nNeurons * 2]
            W_c = W[:, self.nNeurons * 2: self.nNeurons * 3]
            W_o = W[:, self.nNeurons * 3:]

            U_i = U[:, :self.nNeurons]
            U_f = U[:, self.nNeurons: self.nNeurons * 2]
            U_c = U[:, self.nNeurons * 2: self.nNeurons * 3]
            U_o = U[:, self.nNeurons * 3:]

            '''
                BIAS --> IN THIS EXAMPLE I DON'T EXPLORE OPTIMIZATION OF BIAS, BUT I PUT HERE THE MATRICES OF BIAS
                b_i = b[:units]
                b_f = b[units: units * 2]
                b_c = b[units * 2: units * 3]
                b_o = b[units * 3:]
            '''

            print(W_i)
            print(W_f)
            print(W_c)
            print(W_o)
            print(U_i)
            print(U_f)
            print(U_c)
            print(U_o)
            #layer.set_weights = self.PWeights[newStartPositionOfLayers:allValues] #ONLY UPGRADE THE SPECIFIC WEIGHT VALUES FOR THIS LAYER, AND NOT THE ALL WEIGHTS
            newStartPositionOfLayers = newStartPositionOfLayers + 1 #UPGRADE THIS FLAG TO PASS TO NEW LAYER

    def on_train_end(self, logs={}):
        print(logs)

    def on_batch_end(self, batch, logs=None):
        return None #there is not anything left, if there is a need to update the weights in each batch it may be necessary to override this function, however I think it only makes sense to apply the update of the weights at the beginning of the training and not of each batch
