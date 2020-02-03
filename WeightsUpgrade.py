from keras.callbacks import Callback
import numpy

class WeightsUpgrade(Callback):

    def __init__(self, particleWeights):
        self.PWeights = particleWeights

    def on_train_begin(self, logs=None):
        numberLayer = 0 #VARIABLE THAT CONTROLS ITERATIVE LAYERS --> TO UPGRADE THE SPECIFIC WEIGHTS
        newStartPositionOfLayers = 0
        for layer in self.model.layers: #UPDATING THE WEIGHTS OF EACH MODEL LAYER
            numberLayer = numberLayer + 1
            minValue = layer.get_weights()[numberLayer].shape[0]
            maxValue = layer.get_weights()[numberLayer].shape[1]
            allValues = minValue *  maxValue #THIS ONLY RESULTS IF DATASET'S AREN'T THREEDIMENSIONAL
            layer.set_weights = self.PWeights[newStartPositionOfLayers:allValues] #ONLY UPGRADE THE SPECIFIC WEIGHT VALUES FOR THIS LAYER, AND NOT THE ALL WEIGHTS
            newStartPositionOfLayers = allValues #UPGRADE THIS FLAG TO PASS TO NEW LAYER

    def on_batch_end(self, batch, logs=None):
        return None #there is not anything left, if there is a need to update the weights in each batch it may be necessary to override this function, however I think it only makes sense to apply the update of the weights at the beginning of the training and not of each batch
