import keras
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, Flatten, Dropout, Bidirectional
import numpy

#REF
#https://stats.stackexchange.com/questions/365428/difference-between-a-single-unit-lstm-and-3-unit-lstm-neural-network --> VERY GOOD EXPLANATION
#https://stats.stackexchange.com/questions/179101/structure-of-recurrent-neural-network-lstm-gru
#EXPLANATION TIMESTEP AND FEATURES INPUT_SHAPE --> https://datascience.stackexchange.com/questions/33393/understanding-input-of-lstm
#https://datascience.stackexchange.com/questions/25463/questions-about-lstm-cells-units-and-inputs
#https://www.quora.com/What-is-the-relationship-between-timestep-and-number-hidden-unit-in-LSTM
#https://heartbeat.fritz.ai/a-beginners-guide-to-implementing-long-short-term-memory-networks-lstm-eb7a2ff09a27
#https://govind.tech/understanding-stateful-option-in-keras-lstm/ --> LINK MUITO UTIL
def lstm(x_train, x_test, y_train, y_test, neurons, batch_size, epochs):

    '''
    THIS 4 PARAMETERS ARE ALREADY NORMALIZED (MIN-MAX NORMALIZATION)
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param epochs: integer that represents epochs
    :param neurons: number of neurons on lstm layer
    :return: score of model: accuracy
    '''

    try:

        #FIRST, I NEED TO RESHAPE DATA (samples, timesteps, features) --> in this example timesteps = features and
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)#SWAP BETWEEN TIME STEPS AND FEATURES, LIKE HAPPENS IN CNN
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        #I NEED TO CONVERT TARGETS INTO BINARY CLASS, TO PUT THE TARGETS INTO SAME RANGE OF ACTIVATION OF FUNCTIONS LIKE: SOFTMAX OR SIGMOID
        y_train = keras.utils.to_categorical(y_train, 3)
        y_test = keras.utils.to_categorical(y_test, 3)

        #MODEL CREATION
        model = Sequential()
        #batch_input_shape = (batch_size, x_train.shape[1], 1)
        input_shape = (x_train.shape[1], 1)
        model.add(LSTM(neurons, input_shape=input_shape, stateful=False, return_sequences=False))#STATEFUL = FALSE, BECAUSE NO DEPENDENCY BETWEEN DATA, AND RETURN SEQUENCES = FALSE, BECAUSE NOW I DON'T NEED TO CREATE A STACKED LSTM
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        #model.add(Flatten())
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.summary()

        #COMPILE MODEL
        model.compile(optimizer='Adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])  # CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS

        #FIT MODEL
        historyOfTraining = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=False #IF I USE STATEFUL MODE, THIS PARAMETER NEEDS TO BE EQUALS TO FALSE
        )

        predict = model.predict(x=x_test, batch_size=batch_size)
        print(predict)
        print(y_test)

        predict = (predict == predict.max(axis=1)[:, None]).astype(int)
        print(predict)

        numberRights = 0
        for i in range(len(y_test)):
            indexMaxValue = numpy.argmax(predict[i], axis=0)
            if indexMaxValue == numpy.argmax(y_test[i],
                                             axis=0):  # COMPARE INDEX OF MAJOR CLASS PREDICTED AND REAL CLASS
                numberRights = numberRights + 1

        hitRate = numberRights / len(y_test)  # HIT PERCENTAGE OF CORRECT PREVISIONS

        return hitRate

    except:
        raise