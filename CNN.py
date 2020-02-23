import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation, AveragePooling1D
import numpy

#REF: https://keras.io/examples/mnist_cnn/
#https://stackoverflow.com/questions/43396572/dimension-of-shape-in-conv1d/43399308 --> shows the logic to apply
#https://stackoverflow.com/questions/44978768/how-do-i-shape-my-input-data-for-use-with-conv1d-in-keras --> best explanation
#BATCH_NORMALIZATION FITS BETTER ON CNN THAN DROPOUT --> https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html
#GOOD REF: https://issue.life/questions/43235531
#INPUT SHAPE IRIS EXPLANATION: https://datascience.stackexchange.com/questions/36106/input-shape-of-dataset-in-cnn
def cnn(x_train, x_test, y_train, y_test ,batch_size, epochs, filters, kernel_size, stride=1):

    '''
    THIS 4 PARAMETERS ARE ALREADY NORMALIZED (MIN-MAX NORMALIZATION)
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param epochs: integer that represents epochs
    :param filters: integer --> dimensionality of output space(number of output filters in the convolution)
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param stride: by default=1, integer represents stride length of convolution
    :return: score of model: accuracy
    '''

    try:

        #I NEED TO RESHAPE DATA TO: (number samples, time step ,features) --> for this example, time_step is 1, and the reshape format is : (samples, features)
        #input shape in channels last --> (time steps, features), if time step is 1, then (None, features) --> https://keras.io/layers/convolutional/
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) #TIME STEPS = FEATURES AND FEATURES=TIME STEPS
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        #I NEED TO CONVERT TARGETS INTO BINARY CLASS, TO PUT THE TARGETS INTO SAME RANGE OF ACTIVATION OF FUNCTIONS LIKE: SOFTMAX OR SIGMOID
        y_train = keras.utils.to_categorical(y_train, 3)
        y_test = keras.utils.to_categorical(y_test, 3)

        #EXPLANATION BETWEEN PADDING SAME AND VALID: https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        #MODEL CREATION
        input_shape = (x_train.shape[1], 1)
        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=input_shape, padding='valid')) #FIRST CNN
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(strides=1, padding='same')) #I maintain the default value -->  max pool matrix (2.2)
        model.add(Flatten())
        model.add(Dense(3))#FULL CONNECTED LAYER --> OUTPUT LAYER 3 OUTPUTS
        model.add(Activation('softmax'))
        model.summary() #PRINT SUMMARY OF MODEL

        #COMPILE MODEL
        model.compile(optimizer='Adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])  # CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS

        #FIT MODEL
        historyOfTraining = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
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