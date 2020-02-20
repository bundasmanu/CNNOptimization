import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

def mlp(x_train, x_test, y_train, y_test, numberClasses, numberFeatures, neurons, batch_size, epochs):

    '''
    THIS 4 PARAMETERS ARE ALREADY NORMALIZED (MIN-MAX NORMALIZATION)
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param numberClasses: number of classes of problem, p.e --> IRIS Dataset: 3 classes
    :param numberFeatures: number features of problem, p.e --> IRIS Dataset: 4 features
    :return: score of model (loss , accuracy) --> bidimensional array [1,2]
    '''

    try:

        #I DONT NEED TO RESHAPE DATA BECAUSE --> THIS DATASET IS NOT 2D (LIKE IMAGES)

        #BUT I NEED TO CONVERT TARGETS INTO BINARY CLASS, TO PUT THE TARGETS INTO SAME RANGE OF ACTIVATION FUNCTIONS LIKE: SOFTMAX OR SIGMOID
        y_train = keras.utils.to_categorical(y_train, numberClasses)
        y_test = keras.utils.to_categorical(y_test, numberClasses)

        #NOW I NEED TO BUILD MLP MODEL
        model = Sequential()
        model.add(Dense(neurons, input_shape=(numberFeatures,))) #FULL CONNECTED LAYER
        model.add(Activation('relu')) #ACTIVATION FUNCTION OF FULL CONNECTED LAYER
        model.add(Dropout(rate=0.1)) #LAYER THAT PREVENTS OVERFITTING --> http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
        model.add(Dense(50)) # FULL CONNECTED LAYER 2
        model.add(Activation('relu'))  # ACTIVATION FUNCTION OF FULL CONNECTED LAYER
        model.add(Dropout(rate=0.1))
        model.add(Dense(units=numberClasses))
        model.add(Activation('softmax'))

        #DEFINE PARAMETERS OF MODEL COMPILE --> (model optimizer, loss function, metrics)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy']) #CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS

        #TRAIN MODEL
        historyOfModel =model.fit(
            x=x_train,
            y=y_train,
            epochs=epochs,
            verbose=1, #PROGRESS BAR IS ACTIVE
            batch_size=batch_size,
            validation_split=0.3,
            validation_data=(x_test, y_test) #MODEL DOESN'T USE THIS DATA ON TRAINING, AND I ALSO USE THEM IN PREDICT --> PREVENTS OVERFITTING
        )

        finalScores = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1) #BY DEFAULT BATCH_SIZE IS 32, AND IT'S IMPORTANT TO OVERRIDE THIS

        return finalScores

    except:
        raise