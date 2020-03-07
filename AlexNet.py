import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from bitstring import BitArray
from typing import List
import plots
import config

#REF UNDERSTAND: http://www.lapix.ufsc.br/ensino/visao/visao-computacionaldeep-learning/deep-learningreconhecimento-de-imagens/
#ARQUITECHTURE CONFIGURATION: https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/


def alexNet(particleDimensions, x_train, x_test, y_train, y_test):

    try:

        #CONVERT FLOAT DATA --> IF I USE PSO TO OPTIMIZE CNN MODEL
        particleDimensions = [int(particleDimensions[i]) for i in range(len(particleDimensions))]

        model = Sequential()
        model.add(Conv2D(filters=particleDimensions[0], kernel_size=(3,3), padding='same', #zero padding --> default 96 neurons
                         input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=particleDimensions[1], kernel_size=(3,3), padding='same')) #default 256 neurons
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=particleDimensions[2], kernel_size=(3,3), padding='same')) #default 384 neurons
        model.add(Activation('relu'))
        model.add(Conv2D(filters=particleDimensions[2], kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(particleDimensions[3])) #default 2048
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(particleDimensions[4])) #default 2048
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # model.add(Dense(particleDimensions[5])) #default 1000
        # model.add(Activation('relu'))
        # model.add(Dropout(0.4))

        model.add(Dense(4))
        model.add(Activation('softmax'))

        model.summary()

        #OPTIMIZER
        opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

        # COMPILE MODEL
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])  # CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS

        #DATA AUGMENTATION
        # image_gen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
        #                                samplewise_center=False,  # set each sample mean to 0
        #                                featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #                                samplewise_std_normalization=False,  # divide each input by its std
        #                                zca_whitening=False,  # apply ZCA whitening
        #                                zca_epsilon=1e-06,  # epsilon for ZCA whitening
        #                                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        #                                # randomly shift images horizontally (fraction of total width)
        #                                width_shift_range=0.1,
        #                                # randomly shift images vertically (fraction of total height)
        #                                height_shift_range=0.1,
        #                                shear_range=0.,  # set range for random shear
        #                                zoom_range=0.,  # set range for random zoom
        #                                channel_shift_range=0.,  # set range for random channel shifts
        #                                # set mode for filling points outside the input boundaries
        #                                fill_mode='nearest',
        #                                cval=0.,  # value used for fill_mode = "constant"
        #                                horizontal_flip=True,  # randomly flip images
        #                                vertical_flip=False,  # randomly flip images
        #                                # set rescaling factor (applied before any other transformation)
        #                                rescale=None,
        #                                # set function that will be applied on each input
        #                                preprocessing_function=None,
        #                                # image data format, either "channels_first" or "channels_last"
        #                                data_format=None,
        #                                # fraction of images reserved for validation (strictly between 0 and 1)
        #                                validation_split=0.0)
        #
        # image_gen.fit(x_train)
        #
        # history = model.fit_generator(image_gen.flow(
        #     x=x_train,
        #     y=y_train,
        #     batch_size=64),
        #     epochs=30,
        #     validation_data=(x_test, y_test),
        #     workers=6,
        #     shuffle=True,
        # )

        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=32,
            epochs=55,
            validation_data=(x_test, y_test),
            workers=4,
            shuffle=True,
        )

        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        loss = 1.5 * ((1.0 - (1.0 / (particleDimensions[0]+particleDimensions[1]+particleDimensions[2])))
                      + (1.0 - (1.0 / (particleDimensions[3]+particleDimensions[4])))) + 5.0 * (1.0 - scores[1])

        plots.plotTrainValLoss(history)
        plots.plotTrainValAcc(history)

        #SAVE MODEL TO FILE --> IN ORDER TO KEP TRAINING MODEL AFTER THAT
        #model.save(config.SAVED_MODEL_FILE2)
        #del model

        return loss

    except:
        raise

def alexNetAugmentation(particleDimensions, x_train, x_test, y_train, y_test, loadModel):

    try:

        #CONVERT FLOAT DATA --> IF I USE PSO TO OPTIMIZE CNN MODEL
        particleDimensions = [int(particleDimensions[i]) for i in range(len(particleDimensions))]

        #OPTIMIZER
        opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

        # COMPILE MODEL
        loadModel.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])  # CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS

        #DATA AUGMENTATION
        image_gen = ImageDataGenerator(
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       rotation_range=10,
                                       zoom_range=0.1,  # set range for random zoom
                                       horizontal_flip=True,)
                                       #validation_split=0.30  # randomly flip images
                                       #featurewise_center=True

        image_gen.fit(x_train)

        #REF: https://stackoverflow.com/questions/53808335/data-augmentation-in-validation
        #https: // stackoverflow.com / questions / 41174546 / keras - data - augmentation - parameters
        train_generator = image_gen.flow(x_train, y_train, batch_size=64)
        #validation_generator = image_gen.flow(x_test, y_test, batch_size=32)

        history = loadModel.fit_generator(
            generator=train_generator,
            validation_data=(x_test, y_test),
            epochs=60,
            #use_multiprocessing=True,
            steps_per_epoch=x_train.shape[0]/64,
            validation_steps=x_test.shape[0]/16,
            workers=4,
            shuffle=True,
        )

        scores = loadModel.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        loss = 1.5 * ((1.0 - (1.0 / (particleDimensions[0]+particleDimensions[1]+particleDimensions[2])))
                      + (1.0 - (1.0 / (particleDimensions[4]+particleDimensions[5])))) + 5.0 * (1.0 - scores[1])

        plots.plotTrainValLoss(history)
        plots.plotTrainValAcc(history)

        return loss

    except:
        raise

def alexNetForGA(ga_solution, x_train, x_test, y_train, y_test):

    try:

        conv1 = BitArray(ga_solution[0:6]).uint
        conv2 = BitArray(ga_solution[6:13]).uint
        conv3 = BitArray(ga_solution[13:21]).uint
        dense1 = BitArray(ga_solution[21:29]).uint
        dense2 = BitArray(ga_solution[29:35]).uint
        #dense3 = BitArray(ga_solution[47:57]).uint

        if conv1 == 0 or conv2 ==0 or conv3 == 0 or dense1 == 0 or dense2 == 0: #if any of the values is 0, it immediately penalises the individual
            return 100,

        model = Sequential()
        model.add(Conv2D(filters=conv1, kernel_size=(3,3), padding='same', strides=2, #zero padding --> default 96 neurons
                         input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=conv2, kernel_size=(3,3), padding='same')) #default 256 neurons
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=conv3, kernel_size=(3,3), padding='same')) #default 384 neurons
        model.add(Activation('relu'))
        model.add(Conv2D(filters=conv3, kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(dense1)) #default 4096
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(dense2)) #default 2048
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # model.add(Dense(dense3)) #default 1000
        # model.add(Activation('relu'))
        # model.add(Dropout(0.3))

        model.add(Dense(4))
        model.add(Activation('softmax'))

        model.summary()

        #OPTIMIZER
        opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

        # COMPILE MODEL
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])  # CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS

        #DATA AUGMENTATION
        # image_gen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
        #                                samplewise_center=False,  # set each sample mean to 0
        #                                featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #                                samplewise_std_normalization=False,  # divide each input by its std
        #                                zca_whitening=False,  # apply ZCA whitening
        #                                zca_epsilon=1e-06,  # epsilon for ZCA whitening
        #                                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        #                                # randomly shift images horizontally (fraction of total width)
        #                                width_shift_range=0.1,
        #                                # randomly shift images vertically (fraction of total height)
        #                                height_shift_range=0.1,
        #                                shear_range=0.,  # set range for random shear
        #                                zoom_range=0.,  # set range for random zoom
        #                                channel_shift_range=0.,  # set range for random channel shifts
        #                                # set mode for filling points outside the input boundaries
        #                                fill_mode='nearest',
        #                                cval=0.,  # value used for fill_mode = "constant"
        #                                horizontal_flip=True,  # randomly flip images
        #                                vertical_flip=False,  # randomly flip images
        #                                # set rescaling factor (applied before any other transformation)
        #                                rescale=None,
        #                                # set function that will be applied on each input
        #                                preprocessing_function=None,
        #                                # image data format, either "channels_first" or "channels_last"
        #                                data_format=None,
        #                                # fraction of images reserved for validation (strictly between 0 and 1)
        #                                validation_split=0.0)
        #
        # image_gen.fit(x_train)
        #
        # model.fit_generator(image_gen.flow(
        #     x=x_train,
        #     y=y_train,
        #     batch_size=32),
        #     epochs=5,
        #     validation_data=(x_test, y_test),
        #     workers=6,
        #     shuffle=True)

        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=32,
            epochs=30,
            validation_data=(x_test, y_test),
            workers=4,
            shuffle=True,
        )

        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        loss = 1.5 * ((1.0 - (1.0 / (conv1+conv2+conv3)))
                      + (1.0 - (1.0 / (dense1+dense2)))) + 5.0 * (1.0 - scores[1])

        return loss,

    except:
        raise
