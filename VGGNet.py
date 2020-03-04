import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator

#REF UNDERSTAND: http://www.lapix.ufsc.br/ensino/visao/visao-computacionaldeep-learning/deep-learningreconhecimento-de-imagens/

def vggNet(x_train, x_test, y_train, y_test):

    try:

        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))

        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(Dense(4096))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(Dense(4))
        model.add(Activation('softmax'))

        model.summary()

        #OPTIMIZER
        opt = keras.optimizers.RMSprop(learning_rate=0.0001)

        # COMPILE MODEL
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])  # CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS

        #DATA AUGMENTATION
        image_gen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                       samplewise_center=False,  # set each sample mean to 0
                                       featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                       samplewise_std_normalization=False,  # divide each input by its std
                                       zca_whitening=False,  # apply ZCA whitening
                                       zca_epsilon=1e-06,  # epsilon for ZCA whitening
                                       rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                                       # randomly shift images horizontally (fraction of total width)
                                       width_shift_range=0.1,
                                       # randomly shift images vertically (fraction of total height)
                                       height_shift_range=0.1,
                                       shear_range=0.,  # set range for random shear
                                       zoom_range=0.,  # set range for random zoom
                                       channel_shift_range=0.,  # set range for random channel shifts
                                       # set mode for filling points outside the input boundaries
                                       fill_mode='nearest',
                                       cval=0.,  # value used for fill_mode = "constant"
                                       horizontal_flip=True,  # randomly flip images
                                       vertical_flip=False,  # randomly flip images
                                       # set rescaling factor (applied before any other transformation)
                                       rescale=None,
                                       # set function that will be applied on each input
                                       preprocessing_function=None,
                                       # image data format, either "channels_first" or "channels_last"
                                       data_format=None,
                                       # fraction of images reserved for validation (strictly between 0 and 1)
                                       validation_split=0.0)

        image_gen.fit(x_train)

        model.fit_generator(image_gen.flow(
            x=x_train,
            y=y_train,
            batch_size=32),
            epochs=200,
            validation_data=(x_test, y_test),
            workers=4)

        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        return scores[1] #ACCURACY

    except:
        raise