import pyswarms
import CNN as CNN_Approach

#I USE PSO IN ORDER TO OPTIMIZE TWO PARAMETERS:
    #NUMBER OF FILTERS USED ON CONVOLUTION LAYERS
    #NUMBER OF EPOCHS USED IN TRAINING PROCESS

def optimizeCNN(x_train, x_test, y_train, y_test ,batch_size, epochs, filters, kernel_size, stride=1):

    '''
    This is loss function applied by all particles in iterations
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param epochs: integer that represents epochs
    :param filters: integer --> dimensionality of output space(number of output filters in the convolution)
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param stride: by default=1, integer represents stride length of convolution
    :return: loss --> result of application of loss equation
    '''

    try:

        #CALL CNN FUNCTION cnn --> RETURN accuracy
        accuracy = CNN_Approach.cnn(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
                                    epochs=epochs, filters=filters, kernel_size=kernel_size, stride=stride)

        #APPLY LOST FUNCTION --> THE MAIN OBJECTIVE IS TO MINIMIZE LOSS --> MAXIMIZE ACCURACY AND AT SAME TIME MINIMIZE THE NUMBER OF EPOCHS
                                #AND FILTERS, TO REDUCE TIME AND COMPUTACIONAL POWER
        loss = 1.0 * ((1.0 - (1.0/epochs)) + (1.0 - (1.0/filters))) + 2.0 * (1.0 - accuracy)

        return loss

    except:
        raise

def particleIteration(x_train, x_test, y_train, y_test ,batch_size, epochs, filters, kernel_size, stride=1):

    '''
    This is function that calls loss function, and returns all losses return by all particles on one iteration
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param epochs: integer that represents epochs
    :param filters: integer --> dimensionality of output space(number of output filters in the convolution)
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param stride: by default=1, integer represents stride length of convolution
    :return: lossArray --> all losses returned by all particles
    '''

    try:

        return None

    except:
        raise