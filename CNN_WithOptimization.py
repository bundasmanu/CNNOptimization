import pyswarms
import numpy
import CNN as CNN_Approach
import config
import matplotlib.pyplot as plt

#I USE PSO IN ORDER TO OPTIMIZE TWO PARAMETERS:
    #NUMBER OF FILTERS USED ON CONVOLUTION LAYERS
    #NUMBER OF EPOCHS USED IN TRAINING PROCESS

def optimizeCNN(x_train, x_test, y_train, y_test ,batch_size, kernel_size, particleDimensions ,stride=1):

    '''
    This is loss function applied by all particles in iterations
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param particleDimensions: numpy array with dimensions of a n particle --> 2 dimensions (filters, epochs)
    :param stride: by default=1, integer represents stride length of convolution
    :return: loss --> result of application of loss equation
    '''

    try:

        #RETRIEVE DIMENSIONS OF PARTICLE
        numberFilters = int(particleDimensions[0]) #FLOAT TO INT
        numberEpochs = int(particleDimensions[1])

        #CALL CNN FUNCTION cnn --> RETURN accuracy
        accuracy = CNN_Approach.cnn(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
                                    epochs=numberEpochs, filters=numberFilters, kernel_size=kernel_size, stride=stride)

        #APPLY LOST FUNCTION --> THE MAIN OBJECTIVE IS TO MINIMIZE LOSS --> MAXIMIZE ACCURACY AND AT SAME TIME MINIMIZE THE NUMBER OF EPOCHS
                                #AND FILTERS, TO REDUCE TIME AND COMPUTACIONAL POWER
        loss = 1.5 * ((1.0 - (1.0/numberFilters)) + (1.0 - (1.0/numberEpochs))) + 2.0 * (1.0 - accuracy)
        print(accuracy)
        return loss

    except:
        raise

def particleIteration(particles, x_train, x_test, y_train, y_test ,batch_size, kernel_size ,stride=1):

    '''
    This is function that calls loss function, and returns all losses return by all particles on one iteration
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param stride: by default=1, integer represents stride length of convolution
    :param particles: numpy array --> (particles, dimensions)
    :return: lossArray --> all losses returned by all particles
    '''

    try:

        numberParticles = particles.shape[0]
        allLosses = [optimizeCNN(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
                                  kernel_size=kernel_size, particleDimensions=particles[i], stride=stride)for i in range(numberParticles)]

        return allLosses
    except:
        raise

def callCNNOptimization(x_train, x_test, y_train, y_test ,batch_size, kernel_size, numberParticles, iterations, bounds ,stride=1, **kwargs):

    '''
    This is the function that defines all PSO context and calls loss function for every particles (fill all iterations)
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param kernel_size: integer of tuple with only one integer (integer, ) --> length of convolution window
    :param stride: by default=1, integer represents stride length of convolution
    :param numberParticles: integer --> number of particles of swarm
    :param iterations: integer --> number of iterations
    :param bounds: numpy array (minBound, maxBound) --> minBound: numpyArray - shape(dimensions), maxBound: numpyArray - shape(dimensions)
    :return cost: integer --> minimum loss
    :return pos: numpy array with n dimensions --> [filterValue, epochValue], with best cost (minimum cost)
    :return optimizer: SWARM Optimization Optimizer USED IN DEFINITION AND OPTIMIZATION OF PSO
    '''

    try:

        #GET PSO PARAMETERS
        psoType = kwargs.get(config.TYPE)
        options = kwargs.get(config.OPTIONS)

        #DIMENSIONS OF PROBLEM
        dimensions = 2

        #OPTIMIZER FUNCTION
        if psoType == config.GLOBAL_BEST:
            optimizer = pyswarms.single.GlobalBestPSO(n_particles=numberParticles, dimensions=dimensions,
                                                     options=options, bounds=bounds)
        elif psoType == config.LOCAL_BEST:
            optimizer = pyswarms.single.LocalBestPSO(n_particles=numberParticles, dimensions=dimensions,
                                                     options=options, bounds=bounds)
        else:
            raise AttributeError
        #GET BEST COST AND PARTICLE POSITION
        cost, pos = optimizer.optimize(objective_func=particleIteration, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                       batch_size=batch_size, kernel_size=kernel_size, stride=stride, iters=iterations)

        return cost, pos, optimizer

    except:
        raise