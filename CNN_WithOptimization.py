import pyswarms
import numpy
import CNN as CNN_Approach

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
        loss = 1.0 * ((1.0 - (1.0/numberFilters)) + (1.0 - (1.0/numberEpochs))) + 2.0 * (1.0 - accuracy)

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
                                  kernel_size=kernel_size, particles=particles[i], stride=stride)for i in range(numberParticles)]

        return allLosses
    except:
        raise

def callCNNOptimization(x_train, x_test, y_train, y_test ,batch_size, kernel_size, numberParticles, iterations, stride=1):

    try:

        #DEFINITION OF PSO PARAMETERS
        options = {'c1' : 0.3, 'c2' : 0.2, 'w' : 0.9}

        #DIMENSIONS OF PROBLEM
        dimensions = 2

        #DEFINITION OF BOUNDS
        minBound = numpy.ones(2) #MIN BOUND FOR TWO DIMENSIONS IS 1
        maxBound = numpy.ones(2) #ONLY INITIALIZATION
        maxBound[0] = 601 #MAX NUMBER OF FILTERS
        maxBound[1] = 401 #MAX NUMBER OF EPOCHS
        bounds = (minBound , maxBound)

        #OPTIMIZER FUNCTION
        optimizer = pyswarms.single.LocalBestPSO(n_particles=numberParticles, dimensions=dimensions,
                                                 options=options, bounds=bounds)
        #GET BEST COST AND PARTICLE POSITION
        cost, pos = optimizer.optimize(objective_func=particleIteration, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                       batch_size=batch_size, kernel_size=kernel_size, stride=stride, iters=iterations)
        
        return cost, pos

    except:
        raise