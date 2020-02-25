import pyswarms
import config
import LSTM_Model

def lostFunction(particleDimension, x_train, x_test, y_train, y_test, batch_size):

    '''
    THIS FUNCTION REPRESENTS LOSS FUNCTION, USED TO EVALUATE POSITION'S OF PARTICLES (VALUES OF HIS DIMENSIONS, LIKE NEURONS AND EPOCHS)
    :param particleDimension: numpy array --> (1, dimensions) representing neurons [0] and epochs [1]
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :return: float --> loss values, that results of the application of cost function (particle cost)
    '''

    try:

        #RETRIEVE DIMENSIONS VALUES, AND I NEED TO CONVERT FLOAT VALUES (CONTINUOUS) TO INT
        neurons = int(particleDimension[0])
        epochs = int(particleDimension[1])

        #CALL LSTM_MODEL function
        accuracy = LSTM_Model.lstm(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                        neurons=neurons, batch_size=batch_size, epochs=epochs)

        #APPLY COST FUNCTION --> THIS FUNCTION IS EQUALS TO CNN COST FUNCTION
        loss = 1.5 * ((1.0 - (1.0/neurons)) + (1.0 - (1.0/epochs))) + 2.0 * (1.0 - accuracy)
        print(accuracy)
        return loss

    except:
        raise

def particlesLoop(particles, x_train, x_test, y_train, y_test, batch_size):

    '''
    THIS IS THE LOOP FUNCTION, USED TO APPLY LSTM BY ALL PARTICLE IN ALL ITERATIONS --> CALL'S LOST FUNCTION AND RETRIEVES COST OF ALL ITERATIONS
    :param particles: numpy array --> shape (particles, dimensions) --> particles[i] represents i particle, particles[i][0] represents first dimension of a particle
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :return: array with shape:(1, iters * particles) --> all particles losses
    '''

    try:

        numberParticles = particles.shape[0] #NUMBER OF PARTICLES

        allLosses = [lostFunction(particleDimension=particles[i], x_train=x_train, x_test=x_test,
                    y_train=y_train, y_test=y_test, batch_size=batch_size) for i in range(numberParticles)]

        return allLosses #NEED TO RETURN THIS PYSWARMS NEED THIS

    except:
        raise

def applyLSTM_PSO(x_train, x_test, y_train, y_test, batch_size, numberParticles, iterations, dimensions, bounds, **kwargs):

    '''
    THIS IS THE FUNCTION RESPONSIBLE TO CONFIGURE PSO AND APPLY PSO TO OPTIMIZE LSTM PARAMETERS
    :param x_train: samples used in train
    :param x_test: samples used in test
    :param y_train: targets used in train
    :param y_test:  targets used in test
    :param batch_size: integer that represents batch size
    :param numberParticles: integer --> number of particles
    :param iterations: integer --> number of iterations
    :param dimensions: integer --> number of dimensions of the problem
    :param bounds: numpy array (minBound, maxBound) range of axis dimensions --> minBound (1, dimensions), maxBound (1, dimensions)
    :param kwargs: pso arguments, like --> type (GBest or LBest) and options (c1, c2, w ...)
    :return cost --> float: best cost obtained in all particles iterations
    :return pos --> numpy array: (1, dimensions) final position of best particle (with best cost)
    :return optimizer --> SwarmOptimizer: need this object to make plot's about the results
    '''

    try:

        #GET KWARG type ARGUMENT
        topology = kwargs.get(config.TYPE)

        #INITIALIZATION OF PSO --> CONSIDERING TWO POSSIBLE TOPOLOGIES gbest AND lbest
        optimizer = None
        if topology == config.GLOBAL_BEST:
            optimizer = pyswarms.single.GlobalBestPSO(n_particles=numberParticles, dimensions=dimensions,
                                                      options=kwargs.get(config.OPTIONS), bounds=bounds)
        elif topology == config.LOCAL_BEST:
            optimizer = pyswarms.single.LocalBestPSO(n_particles=numberParticles, dimensions=dimensions,
                                                      options=kwargs.get(config.OPTIONS), bounds=bounds)
        else:
            raise AttributeError

        #PSO OPTIMIZATION PASSING LOOP PARTICLES ITERATION FUNCTION particlesLoop, applying lstm for all particle in all iterations
        cost, pos = optimizer.optimize(particlesLoop, x_train=x_train, x_test=x_test, y_train=y_train,
                                       y_test=y_test, batch_size=batch_size,iters=iterations)

        return cost, pos, optimizer

    except:
        raise