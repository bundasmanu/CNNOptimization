import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pyswarms as ps
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Dropout, Conv2D, Activation, BatchNormalization, MaxPooling2D
import keras
import WeightsUpgradeOnTraining, WeightsInitializer
import MLP
import CNN
import LSTM_Model
import CNN_WithOptimization
import plots
import config
import LSTM_PSO
from operator import itemgetter
from itertools import zip_longest
from keras.preprocessing.image import ImageDataGenerator
from deap import base, creator, tools, algorithms
from bitstring import BitArray
from scipy.stats import bernoulli
import AlexNet, VGGNet
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #MAKES MORE FASTER THE INITIAL SETUP OF GPU --> WARNINGS INITIAL STEPS IS MORE QUICKLY
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #THIS LINE DISABLES GPU OPTIMIZATION
from keras.datasets import cifar10

def getDataset(testSize):

    '''
    :param testSize:  integer value between 0-100
    :return: six dataset's --> original input dataset, original output dataset, train dataset(input and output) and test dataset(input and output)
    '''

    trainPercentage = testSize / 100
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    #APPLY NORMALIZATION TO ATTRIBUTES --> EXPLANATION OF THIS APPROACH ON FUNCION
    X = applyNormalization(X)

    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=trainPercentage)

    return X,Y,x_train, x_test, y_train, y_test

def applyNormalization(X):

    '''
    I make a pre analysis and i conclude that the diferent attributes have distant scales
    and in order to optimize the neural network learning, i have decided to apply the min-
    max technique, a normalization technique
    :param X: data of dataset
    :return: X normalized
    '''

    scaler = MinMaxScaler()

    #FIT AND TRANSFORM - X
    scaler.fit(X)
    X = scaler.transform(X)

    return X

'''
    CONVOLUTION NEURAL NETWORK OPTIMIZATION USING PSO (CNN)
'''

def getBestNumberOfNodesAndKernelForCNN(X_train, X_test, Y_train, Y_test, params):

    '''
    The objetive of this function is described in objectiveFunctionPSO() function
    Ref: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
    :param X_train: array --> samples for train
    :param X_test: array --> samples for test
    :param Y_train: array --> samples for train
    :param Y_test: array --> samples for test
    :param params: represents particle parameters --> 2 parameters --> first one input node value and second one kernel length --> unidimensional array
    :return: error (prevision of samples) of a Particle
    '''

    #TRANSFORM DOUBLE VALUES OF PARTICLE DIMENSION FROM DOUBLE TO INTEGER --> PSO USES DOUBLE VALUES
    params = [int(round(params[i])) for i in range(len(params))] #ROUND FUNCTION WAS USED, IN ORDER TO AVOID DOWN UP ROUND'S, IF I DIDN'T CONSIDERED ROUND THIS VALUES ARE DOWN UP (1,6) --> 1, AND I WANT (1,6) --> 2

    print(params[0])
    print(params[1])

    #RESHAPE DOS DADOS DE TREINO
    X_train = X_train.reshape(len(X_train), 4, 1)
    X_test = X_test.reshape(len(X_test), 4, 1)

    #CONVERTION OF VECTOR OUTPUT CLASSES TO BINARY
    Y_train = keras.utils.to_categorical(Y_train, 3)
    Y_test = keras.utils.to_categorical(Y_test, 3)

    #EXPLANATION OF INPUT_SHAPE: input_shape needs only the shape of a sample: (timesteps,data_dim)
    #MODEL CREATION --> SEQUENTIAL OPTION, PERMITES TO CREATES A BUILD OF A CNN MODEL
    model = Sequential()
    model.add(Conv1D(params[0], 2, activation='relu', input_shape=(4,1)))
    model.add(MaxPooling1D(pool_size= 1)) #PODIA TER FEITO APENAS MAXPOOLING E TER DEFINIDO UM VALOR PARA A MATRIX, MAS COMO O EXEMPLO É SIMPLES PENSO QUE ASSIM É MELHOR
    model.add(Flatten())
    model.add(Dense(3, activation='softmax')) #THREE THE NUMBER OF OUPUTS OF PROBLEM --> FULLY CONNECTED LAYER
    model.summary()
    #COMPILE THE MODEL --> 3 ATTRIBUTES https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
    #ADAM IS USED TO CONTROL THE RATE LEARNING OF WEIGHTS OF CNN
    #‘categorical_crossentropy’ for our loss function
    #NAO PRECISAVA DE USAR NADA DISTO --> SERVE APENAS PARA MELHOR ANÁLISE
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X_train, Y_train, epochs=1)

    predictions = model.predict(X_test)  # RETURNS A NUMPY ARRAY WITH PREDICTIONS

    # WELL, I NEED TO COMPARE THE PREDICTIONS WITH REAL VALUES
    numberRights = 0
    for i in range(len(Y_test)):
        indexMaxValue = numpy.argmax(predictions[i], axis=0)
        if indexMaxValue == numpy.argmax(Y_test[i], axis=0): #COMPARE INDEX OF MAJOR CLASS PREDICTED AND REAL CLASS
            numberRights = numberRights + 1

    hitRate = numberRights / len(Y_test)  # HIT PERCENTAGE OF CORRECT PREVISIONS
    print(hitRate)

    #LOSS FUNCTION --> I VALORIZE PARTICLES IF MINOR VALUES OF NODES AND KERNEL'S BUT I PUT MORE IMPORTANCE IN PARTICLES THAT GIVE MORE ACCURACY RATE
    loss = (1.0 * (1.0 - (1/params[0]))) + (0.5 * (1.0 - (1/params[1]))) + (2.0 * (1- hitRate)) #I GIVE THIS WEIGHTS IN ORDER TO OBTAIN GOOD SOLUTIONS, WITH LOW VALUE PARAMETERS, THUS REDUCING COMPUTATIONAL POWER

    print(loss)

    return loss

def objectiveFunctionPSO(particles, X_train, X_test, Y_train, Y_test):

    '''
    This function represents the objetive function used in the calculus
    of the error of application of a CNN. This objetive function considers
    the nodes dimension and kernel evaluation. All particles have specific
    values of this two attributes, and the main objetive is to find the best
    combination of this two attributes, minimizing the error of prevision (I
    considered a simple dataset).
    :param particles: numpy array of shape (nParticles, dimensions)
    :param X_train: array --> samples for train
    :param X_test: array --> samples for test
    :param Y_train: array --> samples for train
    :param Y_test: array --> samples for test
    :return: best cost (minor error) and best Particle founded
    '''

    nParticles = particles.shape[0] #number of Particles
    particleLoss = [getBestNumberOfNodesAndKernelForCNN(X_train, X_test, Y_train, Y_test, particles[i])for i in range(nParticles)]
    return particleLoss

'''
    LONG SHORT-TERM MEMORY OPTIMIZATION USING PSO (LSTM) --> AFTER THAT I COULD CREATE AN EXAMPLE USING THIS TWO TECHNIQUES SIMULTANEOUSLY
'''

def separationWeights(particleWeights, neurons, features):

    '''
    The main objective of this function is to separe a unidimensional array (particle all dimensions) into 2 numpy array's
    represented respectively the kernel and recurrent matrices
    :param particleWeights: array of dimensions of particle
    :param neurons: number of neurons used on LSTM Layer
    :param features: number of features of problem
    :return: 2 numpy array's --> one with kernel matrix with shape (features, (neurons * 4)) and second recurrent kernel with shape (neurons, (neurons * 4))
    '''

    #DEFINITION OF SHAPES OF TWO MATRICES (KERNEL AND RECURRENT MATRICES)
    shapeKernelMatrix = [features, (neurons * 4)]
    shapeRecurrentMatrix = [neurons , (neurons * 4)]

    #DIVISION OF PARTICLES WEIGHTS INTO TWO UNIDIMENSIONAL ARRAY'S --> CONSIDERING THE LENGTH OF TWO MATRICES (KERNEL AND RECURRENT MATRICES)
    unidimensionalKernelMatrix = particleWeights[0: (features * (4 * neurons))] #STARTS ON 0 POSITION AND ENDS ON (features * (4 * neurons))
    unidimensionalRecurrentMatrix = particleWeights[(features * (4 * neurons)):] #STARTS IN THE FINAL POSITION OF KERNEL MATRIX AND TERMINES ON LAST POSITION OF PARTICLES WEIGHT

    #RESHAPE THIS MATRICES
    kernelMatrix = unidimensionalKernelMatrix.reshape(shapeKernelMatrix[0], shapeKernelMatrix[1])
    recurrentMatrix = unidimensionalRecurrentMatrix.reshape(shapeRecurrentMatrix[0], shapeRecurrentMatrix[1])

    return kernelMatrix, recurrentMatrix

def accuracyTimeStempsGreaterThan1(y_test, predictions):

    '''
    This function is used to calculate accuracy between targets values and lstm predictions
    this function considers the approach used when timestemps is greater than 1 (resolution is different from = 1)
    :param y_test: true results of samples
    :param predictions: predictions that are obtained with lstm model
    :return: float value, accuracy value
    '''

    numberRights = 0
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            indexMaxValue = numpy.argmax(predictions[i][j], axis=0)
            if indexMaxValue == numpy.argmax(y_test[i][j], axis=0): #COMPARE INDEX OF MAJOR CLASS PREDICTED AND REAL CLASS
                numberRights = numberRights + 1

    hitRate = numberRights / (y_test.shape[0]*y_test.shape[1])  # HIT PERCENTAGE OF CORRECT PREVISIONS

    return hitRate

def accuracyTimeStempsEqual1(y_test, predictions):

    '''
    This function is used to calculate accuracy between targets values and lstm predictions
    this function considers the approach used when timestemps is equal to 1
    :param y_test: true results of samples
    :param predictions: predictions that are obtained with lstm model
    :return: float value, accuracy value
    '''

    numberRights = 0
    for i in range(y_test.shape[0]):
        indexMaxValue = numpy.argmax(predictions[i], axis=0) #MAX VALUE ON LINE
        if indexMaxValue == numpy.argmax(y_test[i], axis=0): #COMPARE INDEX OF MAJOR CLASS PREDICTED AND REAL CLASS
            numberRights = numberRights + 1

    hitRate = numberRights / y_test.shape[0]  # HIT PERCENTAGE OF CORRECT PREVISIONS

    return hitRate

def objectiveFunctionLSTM(x_train, x_test, y_train, y_test, neurons, batch_size, time_stemps, features, particleWeights):

    '''

    :param X_train: array --> samples for train
    :param X_test: array --> samples for test
    :param Y_train: array --> samples for train
    :param Y_test: array --> samples for test
    :param neurons: number of neurons used in LSTM, this value is defined before
    :param batch_size: length of batch_size, this value is defined before
    :return: error (prevision of samples) of a Particle
    '''

    #EXPLANATION OF BATCH_SHAPE: batch_input_shape needs the size of the batch: (batch_size,timesteps,data_dim)
    #I NEED TO GET ATTENTION TO MULTIPLES, IF I USER MANY LSTM LAYERS --> https://stackoverflow.com/questions/47187149/keras-lstm-batch-input-shape
    #LINK WITH STATEFUL APPROACH --> https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html

    #RESHAPE THE DATA TO USE ON MODEL, FORMAT: (NumberOfExamples, TimeSteps, FeaturesPerStep)
    #EXPLANATION WHEN I NEED TO USE A TIME_STEMP DIFFERENT OF 1 --> https://github.com/keras-team/keras/issues/8568
    examplesWithoutTimeStempsXTrain = int((len(x_train)/time_stemps))
    examplesWithoutTimeStempsXTest = int((len(x_test) / time_stemps))
    x_train = x_train.reshape(examplesWithoutTimeStempsXTrain ,time_stemps, features)
    x_test =  x_test.reshape(examplesWithoutTimeStempsXTest, time_stemps, features)

    #RESHAPE TARGETS --> FORMAT: (NumberOfExamples, TimeSteps) --> https://stackoverflow.com/questions/46165464/reshape-keras-input-for-lstm
    y_train = y_train.reshape(int((len(y_train)/(time_stemps))), time_stemps) #3 POSSIBLE RESULTS PER TIME STEMPS
    y_test = y_test.reshape((int(len(y_test)/(time_stemps))), time_stemps)

    #FINNALY I NEED TO CONVERT THE CLASSES (TARGETS) TO BINARY
    y_train = keras.utils.to_categorical(y_train) #ALREADY MAKE RESHAPE BEFORE, AND NOW I DIDN'T NEED TO REAJUST ARRAY, ONLY NEED TO CONVERT TO BINARY
    y_test = keras.utils.to_categorical(y_test)

    #!!!!!!!!!!!!!!!!!!VERY IMPORTANT!!!!!!!!!!!!!!!!!!!!!
    #WHEN RETURN SEQUENCES = TRUE --> OUTPUT IS (#Samples, #Time steps, #NEURONS) AND IS FALSE : (#Samples, #LSTM units)
    #https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
    #NUMA ARQUITETURA STATEFUL, O Nº DAS AMOSTRAS TEM DE SER DIVISIVEL PELO BATCH SIZE

    #I NEED TO MAKE A SEPARATION BETWEEN THE PARTICLES (PSO) WEIGHTS--> IN TO MATRICES, KERNEL INPUT MATRIX AND RECURRENT KERNEL MATRIX
    inputParticleWeights, recurrentParticleWeigths = separationWeights(particleWeights, neurons, features)

    #DEFINITION OF MY CUSTOM CLASS INITIALIZER OF KERNEL AND RECURRENT WEIGHTS
    initializer = WeightsInitializer.WeightsInitializer(inputParticleWeights, recurrentParticleWeigths)

    model = Sequential()
    #I NEED TO USE THIS WHEN TIME STEPS IF GREATER THAN 1 --> RETURN SEQUENCES = TRUE
    #model.add(LSTM(neurons, batch_input_shape=(batch_size, time_stemps, features), return_sequences=True, stateful=True,
    #               kernel_initializer=initializer.initInputMat, recurrent_initializer= initializer.initRecMat))
    model.add(LSTM(neurons, batch_input_shape=(batch_size, time_stemps, features), #--> WHEN TIME STEPS IS 1 I DON'T USE RETURN SEQUENCES
                    kernel_initializer=initializer.initInputMat, recurrent_initializer= initializer.initRecMat))
    model.add(Dropout(0.5))
    model.add(Dense(3)) #3 OUTPUTS
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # INITIALIZATION OF CALLBACK, AND DEFINE THEM IN MODEL FITNESS
    weightsCallback = WeightsUpgradeOnTraining.WeightsUpgradeOnTraining(particleWeights=100, numberOfNeurons=neurons) #PARTICLES WEIGHTS IS TEMPORARY

    #FITTING MODEL
    model.fit(x_train, y_train, epochs=200, batch_size=batch_size, shuffle=False, callbacks=[weightsCallback])#BATCH_SIZE AND SHUFFLE BECAUSE TIME_STEPS DIFFERENT FROM 1

    predictions = model.predict(x_test, batch_size=batch_size)  # RETURNS A NUMPY ARRAY WITH PREDICTIONS

    # WELL, I NEED TO COMPARE THE PREDICTIONS WITH REAL VALUES
    hitRate = 0
    if time_stemps == 1:
        hitRate = accuracyTimeStempsEqual1(y_test, predictions)
    else:
        hitRate = accuracyTimeStempsGreaterThan1(y_test, predictions)

    #LOSS FUNCTION --> THE OBJECTIVE IS TO MINIMIZE THE LOSS, AND BEST ACCURACY'S MINIMIZE'S LOSS --> EXAMPLE: LOSS = (1- 0,8) < LOSS = (1-0,2)
    loss = (1- hitRate)

    print(loss)

    return loss

def applyLSTMUsingPSO(particles, x_train, x_test, y_train, y_test, neurons, batch_size, time_stemps, features):

    '''

    :param x_train: training samples dataset
    :param x_test: testing samples dataset
    :param y_train: training targets
    :param y_test: testing targets
    :param neurons: number of neurons using in model
    :param batch_size: batch size using on LSTM --> in concordance with training and test dataset's
    :param time_stemps:
    :param features: number features of problem
    :return: numpy array with all particles loss
    '''

    nParticles = particles.shape[0]
    loss, his = [objectiveFunctionLSTM(x_train, x_test, y_train, y_test, neurons, batch_size, time_stemps, features, particles[i]) for i in range(nParticles)] #FALTA AINDA PASSAR OS DADOS DE UMA PARTICULA, MAS POR AGORA NAO INTERESSA --> 1º NECESSÁRIO COLOCAR O MODELO FUNCIONAL
    return loss

def objectiveFunctionAlexNet(particles, x_train, x_test, y_train, y_test):

    try:

        numberParticles = particles.shape[0]
        allLosses = [AlexNet.alexNet(particleDimensions=particles[i], x_train=x_train, x_test=x_test,
                                     y_train=y_train, y_test=y_test) for i in range(numberParticles)]

        return allLosses

    except:
        raise

def main():

    '''
        GET ALL PARTIES NEEDED FROM DATASET
    '''

    # X, Y, x_train, x_test, y_train, y_test = getDataset(25) #TEST PERCENTAGE IS 25%
    #
    # '''
    #     PSO FORMULATION FOR CNN IMPLEMENTATION
    # '''
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # dimensions = 2 # IN FIRST DIMENSION I HAVE REPRESENTED NUMBER OF NODES ON A CNN LAYER, AND IN SECOND DIMENSION KERNEL USED ON CNN LAYER (MATRIX)
    # minBound = numpy.ones(2)#MIN VALUE BOUND --> I CAN ONLY OPTIMIZE A SINGLE LIMIT FOR ALL DIMENSIONS
    # maxBound = 64 * numpy.ones(2) #MAX VALUE BOUND --> I CAN ONLY OPTIMIZE A SINGLE LIMIT FOR ALL DIMENSIONS
    # maxBound[1] = 4 #IN THIS DIMENSION THE MAX VALUE IS 4
    # bounds = (minBound, maxBound) #MAX DIMENSIONS LIMITS RESPECTIVELY FOR NUMBER OF NODES OF A CNN LAYER AND KERNEL DIMENSION
    #
    # #optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options, bounds=bounds)
    #
    # #cost, pos = optimizer.optimize(objectiveFunctionPSO, X_train=x_train, X_test= x_test, Y_train= y_train, Y_test= y_test ,iters=2)
    #
    # '''
    #     PSO FORMULATION FOR LSTM IMPLEMENTATION
    # '''
    #
    # #NEED TO DEFINE INITIAL VALUES OF LSTM (BATCH_SIZE, TIME_STEMP, ...), IN ORDER TO DEFINE THE DIMENSIONS OF PSO --> I CAN CREATE AN PSO OPTIMIZER BEFORE, TO CHECK THIS VALUES, OR DEFINE NEW DIMENSIONS SPECIFIC TO THIS VALUES
    # #THE BOUNDS FOR NOW ARE THE DEFAULT VALUES --> BETWEEN 0 AND 1
    #
    # neurons = 150
    # #BATCH_SIZE NEEDS TO BE A NUMBER MINOR THAN NUMBER OF SAMPLES FOR TRAINING AND TEST, AND NEED TO BE DIVISIVEL BY THEM
    # batch_size = 5 #I HAVE 150 SAMPLES, AND TO REDUCE THE COMPUTACIONAL REQUIREMENTS, I DEFINE 3 TIMES TO LEARN (50*3) = 150
    # time_stemps = 1 #EVERY VALUES ON EVERY ATTRIBUTES HAVE THE SAME FORMAT AND LENGHT --> FLOAT VALUES LIKE: 1.2, LSTM NEEDS TO LOOK AT THIS 3 PIECES
    # data_dimension = 4 #NUMBER OF FEATURES
    #
    # #DEFINITION OF THE DIMENSIONS OF THE PROBLEM --> REPRESENTS THE WEIGHTS OF LSTM LAYER (KERNEL AND RECURRENT MATRIXES) --> I DIDN'T CONSIDER BIAS HERE
    # kernelMatrix_Input = (data_dimension * neurons) * 4 #(data_dimension * neurons) REPRESENTS W_I OR W_F OR W_C OR W_O AND THEN I NEED TO MULTIPLY BY THE 4 HYPHOTESIS (W_I, W_F, W_C, W_O)
    # recurrentKernel = (neurons * neurons) * 4 #(neurons * neurons) REPRESENTS THE NUMBER OF POSSIBLE NEURONS ON A STATE, AND THEN I NEED TO MULTIPLY BY 4 (ALL STATES U_I, U_F, U_C, U_O)
    # dimensions = kernelMatrix_Input + recurrentKernel
    #
    # #I CANT USE THE DATASET DEFINE BEFORE, BECAUSE WITH A 25 PERCENTAGE I CANT GET A POSSIBLE BATCH_SIZE TO DIVIDE BY THIS TWO DATASET'S
    # #LINK WITH THIS EXPLANATION --> https://medium.com/@ellery.leung/rnn-lstm-example-with-keras-about-input-shape-94120b0050e
    # X, Y, x_train, x_test, y_train, y_test = getDataset(20)  # I NEED TO RESTORE THE DATASET PERCENTAGE, IN ORDER TO FIND A VALUE DIVISIVEL BY TRAIN AND TEST DATASET: 150 SAMPLES --> 120 FOR TRAIN AND 30 FOR TEST, AND WITH A BATCH_SIZE= 30 I CAN DIVIDE FOR THIS TWO DATASET'S
    #
    # optimizer = ps.single.GlobalBestPSO(n_particles=1, dimensions=dimensions, options=options) #DEFAULT BOUNDS
    #
    # cost, pos = optimizer.optimize(applyLSTMUsingPSO, x_train=x_train, x_test= x_test, y_train= y_train, y_test= y_test, neurons=neurons, batch_size=batch_size, time_stemps=time_stemps, features=data_dimension ,iters=1) #the cost function has yet to be created
    #
    # '''
    #
    #     MLP WITHOUT PSO
    #
    # '''
    # #DEFINITION OF VARIABLES TO PASS TO mlp function
    # neurons = 100
    # batch_size = 30
    # features = X.shape[1]
    # classes = 3
    # epochs = 30
    #
    # #GET SPLIT OF DATASET --> 70% TRAIN AND 30% PER TEST
    # X, Y, x_train, x_test, y_train, y_test = getDataset(30)
    #
    # scores = MLP.mlp(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size, neurons=neurons, numberFeatures=features, numberClasses=classes, epochs=epochs)
    #
    # print('Loss: ', scores[0])
    # print('\nAccuracy', scores[1])
    #
    # '''
    #     CNN WITHOUT PSO
    # '''
    #
    # #DEFINITION OF VALUES OF PARAMETERS
    # nFilters = 12
    # batch_size = 5
    # epochs = 15 #n value = 6 --> (epochs/batch_size) = 30/5 = 6
    # kernel_size = (4,)#TUPLE OF ONE INTEGER, COULD BE ALSO A SINGLE INTEGER
    # #STRIDE IF I WANT I CAN OVERRIDE THIS VALUE BY DEFAULT IS 1 ON PARAMETER OF cnn function
    #
    # #GET SPLIT OF DATASET --> 70% TRAIN AND 30% PER TEST
    # X, Y, x_train, x_test, y_train, y_test = getDataset(30)
    #
    # score = CNN.cnn(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, filters=nFilters, batch_size=batch_size, epochs=epochs, kernel_size=kernel_size)
    #
    # print('\nAccuracy: ', score)
    #
    # '''
    #     LSTM WITHOUT PSO
    # '''
    #
    # neurons = 50
    # batch_size = 5
    # epochs = 30
    #
    # #DEFINITION OF TRAINING AND TEST DATASET
    # X, Y, x_train, x_test, y_train, y_test = getDataset(30)
    #
    # score = LSTM_Model.lstm(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, neurons=neurons, batch_size=batch_size, epochs=epochs)
    #
    # print('\nAccuracy: ', score)
    #
    # '''
    #     CNN WITH PSO
    # '''
    #
    # #DEFINITION OF CNN PARAMETERS
    # batch_size = 5
    # kernel_size = (4,)
    # stride = 1
    # #EPOCHS AND FILTERS ARE DEFINED BY PARTICLES
    #
    # #DEFINITION OF PSO PARAMETERS
    # numberParticles = 8
    # iterations = 2
    #
    # minBound = numpy.ones(2)  # MIN BOUND FOR TWO DIMENSIONS IS 1
    # maxBound = numpy.ones(2)  # ONLY INITIALIZATION
    # maxBound[0] = 601  # MAX NUMBER OF FILTERS
    # maxBound[1] = 401  # MAX NUMBER OF EPOCHS
    # bounds = (minBound, maxBound)
    #
    # options = {config.C1 : 0.3, config.C2 : 0.2, config.INERTIA : 0.9, config.NUMBER_NEIGHBORS : 4, config.MINKOWSKI_RULE : 2 }
    # #options = {config.C1: 0.3, config.C2: 0.2, config.INERTIA: 0.9}
    # kwargs = {config.TYPE : config.LOCAL_BEST, config.OPTIONS : options}
    # #kwargs = {config.TYPE: config.GLOBAL_BEST, config.OPTIONS: options}
    #
    # cost, pos, optimizer = CNN_WithOptimization.callCNNOptimization(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
    #                                                      kernel_size=kernel_size, numberParticles=numberParticles, iterations=iterations,
    #                                                      bounds=bounds, stride=stride, **kwargs)
    #
    # print(cost)
    # print(pos)
    #
    # #PLOT'S
    # plots.plotCostHistory(optimizer=optimizer)
    #
    # xPlotLimits = numpy.ones(2)
    # xPlotLimits[1] = maxBound[0] #MAX VALUE OF FILTER AXIS IS 601 (X AXIS)
    # yPlotLimits = numpy.ones(2)
    # yPlotLimits[1] = maxBound[1] #MAX VALUE OF EPOCHS AXIS IS 401 (Y AXIS)
    # filename = 'particlesHistoryPlot.html'
    # plots.plotPositionHistory(optimizer=optimizer, xLimits=xPlotLimits, yLimits=yPlotLimits,
    #                           xLabel=config.X_LABEL_FILTERS, yLabel=config.Y_LABEL_EPOCHS ,filename=filename)
    #
    # '''
    #     LSTM WITH PSO
    # '''
    #
    # #DEFINITION OF LSTM PARAMETERS, EPOCHS AND NEURONS ARE DEFINED BY PSO
    # batch_size = 5
    #
    # #DEFINITION OF PSO PARAMETERS
    # numberParticles = 20
    # iterations = 10
    # dimensions = 2 # [0] --> NEURONS , [1] --> EPOCHS
    #
    # #DEFINITION OF DIMENSIONS BOUNDS, X AXIS --> NEURONS and Y AXIS --> EPOCHS
    # minBounds = numpy.ones(2)
    # maxBounds = numpy.ones(2)
    # maxBounds[0] = 251 #I REDUCE THIS DIMENSIONS, IN ORDER TO MAKE OPTIMIZATION MORE QUICKLY
    # maxBounds[1] = 201
    # bounds = (minBounds, maxBounds)
    #
    # #DEFINITION OF DIFFERENT TOPOLOGIES OPTIONS
    # lbest_options = {config.C1 : 0.3, config.C2 : 0.2, config.INERTIA : 0.9, config.NUMBER_NEIGHBORS : 4, config.MINKOWSKI_RULE : 2}
    # lbest_kwargs = {config.TYPE : config.LOCAL_BEST, config.OPTIONS : lbest_options}
    # gbest_options = {config.C1 : 0.3, config.C2 : 0.2, config.INERTIA : 0.9}
    # gbest_kwargs = {config.TYPE : config.GLOBAL_BEST, config.OPTIONS : gbest_options}
    #
    # #PASSING ALL THIS OPTIONS TO LSTM_PSO applyLSTM_PSO FUNCTION
    # cost, pos, optimizer = LSTM_PSO.applyLSTM_PSO(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
    #                                               numberParticles=numberParticles, iterations=iterations, dimensions=dimensions,
    #                                               bounds=bounds, **lbest_kwargs)
    # print(cost)
    # print(pos)
    #
    # #PLOT GRAPHICS ILLUSTRATING THE COST VARIATION AND PARTICLES MOVEMENT AND CONVERGENCE
    # plots.plotCostHistory(optimizer=optimizer)
    # plots.plotPositionHistory(optimizer=optimizer, xLimits=(minBounds[0], maxBounds[0]),
    #                           yLimits=(minBounds[1], maxBounds[1]), filename='lstmParticlesPosConvergence.html',
    #                           xLabel=config.X_LABEL_NEURONS, yLabel=config.Y_LABEL_EPOCHS)

    '''###############################################################################
    #                                                                                #   
    #                          CIFAR - 10 OPTIMIZATION                               #
    #                          CONV_2D - PSO OPTIMIZATION                            #
    #      FOR THIS PROBLEM I ONLY CONSIDER 4 CLASSES (CATS, DOGS, FROGS, HORSES)    #  
    #   OFFICIAL EXPLANATION OF CIFAR-10: http://www.cs.toronto.edu/~kriz/cifar.html #
    #                                                                                #
    '''###############################################################################
    #https://www.researchgate.net/post/Why_cant_I_get_accuracy_very_badly_predicted_on_training_data_with_dataset_of_having_validation_accuracy_of_97_using_resnet50
    #GET DATA
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    '''
        TRAINING DATA HAS 5 BATCHES --> FOR EVERY BATCH I RETRIEVE 20 IMAGES OF EACH CLASS--> TOTAL OF 100 PER CLASS
        AS IN TEST DATASET, there is only one batch, so just get 25 samples per class (100 AT ALL)
        FROG CLASS - 6
        DOG CLASS - 5
        CAT CLASS - 3
        HORSE CLASS - 7
    '''

    #GET ALL INDICES OF CLASSES THAT I DON'T NEED--> FIRST GET POSITIONS TO DELETE AND THEM DELETE THIS OCORRENCES
    values = [0, 1, 2, 4, 8, 9] #INDICES CLASSES TO REMOVE
    deleted_Train_Positions = [i for i in range(y_train.shape[0]) if y_train[i] in values]
    deleted_Test_Positions = [i for i in range(y_test.shape[0]) if y_test[i] in values]

    #GEL ONLY CLASSES THAT I NEED, AND AFTER THAT I NEED TO CONVERT AGAIN TO NUMPY ARRAY
    x_train = [x_train[i] for i in range(x_train.shape[0]) if i not in deleted_Train_Positions]
    x_train = numpy.array(x_train)
    y_train = [y_train[i] for i in range(y_train.shape[0]) if i not in deleted_Train_Positions]
    y_train = numpy.array(y_train)
    x_test = [x_test[i] for i in range(x_test.shape[0]) if i not in deleted_Test_Positions]
    x_test = numpy.array(x_test)
    y_test = [y_test[i] for i in range(y_test.shape[0]) if i not in deleted_Test_Positions]
    y_test = numpy.array(y_test)

    #NOW I NEED TO GET ONLY FEW CLASS PER CLASS (100 samples per class for train) and (20 samples per class for test)
    howManyValuesPerClass_Train = [2000, 2000, 2000, 0, 2000, 0, 0, 0, 2000, 2000] #INDEXES WITH 0 VALUE REPRESENT THE CLASS ALLOWED
    howManyValuesPerClass_Test = [800, 800, 800, 0, 800, 0, 0, 0, 800, 800]  # INDEXES WITH 0 VALUE REPRESENT THE CLASS ALLOWED
    selectedIndexes_Train = numpy.array([])
    selectedIndexes_Test = numpy.array([])

    for i, j in zip_longest(range(y_train.shape[0]), range(y_test.shape[0])):
        if y_train[i][0] not in values: #IF IT'S A ALLOWED CLASS
            if howManyValuesPerClass_Train[y_train[i][0]] < 4000: #ONLY ACCEPTS UNTIL 100
                selectedIndexes_Train = numpy.append(selectedIndexes_Train, i) #ADD POSITION TO SELECTED_INDEXES
                howManyValuesPerClass_Train[y_train[i][0]] = howManyValuesPerClass_Train[y_train[i][0]] + 1
        if j != None:
            if y_test[j][0] not in values:
                if howManyValuesPerClass_Test[y_test[j][0]] < 1000:
                    selectedIndexes_Test = numpy.append(selectedIndexes_Test, j)
                    howManyValuesPerClass_Test[y_test[j][0]] = howManyValuesPerClass_Test[y_test[j][0]] + 1

    #SELECT ONLY INDEXES IDENTIFIED BEFORE, FOR TRAIN AND TEST
    x_train = [x_train[i] for i in range(selectedIndexes_Train.shape[0])]
    x_train = numpy.array(x_train)
    y_train = [y_train[i] for i in range(selectedIndexes_Train.shape[0])]
    y_train = numpy.array(y_train)
    x_test = [x_test[i] for i in range(selectedIndexes_Test.shape[0])]
    x_test = numpy.array(x_test)
    y_test = [y_test[i] for i in range(selectedIndexes_Test.shape[0])]
    y_test = numpy.array(y_test)

    print("Y_TRAIN:\n", y_train[:])
    print("x_TRAIN:\n", x_train[:])
    print("Y_Test:\n", y_test[:])
    print("x_Teste:\n", x_test[:])

    #REFORMULE TARGETS INDEXES 3 --> 0 , 5 --> 1, 6 --> 2, 7 --> 3
    #IN ORDER THO HAVE ONLY THIS CLASSES IN FINAL OUTPUT
    oldPos = [3, 5, 6, 7]
    for i, j in zip_longest(range(y_train.shape[0]), range(y_test.shape[0])):
        getIndex_Train = oldPos.index(y_train[i])
        y_train[i] = getIndex_Train
        if j != None:
            getIndex_Test = oldPos.index(y_test[j])
            y_test[j] = getIndex_Test

    print("Y_TRAIN:\n", y_train[:])
    print("x_TRAIN:\n", x_train[:])
    print("Y_Test:\n", y_test[:])
    print("x_Teste:\n", x_test[:])

    #CHECK SHAPE OF DATA
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    #CHANGE X_TRAIN AND X_TEST ASTYPE --> INT TO FLOAT, IN ORDER TO APPLY NORMALIZATION
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)

    #APPLY NORMALIZATION TO DATA --> RANGE [0-1], PUT SAME IMPORTANCE TO ALL PIXELS, AND TO AVOID MORE HYPERPARAMETERS IN WEIGHTS LEARNING
    for i, j in zip_longest(range(x_train.shape[0]), range(x_test.shape[0])): #DIFFERENT LENGTHS
        for k, l in zip(range(x_train.shape[1]), range(x_test.shape[1])): #SAME LENGTH, DOESN'T MATTER
            for m, n in zip(range(x_train.shape[2]), range(x_test.shape[2])): #SAME LENGTH, DOESN'T MATTER
                for w in range(3):
                    x_train[i][k][m][w] = x_train[i][k][m][w] / 255
                    if j != None:
                        x_test[j][l][n][w] = x_test[j][l][n][w] / 255

    #x_train = (x_train - x_train.mean(axis=(0, 1, 2), keepdims=True)) / x_train.std(axis=(0, 1, 2), keepdims=True)
    #x_test = (x_test - x_test.mean(axis=(0, 1, 2), keepdims=True)) / x_test.std(axis=(0, 1, 2), keepdims=True)
    print("Y_TRAIN:\n", y_train[:])
    print("x_TRAIN:\n", x_train[:])
    print("Y_Test:\n", y_test[:])
    print("x_Teste:\n", x_test[:])

    #COVERSION TARGETS TO CATEGORICAL
    y_train = keras.utils.to_categorical(y_train, 4) #4 NUMBER OF CLASSES (DOGS, CATS, FROGS AND HORSES)
    y_test = keras.utils.to_categorical(y_test, 4)

    #NOW PRE-PROCESSING IS FINISHED, AFTER THAT I CAN MAKE CNN MODEL
    # model = Sequential()
    # model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(filters=16, kernel_size=(3,3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=2))
    # model.add(Dropout(0.3))
    #
    # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(filters=32, kernel_size=(2,2)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=2))
    # model.add(Dropout(0.25))
    # # model.add(Conv2D(filters=64, kernel_size=(2,2)))
    # # model.add(Activation('relu'))
    # # model.add(BatchNormalization())
    # # model.add(MaxPooling2D(pool_size=2, strides=2))
    # model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(units=4))
    # model.add(Activation('softmax'))
    # model.summary()
    #
    # #OPTIMIZER
    # opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    #
    # # COMPILE MODEL
    # model.compile(optimizer=opt, loss='categorical_crossentropy',
    #               metrics=['accuracy'])  # CROSSENTROPY BECAUSE IT'S MORE ADEQUATED TO MULTI-CLASS PROBLEMS
    #
    # data_augmentation = False
    #
    # if data_augmentation == False:
    #     # FIT MODEL
    #     historyOfTraining = model.fit(
    #         x=x_train,
    #         y=y_train,
    #         batch_size=32,
    #         epochs=100,
    #         validation_split=0.2,
    #         shuffle=True
    #     )
    #
    #     predict = model.predict(x=x_test, batch_size=32)
    #     print(predict)
    #     print(y_test)
    #
    #     predict = (predict == predict.max(axis=1)[:, None]).astype(int)
    #     print(predict)
    #
    #     numberRights = 0
    #     for i in range(len(y_test)):
    #         indexMaxValue = numpy.argmax(predict[i], axis=0)
    #         if indexMaxValue == numpy.argmax(y_test[i],
    #                                          axis=0):  # COMPARE INDEX OF MAJOR CLASS PREDICTED AND REAL CLASS
    #             numberRights = numberRights + 1
    #
    #     hitRate = numberRights / len(y_test)  # HIT PERCENTAGE OF CORRECT PREVISIONS
    #
    #     print(hitRate)
    #
    # else: #USE OF IMAGE DATA GENERATOR
    #     image_gen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     zca_epsilon=1e-06,  # epsilon for ZCA whitening
    #     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #     # randomly shift images horizontally (fraction of total width)
    #     width_shift_range=0.1,
    #     # randomly shift images vertically (fraction of total height)
    #     height_shift_range=0.1,
    #     shear_range=0.,  # set range for random shear
    #     zoom_range=0.,  # set range for random zoom
    #     channel_shift_range=0.,  # set range for random channel shifts
    #     # set mode for filling points outside the input boundaries
    #     fill_mode='nearest',
    #     cval=0.,  # value used for fill_mode = "constant"
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False,  # randomly flip images
    #     # set rescaling factor (applied before any other transformation)
    #     rescale=None,
    #     # set function that will be applied on each input
    #     preprocessing_function=None,
    #     # image data format, either "channels_first" or "channels_last"
    #     data_format=None,
    #     # fraction of images reserved for validation (strictly between 0 and 1)
    #     validation_split=0.0)
    #
    #     image_gen.fit(x_train)
    #
    #     model.fit_generator(image_gen.flow(
    #         x=x_train,
    #         y=y_train,
    #         batch_size=32),
    #     epochs=200,
    #     validation_data=(x_test, y_test),
    #     workers= 4)
    #
    #     scores = model.evaluate(x_test, y_test, verbose=1)
    #     print('Test loss:', scores[0])
    #     print('Test accuracy:', scores[1])
    #
    #     '''
    #         CNN CIFAR-10 PSO OPTIMIZATION
    #     '''

    '''
        TEST ALEX_NET
    '''

    # alexNetValues = [64, 128, 256, 256, 32, 512, 0.4, 0.0001]
    #alexNetValues = [32, 64, 64, 32, 16, 64, 0.4, 0.0001]

    #alexNetValuesAugmented = [128, 256, 384, 512, 32, 64, 0.4, 0.0001]
    #AlexNet.alexNet(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, particleDimensions=alexNetValuesAugmented)

    #loaded_model = load_model(config.SAVED_MODEL_FILE2)
    #AlexNet.alexNetAugmentation(loadModel=loaded_model, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, particleDimensions=alexNetValuesAugmented)
    #scores = loaded_model.evaluate(x_test, y_test, verbose=1)
    #print('Test loss:', scores[0])
    #print('Test accuracy:', scores[1])

    '''
        TEST VGG NET
    '''

    #finalScore = VGGNet.vggNet(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    #print(finalScore)

    '''
        ALEX_NET WITH PSO OPTIMIZATION
    '''

    #DEFINITION OF PSO PARAMETERS
    numberParticles = 30
    iterations = 20
    dimensions = 5 # [0-3] --> NUMBER FILTERS, [3-5] --> NUMBER NEURONS

    #DEFINITION OF DIMENSIONS BOUNDS, X AXIS --> NEURONS and Y AXIS --> EPOCHS
    minBounds = numpy.ones(5)
    maxBounds = numpy.ones(5)
    maxBounds[0] = 64
    maxBounds[1] = 128
    maxBounds[2] = 160 #I REDUCE THIS DIMENSIONS, IN ORDER TO MAKE OPTIMIZATION MORE QUICKLY
    maxBounds[3] = 128
    maxBounds[4] = 32
    bounds = (minBounds, maxBounds)

    #DEFINITION OF DIFFERENT TOPOLOGIES OPTIONS
    lbest_options = {config.C1 : 0.3, config.C2 : 0.2, config.INERTIA : 0.9, config.NUMBER_NEIGHBORS : 4, config.MINKOWSKI_RULE : 2}
    lbest_kwargs = {config.TYPE : config.LOCAL_BEST, config.OPTIONS : lbest_options}
    gbest_options = {config.C1 : 0.4, config.C2 : 0.4, config.INERTIA : 0.9}
    gbest_kwargs = {config.TYPE : config.GLOBAL_BEST, config.OPTIONS : gbest_options}

    optimizer = ps.single.GlobalBestPSO(n_particles=numberParticles, dimensions=dimensions,
                                              options=gbest_options, bounds=bounds)

    cost, pos = optimizer.optimize(objective_func=objectiveFunctionAlexNet,x_train=x_train, x_test=x_test , y_train=y_train,
                                   y_test=y_test, iters=iterations)

    print(cost)
    print(pos)

    plots.plotCostHistory(optimizer=optimizer)

    '''
        ALEX NET WITH GENETIC ALGORITHM OPTIMIZATION
    '''

    # population_size = 10
    # num_generations = 5
    # gene_length = 35 # BIT LENGTH --> [64, 128, 256, 256, 64] --> [6, 7, 8, 8, 6] = 35
    #
    # creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
    # creator.create('Individual', list, fitness=creator.FitnessMax)
    #
    # toolbox = base.Toolbox()
    # toolbox.register('binary', bernoulli.rvs, 0.5)
    # toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=gene_length) #REGISTER INDIVIDUAL
    # toolbox.register('population', tools.initRepeat, list, toolbox.individual) #REGISTER POPULATION
    #
    # toolbox.register('mate', tools.cxOrdered) #CROSSOVER TECHNIQUE --> https://www.researchgate.net/figure/The-order-based-crossover-OX-a-and-the-insertion-mutation-b-operators_fig2_224330103
    # toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.6) #MUTATION TECHNIQUE --> https://www.mdpi.com/1999-4893/12/10/201/htm
    # toolbox.register('select', tools.selTournament, tournsize=100) #IN MINIMIZATION PROBLEMS I CAN'T USE ROULETTE
    # toolbox.register('evaluate', AlexNet.alexNetForGA, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test) #EVALUATION FUNCTION
    #
    # population = toolbox.population(n=population_size)
    # r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.2, ngen=num_generations, verbose=True)
    #
    # bestValue = tools.selBest(population, k=1) #I ONLY NEED BEST INDIVIDUAL --> ARRAY BIDIMENSIONAL (K=1, GENE_LENGTH)
    #
    # conv1 = BitArray(bestValue[0][0:6]).uint
    # print(conv1)
    # conv2 = BitArray(bestValue[0][6:13]).uint
    # print(conv2)
    # conv3 = BitArray(bestValue[0][13:21]).uint
    # print(conv3)
    # dense1 = BitArray(bestValue[0][21:29]).uint
    # print(dense1)
    # dense2 = BitArray(bestValue[0][29:35]).uint
    # print(dense2)

if __name__ == "__main__":
    main()