import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import numpy
from sklearn.preprocessing import MinMaxScaler
import pyswarms as ps
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Dropout
import keras
import WeightsUpgradeOnTraining, WeightsInitializer
import MLP
import CNN
import LSTM_Model
import CNN_WithOptimization
import plots
import config
import LSTM_PSO
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #MAKES MORE FASTER THE INITIAL SETUP OF GPU --> WARNINGS INITIAL STEPS IS MORE QUICKLY
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #THIS LINE DISABLES GPU OPTIMIZATION

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
    loss = [objectiveFunctionLSTM(x_train, x_test, y_train, y_test, neurons, batch_size, time_stemps, features, particles[i]) for i in range(nParticles)] #FALTA AINDA PASSAR OS DADOS DE UMA PARTICULA, MAS POR AGORA NAO INTERESSA --> 1º NECESSÁRIO COLOCAR O MODELO FUNCIONAL
    return loss

def main():

    '''
        GET ALL PARTIES NEEDED FROM DATASET
    '''

    X, Y, x_train, x_test, y_train, y_test = getDataset(25) #TEST PERCENTAGE IS 25%

    '''
        PSO FORMULATION FOR CNN IMPLEMENTATION
    '''
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    dimensions = 2 # IN FIRST DIMENSION I HAVE REPRESENTED NUMBER OF NODES ON A CNN LAYER, AND IN SECOND DIMENSION KERNEL USED ON CNN LAYER (MATRIX)
    minBound = numpy.ones(2)#MIN VALUE BOUND --> I CAN ONLY OPTIMIZE A SINGLE LIMIT FOR ALL DIMENSIONS
    maxBound = 64 * numpy.ones(2) #MAX VALUE BOUND --> I CAN ONLY OPTIMIZE A SINGLE LIMIT FOR ALL DIMENSIONS
    maxBound[1] = 4 #IN THIS DIMENSION THE MAX VALUE IS 4
    bounds = (minBound, maxBound) #MAX DIMENSIONS LIMITS RESPECTIVELY FOR NUMBER OF NODES OF A CNN LAYER AND KERNEL DIMENSION

    #optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options, bounds=bounds)

    #cost, pos = optimizer.optimize(objectiveFunctionPSO, X_train=x_train, X_test= x_test, Y_train= y_train, Y_test= y_test ,iters=2)

    '''
        PSO FORMULATION FOR LSTM IMPLEMENTATION
    '''

    #NEED TO DEFINE INITIAL VALUES OF LSTM (BATCH_SIZE, TIME_STEMP, ...), IN ORDER TO DEFINE THE DIMENSIONS OF PSO --> I CAN CREATE AN PSO OPTIMIZER BEFORE, TO CHECK THIS VALUES, OR DEFINE NEW DIMENSIONS SPECIFIC TO THIS VALUES
    #THE BOUNDS FOR NOW ARE THE DEFAULT VALUES --> BETWEEN 0 AND 1

    neurons = 150
    #BATCH_SIZE NEEDS TO BE A NUMBER MINOR THAN NUMBER OF SAMPLES FOR TRAINING AND TEST, AND NEED TO BE DIVISIVEL BY THEM
    batch_size = 5 #I HAVE 150 SAMPLES, AND TO REDUCE THE COMPUTACIONAL REQUIREMENTS, I DEFINE 3 TIMES TO LEARN (50*3) = 150
    time_stemps = 1 #EVERY VALUES ON EVERY ATTRIBUTES HAVE THE SAME FORMAT AND LENGHT --> FLOAT VALUES LIKE: 1.2, LSTM NEEDS TO LOOK AT THIS 3 PIECES
    data_dimension = 4 #NUMBER OF FEATURES

    #DEFINITION OF THE DIMENSIONS OF THE PROBLEM --> REPRESENTS THE WEIGHTS OF LSTM LAYER (KERNEL AND RECURRENT MATRIXES) --> I DIDN'T CONSIDER BIAS HERE
    kernelMatrix_Input = (data_dimension * neurons) * 4 #(data_dimension * neurons) REPRESENTS W_I OR W_F OR W_C OR W_O AND THEN I NEED TO MULTIPLY BY THE 4 HYPHOTESIS (W_I, W_F, W_C, W_O)
    recurrentKernel = (neurons * neurons) * 4 #(neurons * neurons) REPRESENTS THE NUMBER OF POSSIBLE NEURONS ON A STATE, AND THEN I NEED TO MULTIPLY BY 4 (ALL STATES U_I, U_F, U_C, U_O)
    dimensions = kernelMatrix_Input + recurrentKernel

    #I CANT USE THE DATASET DEFINE BEFORE, BECAUSE WITH A 25 PERCENTAGE I CANT GET A POSSIBLE BATCH_SIZE TO DIVIDE BY THIS TWO DATASET'S
    #LINK WITH THIS EXPLANATION --> https://medium.com/@ellery.leung/rnn-lstm-example-with-keras-about-input-shape-94120b0050e
    X, Y, x_train, x_test, y_train, y_test = getDataset(20)  # I NEED TO RESTORE THE DATASET PERCENTAGE, IN ORDER TO FIND A VALUE DIVISIVEL BY TRAIN AND TEST DATASET: 150 SAMPLES --> 120 FOR TRAIN AND 30 FOR TEST, AND WITH A BATCH_SIZE= 30 I CAN DIVIDE FOR THIS TWO DATASET'S

    optimizer = ps.single.GlobalBestPSO(n_particles=1, dimensions=dimensions, options=options) #DEFAULT BOUNDS

    cost, pos = optimizer.optimize(applyLSTMUsingPSO, x_train=x_train, x_test= x_test, y_train= y_train, y_test= y_test, neurons=neurons, batch_size=batch_size, time_stemps=time_stemps, features=data_dimension ,iters=1) #the cost function has yet to be created

    '''
        
        MLP WITHOUT PSO
        
    '''
    #DEFINITION OF VARIABLES TO PASS TO mlp function
    neurons = 100
    batch_size = 30
    features = X.shape[1]
    classes = 3
    epochs = 30

    #GET SPLIT OF DATASET --> 70% TRAIN AND 30% PER TEST
    X, Y, x_train, x_test, y_train, y_test = getDataset(30)

    scores = MLP.mlp(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size, neurons=neurons, numberFeatures=features, numberClasses=classes, epochs=epochs)

    print('Loss: ', scores[0])
    print('\nAccuracy', scores[1])

    '''
        CNN WITHOUT PSO
    '''

    #DEFINITION OF VALUES OF PARAMETERS
    nFilters = 12
    batch_size = 5
    epochs = 15 #n value = 6 --> (epochs/batch_size) = 30/5 = 6
    kernel_size = (4,)#TUPLE OF ONE INTEGER, COULD BE ALSO A SINGLE INTEGER
    #STRIDE IF I WANT I CAN OVERRIDE THIS VALUE BY DEFAULT IS 1 ON PARAMETER OF cnn function

    #GET SPLIT OF DATASET --> 70% TRAIN AND 30% PER TEST
    X, Y, x_train, x_test, y_train, y_test = getDataset(30)

    score = CNN.cnn(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, filters=nFilters, batch_size=batch_size, epochs=epochs, kernel_size=kernel_size)

    print('\nAccuracy: ', score)

    '''
        LSTM WITHOUT PSO
    '''

    neurons = 50
    batch_size = 5
    epochs = 30

    #DEFINITION OF TRAINING AND TEST DATASET
    X, Y, x_train, x_test, y_train, y_test = getDataset(30)

    score = LSTM_Model.lstm(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, neurons=neurons, batch_size=batch_size, epochs=epochs)

    print('\nAccuracy: ', score)

    '''
        CNN WITH PSO
    '''

    #DEFINITION OF CNN PARAMETERS
    batch_size = 5
    kernel_size = (4,)
    stride = 1
    #EPOCHS AND FILTERS ARE DEFINED BY PARTICLES

    #DEFINITION OF PSO PARAMETERS
    numberParticles = 8
    iterations = 2

    minBound = numpy.ones(2)  # MIN BOUND FOR TWO DIMENSIONS IS 1
    maxBound = numpy.ones(2)  # ONLY INITIALIZATION
    maxBound[0] = 601  # MAX NUMBER OF FILTERS
    maxBound[1] = 401  # MAX NUMBER OF EPOCHS
    bounds = (minBound, maxBound)

    options = {config.C1 : 0.3, config.C2 : 0.2, config.INERTIA : 0.9, config.NUMBER_NEIGHBORS : 4, config.MINKOWSKI_RULE : 2 }
    #options = {config.C1: 0.3, config.C2: 0.2, config.INERTIA: 0.9}
    kwargs = {config.TYPE : config.LOCAL_BEST, config.OPTIONS : options}
    #kwargs = {config.TYPE: config.GLOBAL_BEST, config.OPTIONS: options}

    cost, pos, optimizer = CNN_WithOptimization.callCNNOptimization(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
                                                         kernel_size=kernel_size, numberParticles=numberParticles, iterations=iterations,
                                                         bounds=bounds, stride=stride, **kwargs)

    print(cost)
    print(pos)

    #PLOT'S
    plots.plotCostHistory(optimizer=optimizer)

    xPlotLimits = numpy.ones(2)
    xPlotLimits[1] = maxBound[0] #MAX VALUE OF FILTER AXIS IS 601 (X AXIS)
    yPlotLimits = numpy.ones(2)
    yPlotLimits[1] = maxBound[1] #MAX VALUE OF EPOCHS AXIS IS 401 (Y AXIS)
    filename = 'particlesHistoryPlot.html'
    plots.plotPositionHistory(optimizer=optimizer, xLimits=xPlotLimits, yLimits=yPlotLimits,
                              xLabel=config.X_LABEL_FILTERS, yLabel=config.Y_LABEL_EPOCHS ,filename=filename)

    '''
        LSTM WITH PSO
    '''

    #DEFINITION OF LSTM PARAMETERS, EPOCHS AND NEURONS ARE DEFINED BY PSO
    batch_size = 5

    #DEFINITION OF PSO PARAMETERS
    numberParticles = 20
    iterations = 10
    dimensions = 2 # [0] --> NEURONS , [1] --> EPOCHS

    #DEFINITION OF DIMENSIONS BOUNDS, X AXIS --> NEURONS and Y AXIS --> EPOCHS
    minBounds = numpy.ones(2)
    maxBounds = numpy.ones(2)
    maxBounds[0] = 251 #I REDUCE THIS DIMENSIONS, IN ORDER TO MAKE OPTIMIZATION MORE QUICKLY
    maxBounds[1] = 201
    bounds = (minBounds, maxBounds)

    #DEFINITION OF DIFFERENT TOPOLOGIES OPTIONS
    lbest_options = {config.C1 : 0.3, config.C2 : 0.2, config.INERTIA : 0.9, config.NUMBER_NEIGHBORS : 4, config.MINKOWSKI_RULE : 2}
    lbest_kwargs = {config.TYPE : config.LOCAL_BEST, config.OPTIONS : lbest_options}
    gbest_options = {config.C1 : 0.3, config.C2 : 0.2, config.INERTIA : 0.9}
    gbest_kwargs = {config.TYPE : config.GLOBAL_BEST, config.OPTIONS : gbest_options}

    #PASSING ALL THIS OPTIONS TO LSTM_PSO applyLSTM_PSO FUNCTION
    cost, pos, optimizer = LSTM_PSO.applyLSTM_PSO(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, batch_size=batch_size,
                                                  numberParticles=numberParticles, iterations=iterations, dimensions=dimensions,
                                                  bounds=bounds, **lbest_kwargs)
    print(cost)
    print(pos)

    #PLOT GRAPHICS ILLUSTRATING THE COST VARIATION AND PARTICLES MOVEMENT AND CONVERGENCE
    plots.plotCostHistory(optimizer=optimizer)
    plots.plotPositionHistory(optimizer=optimizer, xLimits=(minBounds[0], maxBounds[0]),
                              yLimits=(minBounds[1], maxBounds[1]), filename='lstmParticlesPosConvergence.html',
                              xLabel=config.X_LABEL_NEURONS, yLabel=config.Y_LABEL_EPOCHS)

if __name__ == "__main__":
    main()