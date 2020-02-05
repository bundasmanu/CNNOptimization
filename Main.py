import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import numpy
from sklearn.preprocessing import MinMaxScaler
import pyswarms as ps
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Dropout
import keras
import WeightsUpgradeOnTraining, WeightsInitializer

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
    considered i simple dataset).
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
    y_train = y_train.reshape(int((len(y_train)/3)), 3) #3 POSSIBLE RESULTS THEN 3 TIME STEMPS
    y_test = y_test.reshape((int(len(y_test)/3)), 3)

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
    initializer = WeightsInitializer.WeightsInitializer()

    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, time_stemps, features), return_sequences=True, stateful=True,
                   kernel_initializer=initializer.initInputMat(inputParticleWeights), recurrent_initializer= initializer.initRecMat(recurrentParticleWeigths)))
    #model.add(Dropout(0))
    model.add(Dense(3)) #3 OUTPUTS
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # INITIALIZATION OF CALLBACK, AND DEFINE THEM IN MODEL FITNESS
    weightsCallback = WeightsUpgradeOnTraining.WeightsUpgradeOnTraining(particleWeights=100, numberOfNeurons=neurons) #PARTICLES WEIGHTS IS TEMPORARY

    #FITTING MODEL
    model.fit(x_train, y_train, epochs=5, batch_size=batch_size, shuffle=False, callbacks=[weightsCallback])#BATCH_SIZE AND SHUFFLE BECAUSE TIME_STEPS DIFFERENT FROM 1

    predictions = model.predict(x_test, batch_size=batch_size)  # RETURNS A NUMPY ARRAY WITH PREDICTIONS

    # WELL, I NEED TO COMPARE THE PREDICTIONS WITH REAL VALUES
    numberRights = 0
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            indexMaxValue = numpy.argmax(predictions[i][j], axis=0)
            if indexMaxValue == numpy.argmax(y_test[i][j], axis=0): #COMPARE INDEX OF MAJOR CLASS PREDICTED AND REAL CLASS
                numberRights = numberRights + 1

    hitRate = numberRights / (y_test.shape[0]*y_test.shape[1])  # HIT PERCENTAGE OF CORRECT PREVISIONS
    print(hitRate)

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
    loss = [objectiveFunctionLSTM(x_train, x_test, y_train, y_test, neurons, batch_size, time_stemps, features) for i in range(nParticles)] #FALTA AINDA PASSAR OS DADOS DE UMA PARTICULA, MAS POR AGORA NAO INTERESSA --> 1º NECESSÁRIO COLOCAR O MODELO FUNCIONAL
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

    neurons = 10
    batch_size = 10 #I HAVE 150 SAMPLES, AND TO REDUCE THE COMPUTACIONAL REQUIREMENTS, I DEFINE 3 TIMES TO LEARN (50*3) = 150
    time_stemps = 3 #EVERY VALUES ON EVERY ATTRIBUTES HAVE THE SAME FORMAT AND LENGHT --> FLOAT VALUES LIKE: 1.2, LSTM NEEDS TO LOOK AT THIS 3 PIECES
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

if __name__ == "__main__":
    main()