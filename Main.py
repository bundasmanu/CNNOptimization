import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import numpy
from sklearn.preprocessing import MinMaxScaler
import pyswarms as ps
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, GlobalMaxPooling1D

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

def getBestNumberOfNodesAndKernelForCNN(X, Y, params):

    '''
    The objetive of this function is described in objectiveFunctionPSO() function
    Ref: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
    :param X: array with [X_train, X_test]
    :param Y: array with [Y_train, Y_test]
    :param params: represents particle parameters --> 2 parameters --> first one input node value and second one kernel length
    :return: error (prevision of samples) of a Particle
    '''

    #MODEL CREATION --> SEQUENTIAL OPTION, PERMITES TO CREATES A BUILD OF A CNN MODEL
    model = Sequential()
    model.add(Conv1D(params[0],kernel_size=params[1], activation='relu', input_shape=X[0].shape[1]))
    model.add(GlobalMaxPooling1D()) #PODIA TER FEITO APENAS MAXPOOLING E TER DEFINIDO UM VALOR PARA A MATRIX, MAS COMO O EXEMPLO É SIMPLES PENSO QUE ASSIM É MELHOR
    model.add(Dense(3, activation='softmax')) #THREE THE NUMBER OF OUPUTS OF PROBLEM --> FULLY CONNECTED LAYER

    #COMPILE THE MODEL --> 3 ATTRIBUTES https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
    #ADAM IS USED TO CONTROL THE RATE LEARNING OF WEIGHTS OF CNN
    #‘categorical_crossentropy’ for our loss function
    #NAO PRECISAVA DE USAR NADA DISTO --> SERVE APENAS PARA MELHOR ANÁLISE
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X[0], Y[0], epochs=20)

    predictions = model.predict(X[1]) #RETURNS A NUMPY ARRAY WITH PREDICTIONS

    #WELL, I NEED TO COMPARE THE PREDICTIONS WITH REAL VALUES
    numberRights = 0
    for i in range(len(Y[1])):
        if Y[1][i] == predictions[i]:
            numberRights = numberRights + 1

    hitRate = numberRights/Y[1].shape #HIT PERCENTAGE OF CORRECT PREVISIONS

    #LOSS FUNCTION --> I VALORIZE PARTICLES IF MINOR VALUES OF NODES AND KERNEL'S BUT I PUT MORE IMPORTANCE IN PARTICLES THAT GIVE MORE ACCURACY RATE
    loss = (1.0 * (1.0 - (1/params[0]))) + (0.5 * (1.0 - (1/params[1]))) + (2.0 * (1- hitRate)) #I GIVE THIS WEIGHTS IN ORDER TO OBTAIN GOOD SOLUTIONS, WITH LOW VALUE PARAMETERS, THUS REDUCING COMPUTATIONAL POWER

    return loss

def objectiveFunctionPSO():

    '''
    This function represents the objetive function used in the calculus
    of the error of application of a CNN. This objetive function considers
    the nodes dimension and kernel evaluation. All particles have specific
    values of this two attributes, and the main objetive is to find the best
    combination of this two attributes, minimizing the error of prevision (I
    considered i simple dataset).
    :return: best cost (minor error) and best Particle founded
    '''
    return None

def main():


    '''
        GET ALL PARTIES NEEDED FROM DATASET
    '''

    X, Y, x_train, x_test, y_train, y_test = getDataset(25) #TEST PERCENTAGE IS 25%

    print(X)
    print(Y)

    #PSO FORMULATION
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    dimensions = 2 # IN FIRST DIMENSION I HAVE REPRESENTED NUMBER OF NODES ON A CNN LAYER, AND IN SECOND DIMENSION KERNEL USED ON CNN LAYER (MATRIX)
    bounds = (64, 30) #MAX DIMENSIONS LIMITS RESPECTIVELY FOR NUMBER OF NODES OF A CNN LAYER AND KERNEL DIMENSION

    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options,bounds=bounds)

    cost, pos = optimizer.optimize(objectiveFunctionPSO, iters=100)

if __name__ == "__main__":
    main()