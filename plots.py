from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Animator
from pyswarms.utils.plotters.formatters import Designer

def plotCostHistory(optimizer):

    '''

    :param optimizer: optimizer object returned in the application/definition of PSO
    :return: matplotlib object --> cost 2d plot
    '''

    try:

        plot = plot_cost_history(cost_history=optimizer.cost_history)

        return plot
    except:
        raise

def plotPositionHistory(optimizer):

    '''

    :param optimizer: optimizer object returned in the application/definition of PSO
    :return: matplotlib object --> position 2d plot
    '''

    try:

        plot = plot_contour(pos_history=optimizer.pos_history,
                            mark=(0,0))#BEST POSSIBLE POSITION MARK (* --> IN GRAPHIC)

        return plot
    except:
        raise

def plot3D(optimizer, xValues, yValues, zValues):

    '''

    :param optimizer: optimizer object returned in the application/definition of PSO
    :param xValues: numpy array : 2d (minValue, maxValue) axis
    :param yValues: numpy array : 2d (minValue, maxValue) axis
    :param zValues: numpy array : 2d (minValue, maxValue) axis
    :return: matplotlib object --> position particles 3d plot
    '''

    try:

        #Obtain a position-fitness matrix using the Mesher.compute_history_3d() method.
        positionHistory_3d = Mesher.compute_history_3d(optimizer.pos_history)

        d = Designer(limits=[xValues, yValues, zValues], label=['x-axis', 'y-axis', 'z-axis'])

        plot3d = plot_surface(pos_history=positionHistory_3d,
                              mesher=Mesher, designer=d,
                              mark=(1,1,zValues[0])) #BEST POSSIBLE POSITION MARK (* --> IN GRAPHIC)

        return plot3d

    except:
        raise