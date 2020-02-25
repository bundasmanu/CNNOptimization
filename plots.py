from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Animator
from pyswarms.utils.plotters.formatters import Designer
import matplotlib.pyplot as plt
from IPython.display import Image
import config

def plotCostHistory(optimizer):

    '''

    :param optimizer: optimizer object returned in the application/definition of PSO
    '''

    try:

        plot_cost_history(cost_history=optimizer.cost_history)

        plt.show()
    except:
        raise

def plotPositionHistory(optimizer, xLimits, yLimits, filename):

    '''

    :param optimizer: optimizer object returned in the application/definition of PSO
    :param xLimits: numpy array (minLimit, maxLimit) of x Axis
    :param yLimits: numpy array (minLimit, maxLimit) of y Axis
    :param filename: name of filename returned by plot_contour (html gif)
    '''

    try:

        d = Designer(limits=[xLimits, yLimits], label=[config.X_LABEL, config.Y_LABEL])
        animation = plot_contour(pos_history=optimizer.pos_history,
                     designer=d)

        animation.save(filename, writer='ffmpeg', fps=10)
        Image(url=filename)

        plt.show()
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