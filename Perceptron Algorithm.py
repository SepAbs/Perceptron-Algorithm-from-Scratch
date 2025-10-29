from numpy import arange, array, inner, meshgrid, random
from pandas import read_csv
from time import time
from matplotlib.pyplot import figure, ion, savefig, show, title
from mpl_toolkits.mplot3d import Axes3D
from warnings import filterwarnings
filterwarnings("ignore")

# Setting datapoints and list of some chosen learning rate values.
df, learningRates = read_csv("Q1data.csv"), [0.0000019, 0.00001, 0.123, 0.4352678, 0.5, 0.75, 0.88, 0.9999]
X, Y, Z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
numberPoints, numberDimensions = df.shape
# dataPoints = [([x, y, z], label)...], W = uniformed array with 4 dimensions.
dataPoints, Labels, W = [(array([1] + [df.iloc[Index][Dimension] for Dimension in range(numberDimensions - 1)]), df.iloc[Index][3]) for Index in range(numberPoints)], df.iloc[:, 3], random.rand(numberDimensions,)

def Sign(Number):
    # Retrurning the sign of input number.
    return (-1) ** int((Number < 0) == True)

def Perceptron(dataPoints, W, learningRate):
    optimalW, Index = W, 0
    while Index < numberPoints:
        x, Label = dataPoints[Index]
        if Sign(inner(optimalW, x)) == Label:
            Index += 1
        else:
            optimalW, Index = optimalW + learningRate * Label * x, 0
    return optimalW

# Plotting raw unclassified data points.
ax = figure(figsize = (10, 10)).add_subplot(111, projection = "3d")
ion()
ax.scatter(X, Y, Z, c = Labels, label = Labels)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
savefig("Raw Dataset")
show()

# main()
for Index, learningRate in enumerate(learningRates):
    START = time()
    optimalW = Perceptron(dataPoints, W, learningRate)
    STOP = time()
    print(f"\n\nFor learning rate {learningRate}\nOptimal W is found: W = {optimalW}\n in {STOP - START} UTC time unit\nand the fact that it does classify all data points correctly is {all(Sign(inner(optimalW, dataPoint[0])) == dataPoint[1] for dataPoint in dataPoints)}.")

    # Make data & projection of decision boundary (hyperplane)
    XX, YY = meshgrid(arange(-800, 800), arange(-300, 200))
    ax, Title = figure(figsize = (10, 10)).add_subplot(111, projection = "3d"), f"{Index + 1}th Perceptron Classifier"
    ion()
    ax.scatter(X, Y, Z, c = Labels)
    ax.plot_surface(XX, YY, -1 / optimalW[3] * (optimalW[1] * XX + optimalW[2] * YY + optimalW[0]), alpha = 0.8)
    title(Title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    savefig(Title)
    show()
