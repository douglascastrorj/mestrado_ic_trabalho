from random import seed, uniform, random
from utils import Function, generatePoints, generateY
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from pocket import PocketPLA


def linear_regression(X, y):
    X_pseudo_inverse = np.linalg.pinv(X)
    w = X_pseudo_inverse.dot(y)
    return w
    
def generateNoise(y, percent = 0.1):
    noiseY = [yi for yi in y]
    qtd = len(y) * percent
    for i in range(0, int(qtd)):
        if noiseY[i] == -1:
            noiseY[i] = 1
        else:
            noiseY[i] = -1
    return noiseY


def plot(X, y, perc, func):
    #Plotando dados
    xs_c1 = []
    ys_c1 = []

    xs_c2 = []
    ys_c2 = []
    for i in range(0, len(y)):
        if(y[i] == -1):
            xs_c1.append(X[i][0])
            ys_c1.append(X[i][1])
        else:
            xs_c2.append(X[i][0])
            ys_c2.append(X[i][1])

    #plot data
    plt.scatter(xs_c1, ys_c1,  [3 for i in range(0, len(ys_c2))], marker='o')
    plt.scatter(xs_c2, ys_c2, [3 for i in range(0, len(ys_c2))], marker='^')

    #plot target function
    plt.plot([-1, 1], [func.calculate(-1), func.calculate(1)], label='Target Function')
    #plot perceptron boundary
    plt.plot([-1, 1], [perc.getValue(-1), perc.getValue(1)], 'r--', label='Perceptron Boundary')

    plt.ylim((-1, 1))
    plt.xlim((-1, 1))

    plt.title('Erro fora da amostra')
    plt.legend()
    plt.show()


errors = []
for iter in range(0, 1000):
    func = Function()
    x0 = uniform(-1,1)
    y0 = uniform(-1,1)
    x1 = uniform(-1,1)
    y1 = uniform(-1,1)
    func.buildFromPoints( x0, y0, x1, y1)
    # func._print()

    #Gerando pontos aleatorios com base na funcao
    X = generatePoints(100)
    X_with_x0 = [ [1] + x for x in X] ##adicionando bias
    y = generateNoise( generateY(func, X) )

    # print(y)

    w = linear_regression(X_with_x0, y)
    perc = PocketPLA()
    perc.train(X, y, MAX_ITERATIONS=10, initial_w=w)

    #gerando dados fora da amostra de treino
    X = generatePoints(1000)
    X_with_x0 = [ [1] + x for x in X] ##adicionando bias
    y = generateY(func, X) #generateNoise( generateY(func, X) )

    plot(X,y,perc,func)  

    #Calculando Erro Fora da amostra
    errorCount = 0
    for i in range(0,len(y)):
        x = X[i]
        if perc.classify(x) != y[i]:
            errorCount += 1
    error = errorCount/len(y)
    errors.append(error)


errors = np.array(errors)
print ('Erro fora da amostra: ', errors.mean())
