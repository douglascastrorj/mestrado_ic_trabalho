from random import seed, uniform, random
from utils import Function, generatePoints, generateY
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

def linear_regression(X, y):
    X_pseudo_inverse = np.linalg.pinv(X)
    w = X_pseudo_inverse.dot(y)
    return w

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
    y = generateY(func, X)

    w = linear_regression(X_with_x0, y)

    # print(w)

    perc = Perceptron()
    perc._w = w


    #Calculando Erro dentro da amostra
    errorCount = 0
    for i in range(0,len(y)):
        x = X[i]
        if perc.classify(x) != y[i]:
            errorCount += 1
    error = errorCount/len(y)
    errors.append(error)

errors = np.array(errors)
print ('Erro dentro da amostra: ', errors.mean())
