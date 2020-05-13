from random import seed, uniform, random
from utils import Function, generatePoints
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


def targetFunction(x):
    x1 = x[0]
    x2 = x[1]
    return x1**2 + x2**2 - 0.6

def generateY(f, X):
    y = []
    for xi in X:
        if f(xi) > 0:
            y.append(1)
        else:
            y.append(-1)
    return y

errors = []
for iter in range(0, 1000):

    #DADOS DA AMOSTRA DE TREINO
    #Gerando pontos aleatorios com base na funcao
    X = generatePoints(1000)
    X_with_x0 = [ [1] + x for x in X] ##adicionando bias
    # print(X[0])

    y = generateY(targetFunction, X) 
    y = generateNoise( y)


    w = linear_regression(X_with_x0, y)
    perc = PocketPLA()
    #perc.train(X, y, MAX_ITERATIONS=50, initial_w=w)

  
    #DADOS FORA DA AMOSTRA
    #Gerando pontos aleatorios com base na funcao
    X = generatePoints(1000)
    X_with_x0 = [ [1] + x for x in X] ##adicionando bias
    # print(X[0])

    y = generateY(targetFunction, X) 
    y = generateNoise( y)
    
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
