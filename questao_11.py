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


def transformToAnotherSpace(X):
    transformed = []
    for x in X:
        x1 = x[1]
        x2 = x[2]
        transformed.append([1, x1, x2, x1*x2, x1**2, x2**2])
    return transformed

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

np.set_printoptions(suppress=True)
ws = np.zeros(6)
N = 10000
for i in range(0, N):
    #Gerando pontos aleatorios com base na funcao
    X = generatePoints(1000)
    X_with_x0 = [ [1] + x for x in X] ##adicionando bias
    X_with_x0 = transformToAnotherSpace(X_with_x0)
    # print(X[0])

    y = generateY(targetFunction, X) 
    # print(y)
    y = generateNoise( y)


    w = linear_regression(X_with_x0, y)
    # print(w)
    ws = ws + np.array(w)

print('W Medio ', ws/N)