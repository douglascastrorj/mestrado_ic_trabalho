import numpy as np

from random import seed, uniform, random
import matplotlib.pyplot as plt

from perceptron import Perceptron
from utils import Function, generatePoints, generateY


errors = []

for k in range(0, 1000):

    #Gerando funcao target aleatoria
    func = Function()
    # func.set(1, 0)
    x0 = uniform(-1,1)
    y0 = uniform(-1,1)
    x1 = uniform(-1,1)
    y1 = uniform(-1,1)
    func.buildFromPoints( x0, y0, x1, y1)
    # func._print()

    #Gerando pontos aleatorios com base na funcao
    X = generatePoints(100)
    y = generateY(func, X)

    perc = Perceptron()
    perc.train(X, y)

    #gerando novos pontos fora da amostra
    X = generatePoints(1000)
    y = generateY(func, X)

    h = [ perc.classify(x) for x in X]

    # print('Target ',y)
    # print('Predicted ',h)

    errorCount = 0
    for i in range(0, len(y)):
        if y[i] != h[i]:
            errorCount += 1

    errProb = float(errorCount)/len(y)
    errors.append(errProb)

print('P(f(x) != g(x)) = ', np.array(errors).mean())