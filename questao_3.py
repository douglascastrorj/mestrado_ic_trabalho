import numpy as np

from random import seed, uniform, random
import matplotlib.pyplot as plt

from perceptron import Perceptron
from utils import Function, generatePoints, generateY



iterations = []

for i in range(0, 1000):

    #Gerando funcao aleatoria

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

    #Treinando modelo com perceptron
    perc = Perceptron()
    perc.train(X, y)

    # print( 'Iterations: ', perc.iterations)
    iterations.append(perc.iterations) 


       
    #Plotando dados na amostra de treinamento
    xs = [ x[0] for x in X]
    ys = [ x[1] for x in X]



print('Iteracoes medias para convergir: ' , np.array(iterations).mean())