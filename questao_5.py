from random import seed, uniform, random
import matplotlib.pyplot as plt

from perceptron import Perceptron
from utils import Function, generatePoints, generateY, plot




#Gerando funcao aleatoria

func = Function()
# func.set(1, 0)
x0 = uniform(-1,1)
y0 = uniform(-1,1)
x1 = uniform(-1,1)
y1 = uniform(-1,1)
func.buildFromPoints( x0, y0, x1, y1)

func._print()


#Gerando pontos aleatorios com base na funcao
X = generatePoints(10)
y = generateY(func, X)


#Treinando modelo com perceptron com 10 pontos
perc10 = Perceptron()
perc10.train(X, y)


plot(X,y, perc10, func, title='Ein N = 10')

X = generatePoints(100)
y = generateY(func, X)


#Treinando modelo com perceptron com 100 pontos
perc100 = Perceptron()
perc100.train(X, y)

plot(X,y, perc100, func, title='Ein N = 100')

# gerando dados fora da amostra
X = generatePoints(1000)
y = generateY(func, X)


plot(X,y, perc10, func, title='Eout N = 10')
plot(X,y, perc100, func, title='Eout N = 100')

