from random import seed, uniform, random
from utils import Function, generatePoints, generateY
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

def linear_regression(X, y):
    X_pseudo_inverse = np.linalg.pinv(X)
    w = X_pseudo_inverse.dot(y)
    return w

func = Function()
x0 = uniform(-1,1)
y0 = uniform(-1,1)
x1 = uniform(-1,1)
y1 = uniform(-1,1)
func.buildFromPoints( x0, y0, x1, y1)
# func._print()

#Gerando pontos aleatorios com base na funcao
X = generatePoints(100)
X_with_x0 = [ [1] + x for x in X]
y = generateY(func, X)

w = linear_regression(X_with_x0, y)

print(w)

perc = Perceptron()
perc._w = w


#Plotando dados

xs = [ x[0] for x in X]
ys = [ x[1] for x in X]

#plot data
plt.plot(xs, ys, 'bo')
#plot target function
plt.plot([-1, 1], [func.calculate(-1), func.calculate(1)], label='Target Function')
#plot perceptron boundary
plt.plot([-1, 1], [perc.getValue(-1), perc.getValue(1)], 'r--', label='Perceptron Boundary')
plt.legend()
plt.show()