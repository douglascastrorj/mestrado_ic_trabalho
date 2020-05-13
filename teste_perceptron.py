from random import seed, uniform, random
import matplotlib.pyplot as plt

from perceptron import Perceptron
from utils import Function, generatePoints, generateY




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
X = generatePoints(100)
y = generateY(func, X)


#Treinando modelo com perceptron
perc = Perceptron()
perc.train(X, y)

print (perc._w)
print( 'Iterations: ', perc.iterations)



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