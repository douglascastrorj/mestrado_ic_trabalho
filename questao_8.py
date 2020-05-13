from random import seed, uniform, random
from utils import Function, generatePoints, generateY
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

def linear_regression(X, y):
    X_pseudo_inverse = np.linalg.pinv(X)
    w = X_pseudo_inverse.dot(y)
    return w

iterations = []
for iter in range(0, 1000):

    func = Function()
    x0 = uniform(-1,1)
    y0 = uniform(-1,1)
    x1 = uniform(-1,1)
    y1 = uniform(-1,1)
    func.buildFromPoints( x0, y0, x1, y1)
    # func._print()

    #Gerando pontos aleatorios com base na funcao
    X = generatePoints(10)
    X_with_x0 = [ [1] + x for x in X] ##adicionando bias
    y = generateY(func, X)


    w = linear_regression(X_with_x0, y)

    print('training perceptron: #', iter)

    perc = Perceptron()
    perc.train(X, y, MAX_ITERATIONS=10000, initial_w=w)
    print('Levou ', perc.iterations, ' iteracoes para convergir')
    iterations.append(perc.iterations)

    
    # # Plotando dados na amostra de treinamento
    # xs = [ x[0] for x in X]
    # ys = [ x[1] for x in X]

    # #plot data
    # plt.scatter(xs, ys, [1 for i in range(0, len(ys))]) #ultimo parametro e o tamanho das bolinhas
    # #plot target function
    # plt.plot([-1, 1], [func.calculate(-1), func.calculate(1)], label='Target Function')
    # #plot perceptron boundary
    # plt.plot([-1, 1], [perc.getValue(-1), perc.getValue(1)], 'r--', label='Perceptron Decision Boundary')
    # plt.ylim((-1, 1))
    # plt.xlim((-1, 1))

    # plt.title('Comparativo utilizando amostras de treinamento')
    # plt.legend()
    # plt.show()


   

iterations = np.array(iterations)
print('N de iteracoes media para convergir: ', iterations.mean())


# N de iteracoes media para convergir:  3.24