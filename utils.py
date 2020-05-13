from random import seed, uniform, random
import matplotlib.pyplot as plt

class Function:

    def __init__(self):
        self.w = [0, 0]

    def set(self, w0, w1):
        self.w = [w0, w1]
    
    def randomize(self):
        w0 = uniform(-1,1)
        w1 = uniform(-1,1)
        self.set(w0, w1)

    def buildFromPoints(self, x0, y0, x1, y1):
        #y -y0 = m(x-x0)
        m = (y1 - y0) / (x1 - x0)
        # y - y0 = m (x - x0)
        y = -1*m*x0 + y0

        self.w = [m, y]

    def _print(self):
        if self.w[1] >= 0:
            print (self.w[0],'x + ', self.w[1])
        else:
            print (self.w[0],'x - ', self.w[1] * -1)

    def calculate(self,x):
        return self.w[0] * x + self.w[1]



def generatePoints(N):
    X = []
    for i in range(0, N):
        value1 = uniform(-1,1)
        value2 = uniform(-1,1)
        xi = [value1, value2]
        X.append(xi)
    return X

def generateY(function, points):
    y = []
    for point in points:
        if(function.calculate(point[0]) > point[1]):
            y.append(1)
        else:
            y.append(-1)
    
    return y



def plot(X, y, perc, func, title=''):
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

    plt.title(title)
    plt.legend()
    plt.show()
