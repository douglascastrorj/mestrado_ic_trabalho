


class Perceptron:

    def __init__(self):
        self._w = []
        self.iterations = 0
        self.convergiu = False

    def multV(self, w, x):
        total = 0
        for i in range(0, len(w)):
            total += w[i] * x[i]
        return total

    def mult(self, vector, number):
        v = [ 0 for i in range(0, len(vector))]
        for i in range(0, len(vector)):
            v[i] = vector[i] * number
        return v

    def sumV(self, v1, v2):
        v = [ 0 for i in range(0, len(v1))]
        for i in range(0,len(v1)):
            v[i] = v1[i] + v2[i]
        return v

    def sign(self, x):
        if(x > 0): 
            return 1
        else: 
            return -1

    
    def getValue(self, x):
        b = self._w[0]
        w1 = self._w[1]
        w2 = self._w[2]
        y = (-1*(b / w2) / (b / w1))*x + (-1*b / w2)
        return y

    def classify(self, x):
        wx = self.multV(self._w,[1] + x)
        hx = self.sign(wx)
        return hx

    def train(self, data, y, MAX_ITERATIONS = 1000, initial_w = []):
        self.convergiu = False

        if len(data) == 0:
            return

        x = [ [1] + xi for xi in data] # [x0 = 1, x1, x2, ..., xn]
        if(len(initial_w) == 0):
            w = [0 for i in range(0, len(x[0]))]    
        else:
            w = initial_w

        iterations = 0
        while True:
            iterations = iterations + 1
            erro = False
            for i in range(0, len(x)):
                xi = x[i]
                wx = self.multV(w,xi)
                hx = self.sign(wx)

                if(hx != y[i]):
                    erro = True
                    yx = self.mult(xi, y[i])
                    w = self.sumV(w, yx) #w = w + yi*xi
            if erro == False:
                self.convergiu = True
                break
            if iterations > MAX_ITERATIONS:
                print('Atingiu maximo de iteracoes')
                break
        
        self.iterations = iterations
        self._w = w


