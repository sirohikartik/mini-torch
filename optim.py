import numpy as np
class SGD():
    def __init__(self,params,lr,momentum = 0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        for i,p in enumerate(self.params):
            self.velocity[i] = (
                self.momentum * self.velocity[i]
                - self.lr * p.grad
            )
            p.data += self.velocity[i]
        

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)


class Adam():
    def __init__(self,params,lr,a = 0.9, b = 0.99,ep = 0.0001):
        self.params = params
        self.lr = lr
        self.a = a
        self.b = b
        self.vel = [np.zeros_like(p.data) for p in self.params]
        self.mom = [np.zeros_like(p.data) for p in self.params]
        self.ep = ep
        self.t = 0
    def step(self):
        self.t+=1
        for i,p in enumerate(self.params):
            self.mom[i] = self.a* self.mom[i] + (1-self.a)*p.grad
            self.vel[i] = self.b*self.vel[i] + (1-self.b)*np.square(p.grad)
            m = np.divide(self.mom[i],(1-self.a**self.t))
            v = np.divide(self.vel[i],(1-self.b**self.t))
            p.data = p.data - self.lr*(np.divide(m,(np.sqrt(v)+self.ep)))
            
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)
        




