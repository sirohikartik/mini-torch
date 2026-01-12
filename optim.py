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


