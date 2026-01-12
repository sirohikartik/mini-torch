import numpy as np

class Tensor():
    def __init__(self, data, children=(), op=''):
        self.data = np.array(data)
        self.children = children
        self.op = op
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad.sum()
            
        out._backward = _backward
        return out

    def __mul__(self, other):  # matrix multiply
        out = Tensor(self.data @ other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(self.data,0),(self,),'r')

        def _backward():
            self.grad += (self.data >0 )* out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(t):
            if t not in visited:
                visited.add(t)
                for child in t.children:
                    build(child)
                topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

