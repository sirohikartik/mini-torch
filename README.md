# mini-torch


mini-torch is a minimal deep learning framework built from scratch in Python.
It implements reverse-mode automatic differentiation, a dynamic computation
graph, and a basic neural network training loop.

The project is designed for learning and experimentation, not performance.

---

## Features

- Custom `Tensor` class with:
  - Dynamic computation graph
  - Reverse-mode automatic differentiation
  - Gradient accumulation
- Supported operations:
  - Addition and subtraction
  - Matrix multiplication
  - ReLU activation
  - Sum reduction
- Backpropagation via topological sorting
- SGD optimizer with momentum
- Simple multi-layer perceptron (MLP)
- End-to-end training loop

---

## Example

```python
from tensor import Tensor
import numpy as np

a = Tensor([[1.0, 2.0]])
b = Tensor([[3.0], [4.0]])

c = a @ b
loss = c.sum()
loss.backward()

print(a.grad)
print(b.grad)
```


