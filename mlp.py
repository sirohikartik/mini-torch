import numpy as np
import math
from tensor import Tensor
from optim import SGD
class MLP():
    def __init__(self,input,output):
        self.fc1 = Tensor(np.random.randn(input,10))
        self.b1 = Tensor(np.random.randn(1,10))
        self.fc2 = Tensor(np.random.randn(10,output))
        self.b2 = Tensor(np.random.randn(1,output))
    def forward(self,x):
        o = x * self.fc1 + self.b1
        o = o.relu()
        o = o * self.fc2 + self.b2
        return o

    def params(self):
        return [self.fc1,self.fc2]

# collecting training data let's approximate sin(x)

train = np.arange(1,1000)
train = np.array([[i] for i in train])
label = np.array([math.sin(i[0]) for i in train])


model = MLP(1,1)

optimizer = SGD(model.params(),0.1)
epochs = 100
for epoch in range(epochs):
    losses = []
    for i in range(len(train)):
        optimizer.zero_grad()
        input = train[i]
        output = label[i]

        pred = model.forward(Tensor([input]))
        loss = pred - Tensor(output)
        loss = (loss*loss).sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
    print(f"{epoch}/{epochs} {np.array(losses).mean()}")

input = [3.14]
print(model.forward(Tensor([input])).data)
