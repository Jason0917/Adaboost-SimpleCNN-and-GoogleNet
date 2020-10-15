import torch

w1 = torch.tensor(0.15, requires_grad=True)
w2 = torch.tensor(0.2, requires_grad=True)
w3 = torch.tensor(0.25, requires_grad=True)
w4 = torch.tensor(0.3, requires_grad=True)
w5 = torch.tensor(0.4, requires_grad=True)
w6 = torch.tensor(0.45, requires_grad=True)
w7 = torch.tensor(0.5, requires_grad=True)
w8 = torch.tensor(0.55, requires_grad=True)

i1 = torch.tensor(0.05)
i2 = torch.tensor(0.1)
b1 = torch.tensor(0.35)
b2 = torch.tensor(0.6)
o1_real = torch.tensor(0.01)
o2_real = torch.tensor(0.99)

lr = 0.5
epoch = 5000

for t in range(epoch):
    h1 = i1*w1 + i2*w2 + b1
    h2 = i1*w3 + i2*w4 + b1
    h1 = torch.sigmoid(h1)
    h2 = torch.sigmoid(h2)

    o1 = h1*w5 + h2*w6 + b2
    o2 = h1*w7 + h2*w8 + b2
    o1 = torch.sigmoid(o1)
    o2 = torch.sigmoid(o2)

    loss = 0.5*(o2_real-o2).pow(2) + 0.5*(o1_real-o1).pow(2)
    loss.backward()
    #print(loss)

    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        w3 -= lr * w3.grad
        w4 -= lr * w4.grad
        w5 -= lr * w5.grad
        w6 -= lr * w6.grad
        w7 -= lr * w7.grad
        w8 -= lr * w8.grad
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
        w4.grad.zero_()
        w5.grad.zero_()
        w6.grad.zero_()
        w7.grad.zero_()
        w8.grad.zero_()

h1 = i1*w1 + i2*w2 + b1
h2 = i1*w3 + i2*w4 + b1
h1 = torch.sigmoid(h1)
h2 = torch.sigmoid(h2)

o1 = h1*w5 + h2*w6 + b2
o2 = h1*w7 + h2*w8 + b2
o1 = torch.sigmoid(o1)
o2 = torch.sigmoid(o2)

print(o1)
print(o2)


