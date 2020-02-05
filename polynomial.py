# polynomial
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.optim as optim

# make_features, f(x), get_batch these functions are aim to generate the data.

def make_features(x):
    x = x.unsqueeze(1)
    return(torch.cat([x ** i for i in range(1, 4)], 1))  # torch.cat(,0/1) 0是将张量竖着拼接，1是横着


def f(x):
    w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
    b_target = torch.FloatTensor([0.9])
    return x.mm(w_target) + b_target[0]


def get_batch(batch_size = 32):
    random = torch.randn(batch_size)
    random = np.sort(random)
    random = torch.tensor(random)
    x = make_features(random)
    y = f(x)

    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)


class polynomial_model(nn.Module):
    def __init__(self):
        super(polynomial_model,self).__init__()
        self.poly = nn.Linear(3,1)  # 定义一个三次多项式，x最高次为3，常数项为1

    def forward(self,x):
        output = self.poly(x)
        return output


def main():
    w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
    b_target = torch.FloatTensor([0.9])

    if torch.cuda.is_available():
        model = polynomial_model().cuda()
    else:
        model = polynomial_model()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch = 0
    while True:
        # get data
        batch_x, batch_y = get_batch()

        # forward
        out = model(batch_x)
        loss = criterion(out,batch_y)
        print_loss = loss.item()

        # reset gradients
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update parameters
        optimizer.step()

        epoch += 1
        if print_loss < 1e-3:
            break

    print("Loss: {:.6f}  after {} batches".format(loss.item(), epoch))

    print(
        "==> Learned function: y = {:.2f} + {:.2f}*x + {:.2f}*x^2 + {:.2f}*x^3".format(model.poly.bias[0],
                                                                                       model.poly.weight[0][0],
                                                                                       model.poly.weight[0][1],
                                                                                       model.poly.weight[0][2]))
    print("==> Actual function: y = {:.2f} + {:.2f}*x + {:.2f}*x^2 + {:.2f}*x^3".format(b_target[0], w_target[0][0],
                                                                                        w_target[1][0], w_target[2][0]))

    predict = model(batch_x)

    batch_x = batch_x.cpu()
    batch_y = batch_y.cpu()
    predict = predict.cpu()
    predict = predict.data.numpy()

    x = batch_x.numpy()[:,0]
    plt.plot(x, batch_y.numpy(), 'ro')

    plt.plot(x, predict,'b')
    plt.show()


if __name__ == '__main__':
    main()
