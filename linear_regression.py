# one dimensional linear regression
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.optim as optim


class LinearRegression(nn.Module):
    def __init__(self):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(LinearRegression,self).__init__()  # 等价与nn.Module.__init__()
        self.linear = nn.Linear(1,1)  # 作用 wx + b 相当于一个线性层

    def forward(self, x):
        output = self.linear(x)
        return output


# create dataset
x_train = np.array(
    [[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59], [2.167], [7.042], [10.791], [5.313],
     [7.997], [3.1]], dtype=np.float32)

y_train = np.array(
    [[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53], [1.221], [2.827], [3.465], [1.65],
     [2.904], [1.3]], dtype=np.float32)

# 将numpy数组转化为torch tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()


def main():

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)  # Adam优化，学习率为1e-5

    num_iterations = 10000

    model.train()
    for i in range(num_iterations):

        if torch.cuda.is_available():
            input = Variable(x_train).cuda()
            target = Variable(y_train).cuda()
        else:
            input = Variable(x_train)
            target = Variable(y_train)

        # print(input.shape)

        # forward
        out = model.forward(input)
        loss = criterion(out,target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(i + 1, num_iterations, loss.item()))

    model.eval()
    model.cpu()
    predict = model(Variable(x_train))
    predict = predict.data.numpy()
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
    plt.plot(x_train.numpy(), predict, label='Fitting Line')
    plt.show()


if __name__ == '__main__':
    main()