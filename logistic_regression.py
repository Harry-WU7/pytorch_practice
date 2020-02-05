# logistic regression
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.optim as optim
import h5py
import opt_utils

# 猫数据集
'''
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_y.shape[1]  # 训练集里图片的数量。
m_test = test_set_y.shape[1]  # 测试集里图片的数量。
num_px = train_set_x_orig.shape[1]  # 训练、测试集里面的图片的宽度和高度（均为64x64）。

# 现在看一看我们加载的东西的具体情况
print("训练集的数量: m_train = " + str(m_train))
print("测试集的数量 : m_test = " + str(m_test))
print("每张图片的宽/高 : num_px = " + str(num_px))
print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print("训练集_标签的维数 : " + str(train_set_y.shape))
print("测试集_图片的维数: " + str(test_set_x_orig.shape))
print("测试集_标签的维数: " + str(test_set_y.shape))

# 将训练集的维度降低并转置。
train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print ("测试集_标签的维数 : " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255
'''


# 点数据集
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x, train_y = opt_utils.load_dataset(is_plot=True)
plt.show()


# define model
class Logistic_regression(nn.Module):
    def __init__(self):
        super(Logistic_regression,self).__init__()
        self.linear = nn.Linear(300,300)
        self.sm = nn.Sigmoid()

    def forward(self,x):
        x = self.linear(x)  # 2*300*300*1
        x = self.sm(x)
        return x


train_x = torch.from_numpy(train_x)  # size:(2,300)
train_y = torch.from_numpy(train_y)  # size:(1,300) 是标签


if torch.cuda.is_available():
    model = Logistic_regression().cuda()
else:
    model = Logistic_regression()


def main():

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    num_epochs = 100000

    model.train()
    for i in range(num_epochs):
        if torch.cuda.is_available():
            input = Variable(train_x).cuda()
            target = Variable(train_y).cuda()
        else:
            input = Variable(train_x)
            target = Variable(train_y)

        # input = torch.tensor(input, dtype=torch.float32)
        out = model.forward(input)
        loss = criterion(out,target)

        # reset 0
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update parameter
        optimizer.step()

        if (i+1) % 10000 == 0:
            print("the number of iteration: ",i+1,"the loss is: ",str(loss.item()))

    model.eval()
    model.cpu()
    predict = model(Variable(train_x))
    predict = predict.data.numpy()
    plt.plot(train_x.numpy(), train_y.numpy(), 'ro', label='Original data')
    plt.plot(train_x.numpy(), predict, label='Fitting Line')
    plt.show()


if __name__ == '__main__':
    main()
