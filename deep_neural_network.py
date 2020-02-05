# deep neural networks to test the MNIST dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()

        # 构造三层神经网络，sequential()可看作一个容器
        self.layer_1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True)
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2,),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True)
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(n_hidden_2,out_dim)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x


def main():
    # hyperparameters:
    batch_size = 64
    num_epoch = 20
    learning_rate = 1e-2

    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5],[0.5])])

    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=data_tf, download=True)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

    model = Net(28*28, 300, 100, 10)
    if torch.cuda.is_available():
        model = model.cuda()

    critierion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    epoch = 0
    model.train()
    for data in train_loader:
        img,label = data
        img = img.view(img.size(0), -1)

        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = model(img)
        loss = critierion(out,label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch+=1

        if epoch % 10 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

        model.eval()
        eval_loss = 0
        eval_acc = 0

        for data in test_loader:

            img, label = data
            img = img.view(img.size(0), -1)

            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            loss = critierion(out, label)
            eval_loss += loss.data.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)),eval_acc / (len(test_dataset))))


if __name__ == '__main__':
    main()


