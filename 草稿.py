import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.optim as optim

m = torch.nn.Linear(300,300)
a = Variable(torch.randn(2,300))
b = Variable(torch.randn(1,300))

output = m(a)

print(output.shape)