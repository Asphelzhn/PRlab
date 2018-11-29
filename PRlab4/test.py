import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

class SimpleNet(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer1_2 = nn.Linear(n_hidden_1, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = F.sigmoid(self.layer1(x))
        x = F.dropout(x, training=self.training)
        x = F.sigmoid(self.layer1_2(x))
        x = F.dropout(x, training=self.training)

        x = F.sigmoid(self.layer2(x))
        x = F.dropout(x, training=self.training)
        x = self.layer3(x)
        return x


def main():
    testdata = np.loadtxt("TestSamples.csv", dtype=np.float, delimiter=",")
    num, dim = np.shape(testdata)
    net = SimpleNet(81, 500, 200, 10)
    net.load_state_dict(torch.load("final.pth"))
    val_out = net(torch.from_numpy(testdata).type(torch.FloatTensor))
    category = torch.softmax(val_out, dim=1).detach().numpy().argmax(axis=1).flatten().tolist()
    sum = 0
    np.savetxt('result.csv', category, delimiter=',',fmt="%d")
    print(category)



if __name__=='__main__':
    main()

