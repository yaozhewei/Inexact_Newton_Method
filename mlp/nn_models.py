import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, hidden=128):
        super(mlp, self).__init__()
        n = int(hidden)
        self.fc1 = nn.Linear(784, n)
        self.fc2 = nn.Linear(n, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.softplus(self.fc1(x))
        x = self.fc2(x)
        return x