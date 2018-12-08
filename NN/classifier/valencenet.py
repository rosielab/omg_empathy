import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

class ValenceNet(nn.Module):
    def __init__(self):
        super(ValenceNet, self).__init__()

        self.fc1 = nn.Linear(in_features=6, out_features=32)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=32, out_features=3)

        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.fc(x)
        return x
