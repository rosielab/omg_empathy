import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

class ValenceNet(nn.Module):
    def __init__(self, lstm_enabled=True):
        super(ValenceNet, self).__init__()

        self.fc1 = nn.Linear(in_features=6, out_features=32)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.drop2 = nn.Dropout(p=0.2)
        # self.fc3 = nn.Linear(in_features=32, out_features=64)
        # self.drop3 = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(32, 32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=32, out_features=1)

        self.relu = nn.ReLU() 
        self.reset_hidden_states()
        self.lstm_enabled = lstm_enabled

    def reset_hidden_states(self, size=1, zero=True):
        if zero == True:
            self.hx = Variable(torch.zeros(1, size, 32))
            self.cx = Variable(torch.zeros(1, size, 32))
        else:
            self.hx = Variable(self.hx.data)
            self.cx = Variable(self.cx.data)

        if next(self.parameters()).is_cuda == True:
            self.hx = self.hx.cuda()
            self.cx = self.cx.cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop2(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.drop3(x)
        if (self.lstm_enabled):
            x, (self.hx, self.cx) = self.lstm(x, (self.hx, self.cx))
        x = self.fc(x)
        return x
