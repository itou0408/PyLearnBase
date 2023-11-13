import torch.nn as nn
from config import model_params

class TradingModel(nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()
        self.fc1 = nn.Linear(model_params['input_size'], model_params['hidden_size'])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(model_params['hidden_size'], model_params['output_size'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
