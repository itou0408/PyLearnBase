import torch.nn as nn


class TradingModel(nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 価格情報が1次元
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # 出力は買い(1)か売り(0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
