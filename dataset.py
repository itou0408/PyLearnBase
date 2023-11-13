import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def create_dataset():
    # ダミーデータの生成
    x = np.random.rand(100, 1)  # 100日分の価格データ
    y = np.random.randint(0, 2, (100, 1))  # 100日分の売買判断（0または1）

    # PyTorchテンソルに変換
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # データセットとデータローダーの作成
    dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=10)

    return train_loader, test_loader
