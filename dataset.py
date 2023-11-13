import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import dataset_params

def create_dataset():
    t = np.arange(0, dataset_params['data_length'])
    price = (dataset_params['initial_price'] +
             dataset_params['amplitude'] * np.sin(dataset_params['frequency'] * np.pi * t))

    price_tensor = torch.tensor(price, dtype=torch.float32).view(-1, 1)
    target = torch.randint(0, 2, (1000, 1), dtype=torch.float32)

    dataset = TensorDataset(price_tensor, target)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader
