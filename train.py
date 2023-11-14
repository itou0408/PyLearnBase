import torch
import torch.optim as optim
import torch.nn as nn
from config import training_params


def train_model(model, train_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=training_params['learning_rate'])
    history = {'loss': []}

    for epoch in range(training_params['num_epochs']):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader.dataset)
        history['loss'].append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')

    return history
