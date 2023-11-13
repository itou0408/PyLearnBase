import torch


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predicted = (torch.sigmoid(output) > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy}')
    return {'accuracy': accuracy}
