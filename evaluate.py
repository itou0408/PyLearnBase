import torch
from config import trading_params


def evaluate_model(model, test_loader):
    model.eval()
    total_value = trading_params['initial_value']
    holding_dollars = 0
    price_history = []
    value_history = []

    with torch.no_grad():
        for data, _ in test_loader:
            output = model(data)
            decisions = torch.sigmoid(output).squeeze()

            for i in range(data.size(0)):
                current_price = data[i].item()
                price_history.append(current_price)
                decision = decisions[i].item()

                if decision > 0.5:  # 買いの場合
                    holding_dollars += total_value / current_price
                    total_value = 0
                elif decision <= 0.5 and holding_dollars > 0:  # 売りの場合
                    total_value += holding_dollars * current_price
                    holding_dollars = 0

                value_history.append(
                    total_value + holding_dollars * current_price)

    return {'price_history': price_history, 'value_history': value_history}
