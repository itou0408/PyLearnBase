from model import TradingModel
from dataset import create_dataset
from train import train_model
from evaluate import evaluate_model
from results_display import display_results


def main():
    train_loader = create_dataset()
    model = TradingModel()
    train_history = train_model(model, train_loader)
    test_history = evaluate_model(model, train_loader)  # 本来は別のテストデータセットを使用
    display_results(train_history, test_history)


if __name__ == "__main__":
    main()
