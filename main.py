from model import TradingModel
from dataset import create_dataset
from train import train_model
from evaluate import evaluate_model
from results_display import display_results


def main():
    # データセットの生成
    train_loader, test_loader = create_dataset()

    # モデルの初期化
    model = TradingModel()

    # モデルのトレーニング
    train_history = train_model(model, train_loader)

    # モデルの評価
    test_history = evaluate_model(model, test_loader)

    # 評価結果の表示
    display_results(train_history, test_history)


if __name__ == "__main__":
    main()
