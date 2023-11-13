from model import TradingModel
from dataset import create_dataset
from train import train_model
from evaluate import evaluate_model

# データセットの生成
train_loader, test_loader = create_dataset()

# モデルの初期化
model = TradingModel()

# モデルのトレーニング
train_model(model, train_loader)

# モデルの評価
evaluate_model(model, test_loader)
