# モデル関連のパラメータ
model_params = {
    'input_size': 1,
    'hidden_size': 64,
    'output_size': 1,
}

# トレーニング関連のパラメータ
training_params = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
}

# データセット関連のパラメータ
dataset_params = {
    'initial_price': 100,
    'amplitude': 20,
    'frequency': 0.02,
    'data_length': 1000,
}

# 取引関連のパラメータ
trading_params = {
    'initial_value': 1000000,  # 初期資金
}
