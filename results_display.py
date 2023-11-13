import matplotlib.pyplot as plt


def display_results(train_history, test_history):
    plt.figure(figsize=(12, 5))

    # トレーニング損失
    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 評価精度
    plt.subplot(1, 2, 2)
    plt.bar(['Accuracy'], [test_history['accuracy']], color='orange')
    plt.title('Evaluation Accuracy')
    plt.ylabel('Accuracy')

    plt.show()
