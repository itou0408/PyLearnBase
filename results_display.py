import matplotlib.pyplot as plt


def display_results(train_history, test_history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_history['price_history'], label='Price')
    plt.plot(test_history['value_history'], label='Value', color='orange')
    plt.title('Price and Total Value Over Time')
    plt.xlabel('Day')
    plt.ylabel('Value/Price')
    plt.legend()

    plt.show()
