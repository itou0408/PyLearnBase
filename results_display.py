import matplotlib.pyplot as plt


def display_results(train_history, test_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(['Final Value'], [test_history['final_value']], color='orange')
    plt.title('Final Trading Value')
    plt.ylabel('Value')

    plt.show()
