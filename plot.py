import matplotlib.pyplot as plt
import numpy as np

def plot(metrics, save_dir):
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    for fold in metrics:
        train_acc.append(fold['train_acc'])
        train_loss.append(fold['train_loss'])
        test_acc.append(fold['test_acc'])
        test_loss.append(fold['test_loss'])

    train_acc = np.array(train_acc).mean(axis=0)
    test_acc = np.array(test_acc).mean(axis=0)
    train_loss = np.array(train_loss).mean(axis=0)
    test_loss = np.array(test_loss).mean(axis=0)

    epochs = np.arange(len(train_acc))

    # plot loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label='Train Loss')
    ax1.plot(epochs, test_loss, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()

    # plot acc
    ax2.plot(epochs, train_acc, label='Train Acc')
    ax2.plot(epochs, test_acc, label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics.png')
    plt.close()