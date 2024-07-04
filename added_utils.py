import matplotlib.pyplot as plt

# zero to mastery pytorch course by mdbourke (github)
def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

def plot_loss_curves(results):
    acc = results["train_acc"]
    test_accuracy = results["test_acc"]
    class_accuracy = results["class_acc"]
    epochs = range(len(results["train_acc"]))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="train_accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.title("Training Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.plot(epochs, class_accuracy, label="class_accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.show()
