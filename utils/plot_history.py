import matplotlib.pyplot as plt
import numpy as np
def plot_history(history):
    # Plotting Loss
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['real_forged_accuracy'])
    plt.plot(history['individual_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Real/Forged', 'Train Individual'], loc='lower right')

    # Customize x and y ticks for accuracy plot
    plt.xticks(ticks=range(0, len(history['real_forged_accuracy']), 2))  # Change x-axis ticks interval
    plt.yticks(ticks=np.arange(0, 1.1, 0.1))  # Change y-axis ticks interval

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['real_forged_loss'])
    plt.plot(history['individual_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Real/Forged', 'Train Individual'], loc='upper left')

    # Customize x and y ticks for loss plot
    plt.xticks(ticks=range(0, len(history['real_forged_loss']), 2))  # Change x-axis ticks interval
    plt.yticks(ticks=np.arange(0, max(history['real_forged_loss'] + history['individual_loss']) + 0.1, 0.2))  # Change y-axis ticks interval

    plt.show()
