# This script contains utility functions that I often used in this project:

# - spliting the training, validation and test set.
# - The Cost functions of the VAE model (KL divergence and Gaussian Likelihood Error)
# - graphs of the training and validation loss 

# import libraries 
import numpy as np
import pandas as pd
import numpy.random as nrd
import matplotlib.pyplot as plt 

# Pytorch modules 
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

# scikit-learn for create_data_partition function 
from sklearn.model_selection import train_test_split 

# import the configuration file
from models_util import configs


# set the device and seed. Retrieve seed and run the set_seed() for reproducibility 
device = configs.get_device()
seed = configs.get_seed()
configs.set_seed(seed)

def create_data_partition(data: np.array, 
                          test_perc: float,
                          val: bool=False, 
                          val_perc: float = None,
                          random = seed):
    """
    Split the Data into training, validation and test set.
    Kept the caret name of the function from R.
    
    
    """
    train_np, test_np = train_test_split(
    data,
    test_size=test_perc,
    shuffle=True,
    random_state=random
    )

    if val:
        val_perc = val_perc/(1 - test_perc)
        train_np, val_np = train_test_split(
            train_np,
            test_size=val_perc,
            shuffle=True,
            random_state=random
            )
        
        return train_np, val_np, test_np
    
    return train_np, test_np


if __name__ == "__main__":
    print("Run the script locally")
else:
    print(f"Importing {__name__}, running in {device} with seed: {seed}" )




def plot_training_loss_sp(ax, minibatch_losses, num_epochs, averaging_iterations=20, custom_label=''):
    """
    Plots batch and batch-average loss in x axis.
    It returns matplotlib ax plots, so it can be used in subplots.
    Based on Sebastian Racschka's code. 
    
    Parameters:
      ax: matplotlib axis on which to plot.
      minibatch_losses: list/array of loss values per iteration.
      num_epochs: total number of epochs.
      averaging_iterations: window size for computing the running average.
      custom_label: A label for the specific loss type.
    """
    iter_per_epoch = len(minibatch_losses) // num_epochs

    # Plot raw loss
    ax.plot(range(len(minibatch_losses)),
            minibatch_losses,
            label=f'Minibatch Loss{custom_label}',
            color="green", alpha=0.8)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')

    # Set y-axis limits if losses are positive (adjust based on later iterations)
    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    # set y-axis limits based on prior knowledge of losses 
    if np.min(minibatch_losses) > 0:
        ax.set_ylim([
            np.min(minibatch_losses[num_losses:]) * 0.98,
            np.max(minibatch_losses[num_losses:]) * 1.02
        ])

    # Plot running average of the loss
    running_avg = np.convolve(minibatch_losses,
                              np.ones(averaging_iterations) / averaging_iterations,
                              mode='valid')
    ax.plot(running_avg,
            label=f'Running Average{custom_label}',
            color="purple")
    ax.legend()

    # Create a second x-axis for epochs.
    ax2 = ax.twiny()
    new_labels = list(range(num_epochs + 1))
    new_positions = [e * iter_per_epoch for e in new_labels]
    
    # Show only every 5th label 
    ax2.set_xticks(new_positions[::5])
    ax2.set_xticklabels(new_labels[::5])
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax.get_xlim())
    
    plt.tight_layout()
