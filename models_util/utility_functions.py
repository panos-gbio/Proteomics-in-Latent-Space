# This script contains utility functions that I often used in this project:

# - spliting the training, validation and test set.
# - The Cost functions of the VAE model (KL divergence and Gaussian Likelihood Error)
# - graphs of the training and validation loss 

# import libraries 
import numpy as np
import pandas as pd
import numpy.random as nrd

# Pytorch modules 
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

# scikit-learn for create_data_partition function 
from sklearn.model_selection import train_test_split 

# import the configuration file
from models_util import configs


# set the device and seed. Eetrieve seed and run the set_seed() for reproducibility 
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
