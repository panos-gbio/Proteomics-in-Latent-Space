# Configuration file for reproducibility and device assignment for the project
# The purpose of a global configuration file is to set the parameter variables
# for the necessary modules of the project

# When you set the config variables, the modules will refer to the same config
# file and share the same variables. 


# libraries used
import torch
import random
import numpy as np
import os
import sys

# global variables are stored in the module's named space
project_seed = None
project_device = None

def set_device(force_cpu = False):

    """
    Returns the device variable for all the modules of the project.
    Set the force_cpu to True if you want to run everything in CPU. 
    """
    # I convert the local variable to a global variable from the
    # namespace and the function updates the global variable
    global project_device
    
    if force_cpu:
        project_device = torch.device("cpu")
        return project_device
    
    project_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return project_device


def set_seed(seed=123):

    """
    Set the seed for reproducibility of the project.
    For all the libraries.
    """
    # same as above
    global project_seed
    
    project_seed = seed
    
    random.seed(seed)              # Python random module
    np.random.seed(seed)           # NumPy random
    torch.manual_seed(seed)        # PyTorch CPU

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # GPUs, if available 
    print(f"During configuration random seed {seed} has been set.")
# maybe add seed for the Dataloader? 

def get_device():

    """
    Returns the device variable  set when it is called
    """
    if project_device is None:
        raise RuntimeError("Device is not detected. Import configs.py and set the device.")
    return project_device

def get_seed():

    """
    Returns the seed variable when it is called 
    """
    if project_seed is None:
        raise RuntimeError("Seed is not detected. Import configs.py and set the seed.")
    return project_seed

# for reproducibility checks 
def get_configs():

    """
    Returns the configuration variables of the project. 
    """
    return f"Seed: {project_seed}, Device: {project_device}"



if __name__ == "__main__":
    print("Run the script locally.")
else:
    print(f"Importing {__name__} module")
    print("First set device and seed for reproducibility.")
    print("-----------------------------------------------")