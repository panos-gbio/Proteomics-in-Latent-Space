# This is a custom datset class. The purpose of creating a custom dataset was to pass to the Dataloaer the protein dataset
# as well as a mask indicating the positions of the missing values. 
# This mask will be used during the calculation of Gaussial likelihood error of the VAE model. 


# import the libraries 
import numpy as np
import pandas as pd
import os 
import math

# Pytorch modules 
import torch

# this for the custom Dataset 
from torch.utils.data import Dataset

# import the configuration file
from models_util import configs 


# set the device and seed. Eetrieve seed and run the set_seed() for reproducibility 
device = configs.get_device()
seed = configs.get_seed()
configs.set_seed(seed)


# class
class ProteinDataset(Dataset):
    """
    It passes the whole dataset matrix to memory.
    Then, a mask matrix indicating NAs is created.
    Finally all NAs are replaced with zeroes
    Returns to the DataLoader the Dataset, Mask and
    index of the examples

    """
    nan = torch.tensor(float("NaN"))
    DEFAULT_DTYPE = torch.get_default_dtype()

    def __init__(self, data: np.array, row_ids = None, fill_na: float = 0.0):
        """
        Parameters
        ---------------------------
        data : np.array object 
            The protein dataset with the MS1 signals scaled from (0,1) with/o
            missing values.
        row_ids : np.array object
            It contains the protein ID symbols, character vector from the original
            protein table
        fill_na : int, optional
            the replacement of NAs, selected 0
        """
        if not isinstance(data, (np.ndarray)):
            raise TypeError(f"Dataset should be Numpy array object of 32 bits")
        else:
            print(f"Protein Dataset is passed to memory")
        
        # load the matrix
        self.proteins = torch.FloatTensor(data)
        
        # create boolean mask
        self.mask = torch.tensor(np.isnan(data))

        # replace values if there NAs
        self.proteins = torch.where(self.proteins.isnan(), torch.FloatTensor([fill_na]),
            self.proteins)
        
        self.length_ = len(self.proteins)

        # create a target tensor in case i need it 
        self.y = torch.where(~self.mask, self.nan, self.proteins)

        # store original indices and row names from the expression matrix 
        self.indices = np.arange(self.length_)
        self.row_ids = row_ids

        # instantiate original protein symbols
        if row_ids is None:
            print(f"No Protein Symbols were identified")
        else:
            print(f"Matrix with original Protein Symbols is identified")
            self.columns = row_ids
    
    def __len__(self):
        return self.length_
    
    def get_row_names(self, idx):
        return self.row_ids[idx]

    def __getitem__(self, idx): # __getitem__ method returns ONLY tensors 
        if self.row_ids is None:
            return (self.proteins[idx], self.mask[idx],
                torch.tensor(self.indices[idx],dtype=torch.int32))
        else:
            return (self.proteins[idx], self.mask[idx],
                torch.tensor(self.indices[idx],dtype=torch.int32))


if __name__ == "__main__":
    print(f"Run the script locally,using seed {seed} and device: {device}")
    print(os.getcwd())
else:
    print(f"Importing {__name__}, running in {device} with seed: {seed}" )