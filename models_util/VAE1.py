# The Variatonal Autoencoder model class. 
# The architecture was based on pimms implementation of VAE for imputations by Rasmussen 
# et al. 
# More details can be found at: https://github.com/RasmussenLab/pimms
# The Cost function is based on the ELBO and I used bits per dim for KL-divergence
# reguralizaton. https://github.com/ronaldiscool/VAETutorial


# import libraries 
import numpy as np
import pandas as pd
import numpy.random as nrd
import os 
import math
# Pytorch modules 
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim


# this for the custom Dataset 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 
import seaborn as sns
import matplotlib.pyplot as plt

# Import tqdm for progress bar
from tqdm.auto import tqdm

# for timing functions
from timeit import default_timer as timer 

# import the configuration file 
from models_util import configs 


# set the device and seed. Eetrieve seed and run the set_seed() for reproducibility 
device = configs.get_device()
seed = configs.get_seed()
configs.set_seed(seed)

# model parameters set to device agnostic
PI = torch.tensor(torch.pi, device=device)
log_of_2 = torch.log(torch.tensor(2., device=device))


# Model class
class VAE(nn.Module):
    def __init__(self, 
                 n_features: int,
                 latent_dim: int,
                 hidden_layer : bool = False,
                 hidden_dim: int = None,
                 output_activ = None):
        """
        Parameters:
        ------------------
        n_features : int
            Number of input features (columns of the protein table e.g. 150).
        latent_dim : int
            Size of latent space (e.g., 20).
        hidden_layer : bool
            A boolean value indicating whether a hidden layer is added or not.
            By default is False
        hidden_dim : int
            Number of neurons in the hidden layer (e.g., 50), if there is a hidden
            layer. Default value is None 
        output_activ : nn.Module
            If None, does not apply an activation function to the decoder output. Usefull
            when data is scaled to (0,1) or (-1,1). Not recommended for raw measurements. 
        
        Information
        ------------------
        The VAE has maximum one hidden layer with LeakyReLu activation function
        to the encoder. The decoder is either linear or an activation function
        is applied if scaled data is used. 
        For Regularlization I used dropout rate equal to 0.2, got good results.
        I added the choice of a model without hidden layer and with a linear 
        transformation if the data is unscaled and raw values might be used.
        The x_mu (averages) output of the decoder can be transformed by activation functions or
        not. The log_var (variances) will not be transformed and are extracted from
        a separate head of the decoder's architecture.

        """ 

        super().__init__()

        # Load the parameters 
        self.n_features = n_features
        self.hidden_layer = hidden_layer
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_activ = output_activ

        # the encoder module
        if hidden_layer == False: # this is linear transformation 
            self.encoder = nn.Sequential(
            # one linear Layer the latent Space: z_mu and z_logvar
            nn.Linear(self.n_features, self.latent_dim * 2)
        )
        else:
            self.encoder = nn.Sequential(
            # one hidden layer
            nn.Linear(in_features=self.n_features, out_features=self.hidden_dim),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(.1),

            # Latent Space: z_mu and z_logvar 
            nn.Linear(self.hidden_dim, self.latent_dim * 2)
            )
       
        
        # decoder module
        # No hidden Layers - separate the two heads
        if hidden_layer == False:
            decoder_list =[
            # one linear Layer the latent Space: z_mu and z_logvar
            nn.Linear(latent_dim, n_features)]
                      
            if self.output_activ:
                decoder_list.append(self.output_activ)
            
            self.head_mu = nn.Sequential(*decoder_list)
            self.head_logvar = nn.Linear(latent_dim, n_features)


        # Decoder with one hidden layer 
        else:
            # Common corpus of the decoder
            self.decoder_common = nn.Sequential(
            # From latent to hidden 
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(.1))

            # Decoder output separated in two heads: x_mu and x_logvar 
            head_mu_list = [nn.Linear(self.hidden_dim, self.n_features)]
            if self.output_activ:
                head_mu_list.append(self.output_activ)
            
            # unpack the n x_mu variables
            self.head_mu = nn.Sequential(*head_mu_list)
            
            # unpack the n x_logvar variables 
            self.head_logvar = nn.Linear(self.hidden_dim, self.n_features)


    def encode(self, x: torch.tensor):
        z_variables = self.encoder(x)
        # unpack mean and logvar of variables (latent_dim * 2) 
        z_mu = z_variables[:, :self.latent_dim]
        z_logvar = z_variables[:, self.latent_dim:]
        return z_mu, z_logvar
    

    def init_weights(self):
        """Apply Kaiming init to layers of encoder and decoder.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # condiotional for linear layers of the module generator 
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_latent_variables(self, x, detach = False):
        """
        If detach = True the latent variables are not part of the 
        computation graph. Better for downstream analysis.
        """
        
        z_mu, z_logvar = self.encode(x)
        if detach:
            z_mu = z_mu.detach()
            z_logvar = z_logvar.detach()
        return z_mu, z_logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std #samples from the latent space 
    

    def decode(self, z):
        if self.hidden_layer:
            common_layer = self.decoder_common(z)
            x_mu = self.head_mu(common_layer)
            x_logvar = self.head_logvar(common_layer)
            return x_mu, x_logvar
        else:
            x_mu = self.head_mu(z)
            x_logvar = self.head_logvar(z) 
            return x_mu, x_logvar


    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.decode(z)
        return x_mu, x_logvar, z_mu, z_logvar


if __name__ == "__main__":
    print("Run the script locally")
else:
    print(f"Importing {__name__}, running in {device} with seed: {seed}" )