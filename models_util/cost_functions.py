# This script contains The Cost functions of the VAE model:
#  KL divergence for the difference of the distribution of the latent variables with Gaussian p(x) and q(z|x)
#  Gaussian Log-Likelihood Error for the reconstruction p(x_mu|z) 

# If you want to test the file make sure you run it from the
# project root \your\path\to\proteomics_latent_space and not directly from
# models_util_folder 


# Libraries used 
import os
import sys 

# import configration file 
from models_util import configs

# Pytorch modules 
import torch
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
import torch.nn.functional as F
from torch import nn
import torch.optim as optim


# set the device and seed. Eetrieve seed and run the set_seed() for reproducibility 
device = configs.get_device()
seed = configs.get_seed()
configs.set_seed(seed)


# model parameters set to device agnostic
PI = torch.tensor(torch.pi, device=device)
log_of_2 = torch.log(torch.tensor(2., device=device))


def kld_loss(z_mu, z_logvar):
    # classical interpretation of KL divergence 
    return 0.5 * (z_mu**2 + torch.exp(z_logvar) - 1 - z_logvar)
    
# def get_kl_perdim():



def gaussian_loss(x_batch, x_mu, x_logvar, mask):
        
     """
    Computes Gaussian log probability loss, considering only non-missing values.
        
    Parameters:
    -----------
    x_batch : Tensor
        Original input data.
    x_mu : Tensor
        Reconstructed mean output from decoder.
    x_logvar : Tensor
        Log variance from decoder.
    mask : Tensor (Boolean)
        Mask matrix indicating missing values (True = missing, False = observed).

    Returns:
    --------
    Mean Gaussian per batch loss over non-missing values.
    """
     if mask == None:
        # compute the log-likelihood tensor (for each point in original X dataset)
        log_prob = -0.5 * (torch.log(2. * PI) + x_logvar + (x_batch - x_mu)**2 / torch.exp(x_logvar))
     else:
        # compute the log-likelihood tensor (for each point in original X dataset)
        log_prob = -0.5 * (torch.log(2. * PI) + x_logvar + (x_batch - x_mu)**2 / torch.exp(x_logvar))
        
        # Remove entries corresponding to NA positions  
        log_prob = log_prob[~mask]
        
     return -log_prob.mean() #mean per batch (basically the mean value if the whole tensor)
    


def log_contbernoulli(x_batch, x_mu_logits, mask=None, eps =1e-6):
    """
    Computes Continuous Bernoulli log probability loss, using a mask for NAs.

    Parameters:
    -----------
    x_batch : Tensor
        Original input data (in [0,1]).
    x_mu_logits : Tensor
        The Decoder output without sigmoid activation - more stable for training. 
    mask : Tensor (Boolean)
        Mask matrix indicating missing values (True = missing, False = observed).
    eps : float
        Small epsilon to clamp x_batch and avoid log(0) issues.

    Returns:
    --------
    Average Continuous Bernoulli loss over non-missing values.
    """

    # Clamp original x_batch to avoid log(0) issues
    x_batch = x_batch.clamp(min=eps, max=1 - eps)

    # Define the Continuous Bernoulli distribution
    dist = ContinuousBernoulli(probs=x_mu_logits)
    log_prob = dist.log_prob(x_batch)

    # Apply mask (exclude missing values)
    if mask is not None:
        log_prob = log_prob[~mask]

    # Return mean negative log-likelihood over non-missing values
    return -log_prob.mean()



def loss_fun_gauss(x_batch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=None,freebits=0.1):
    
    # if mask.shape != x_batch.shape:
    #     raise TypeError("The dimensions of batch and mask matrices do not match")
    
    l_rec = gaussian_loss(x_batch, x_mu, x_logvar, mask)
    l_reg = torch.sum((F.relu(kld_loss(z_mu, z_logvar) # it sums all the latent-dimension/row-sample
                              - freebits * log_of_2)    # returns a scalar per row 
                       + freebits * log_of_2),
                      1)
    l_reg = torch.mean(l_reg) #mean KL per batch  (mean of all row samples)

    # store the losses as numbers and not tensors for loss curves 
    lst.append(l_rec.detach().item())
    lst.append(l_reg.detach().item())
    
    # this returns them as tensors - necessary for backprop. 
    return l_rec + l_reg



def loss_fun_contbern(x_batch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=None,freebits=0.1):
    
    # if mask.shape != x_batch.shape:
    #     raise TypeError("The dimensions of batch and mask matrices do not match")
    

    l_rec = log_contbernoulli(x_batch, x_mu, mask, eps=1e-6)
    l_reg = torch.sum((F.relu(kld_loss(z_mu, z_logvar) # it sums all the latent-dimension/row-sample
                                - freebits * log_of_2)    # returns a scalar per row 
                        + freebits * log_of_2),
                        1)
    l_reg = torch.mean(l_reg) #mean KL per batch  (mean of all row samples)

    # store the losses as numbers and not tensors for loss curves 
    lst.append(l_rec.detach().item())
    lst.append(l_reg.detach().item())
        
     # this returns them as tensors - necessary for backprop. 
    return l_rec + l_reg


def loss_fun_bce(x_batch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=None,freebits=0.1):
    
    # if mask.shape != x_batch.shape:
    #     raise TypeError("The dimensions of batch and mask matrices do not match")
    
    bce = F.binary_cross_entropy_with_logits(x_mu, x_batch, reduction='none')
    bce = bce[~mask]
    l_rec = bce.mean()


    l_reg = torch.sum((F.relu(kld_loss(z_mu, z_logvar) # it sums all the latent-dimension/row-sample
                                - freebits * log_of_2)    # returns a scalar per row 
                        + freebits * log_of_2),
                        1)
    l_reg = torch.mean(l_reg) #mean KL per batch  (mean of all row samples)

    # store the losses as numbers and not tensors for loss curves 
    lst.append(l_rec.detach().item())
    lst.append(l_reg.detach().item())
        
     # this returns them as tensors - necessary for backprop. 
    return l_rec + l_reg


def loss_fun_mse(x_batch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=None,freebits=0.1):
    
    # if mask.shape != x_batch.shape:
    #     raise TypeError("The dimensions of batch and mask matrices do not match")
    
    mse = F.mse_loss(x_mu, x_batch, reduction='none')
    mse = mse[~mask]
    l_rec = mse.mean()


    l_reg = torch.sum((F.relu(kld_loss(z_mu, z_logvar) # it sums all the latent-dimension/row-sample
                                - freebits * log_of_2)    # returns a scalar per row 
                        + freebits * log_of_2),
                        1)
    l_reg = torch.mean(l_reg) #mean KL per batch  (mean of all row samples)

    # store the losses as numbers and not tensors for loss curves 
    lst.append(l_rec.detach().item())
    lst.append(l_reg.detach().item())
        
     # this returns them as tensors - necessary for backprop. 
    return l_rec + l_reg

if __name__ == "__main__":
    print(f"Run the script locally,using seed {seed} and device: {device}")
    print(os.getcwd())
else:
    print(f"Importing {__name__}, running in {device} with seed: {seed}" )