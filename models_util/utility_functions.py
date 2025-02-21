# This script contains utility functions that I often used in this project:

# - spliting the training, validation and test set.
# - the training loop
# - graphs of the training and validation loss 
# etc, etc 

# import libraries 
import numpy as np
import pandas as pd
import numpy.random as nrd
import matplotlib.pyplot as plt
import seaborn as sns 
from typing import Callable

# Import tqdm for progress bar
from tqdm.auto import tqdm

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



# Train - Val loop for the models
def train_val_loop(model: nn.Module,
                   loss_fun: Callable,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   model_name = "",
                   model_path = "",
                   epoch=20,
                   learn_r=0.005,
                   freebits=0.1,
                   batch_size=128,
                   norm = 0):



    # set optimizer and learning rate
    optimizer = optim.Adam(model.parameters(), lr=learn_r)

    # create a string for the hyperparameters 
    hyperparam_str = f"ep{epoch}_norm{norm}_bits{freebits}_bs{batch_size}_lr{optimizer.param_groups[0]["lr"]}"

    # check for model name
    if model_name == "":
        raise RuntimeError("Insert a model name")
    # add model name to the hyperparam_str
    final_name = model_name + "_" + hyperparam_str

    # where to save the model
    if model_path == "":
        raise RuntimeError("Write a path to save the model")
    model_path = model_path + f"\\{final_name}.pth"
    
    # Storage
    # for each batch/iteration
    batch_dict = {
        "iteration": [],
        "Train total Loss": [],
        "Train KL Loss": [], 
        "Train Rec Loss": []
        }

    # for each epoch
    epoch_dict = {
        "epoch": [],
        "Train total Loss": [],
        "Train KL Loss": [], 
        "Train Rec Loss": [],
        "Val total Loss": [],
        "Val KL Loss": [],
        "Val Rec Loss": []
        }


    for epoch in tqdm(range(epoch+1)):
        
        
        # initialize the loss metrics at epoch zero
        if epoch == 0:
            print(f"Performing pre-training evaluation on the model in epoch {epoch}")
            val_loss, val_kl, val_rl = 0,0,0
            model.eval()
            with torch.inference_mode(): # it doesnt update parameters 
                lst = []
                for val_batch, t_mask, tidx in val_loader:
                    x_mu, x_logvar, z_mu, z_logvar = model(val_batch)
                    loss = loss_fun(val_batch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=t_mask,freebits=freebits)
                    val_loss += loss.detach().item()
                    val_kl += lst[-1]
                    val_rl += lst[-2]
                
                val_loss = val_loss/len(val_loader)
                val_kl = val_kl/len(val_loader)
                val_rl = val_rl/len(val_loader)
                
                epoch_dict["epoch"].append(epoch)
                epoch_dict["Train total Loss"].append(val_loss)
                epoch_dict["Train KL Loss"].append(val_kl)
                epoch_dict["Train Rec Loss"].append(val_rl)
                epoch_dict["Val total Loss"].append(val_loss)
                epoch_dict["Val KL Loss"].append(val_kl)
                epoch_dict["Val Rec Loss"].append(val_rl)
            
            print(f"\nVal loss: {val_loss:.3f}| Val KL: {val_kl} | Val Rec: {val_rl:.3f}\n")
        
        # begin training the model from iteration 0 and after epoch 0 
        else:
            print(f"Epoch {epoch}\n--------------------")
            train_loss, train_kl, train_rl = 0,0,0
            lst = [] # this list stores the averaged losses/batch that are computed from the loss
            _iter = 0
            # print("Iter initialized before loop")			
            for batch, (xbatch, xmask, xidx) in enumerate(train_loader):
                model.train()
                # device
                xbatch, xmask = xbatch.to(device), xmask.to(device)

                #
                optimizer.zero_grad()

                x_mu, x_logvar, z_mu, z_logvar = model(xbatch)

                loss = loss_fun(xbatch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=xmask,freebits=freebits)
                train_loss += loss.detach().item()
                train_kl += lst[-1]
                train_rl += lst[-2]

                batch_loss = loss.detach().item()
                batch_kl = lst[-1]
                batch_rl = lst[-2]

                loss.backward()
                        
                # Optional gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=norm)
                optimizer.step()

                # update the batch dictionary 
                batch_dict["iteration"].append(_iter)
                batch_dict["Train total Loss"].append(batch_loss)
                batch_dict["Train KL Loss"].append(batch_kl)
                batch_dict["Train Rec Loss"].append(batch_rl)

                _iter +=1 

                # print every round of 10 batches the losses - smooths the results 
                # if batch % 10 == 0:
                #     print(f"Iter {batch} and a total {batch*batch_size}/{len(train_loader.dataset)} proteins have passed.")
                #     print(f"Current Loss: {train_loss/(batch+1)} | KL Loss: {train_kl/(batch+1)}| Rec Loss: {train_rl/(batch+1)}")


            # calculate per epoch the metrics - divide by number of batches 
            train_loss = train_loss/len(train_loader)
            train_kl = train_kl/len(train_loader)
            train_rl = train_rl/len(train_loader)
            
            # add them to the dictionary 
            epoch_dict["epoch"].append(epoch)
            epoch_dict["Train total Loss"].append(train_loss)
            epoch_dict["Train KL Loss"].append(train_kl)
            epoch_dict["Train Rec Loss"].append(train_rl)
            

            # pass the validation set to the VAE 
            val_loss, val_kl, val_rl = 0,0,0
            model.eval()
            with torch.inference_mode(): # it doesnt update parameters based on gradients 
                lst = []
                for val_batch, t_mask, tidx in val_loader:

                    x_mu, x_logvar, z_mu, z_logvar = model(val_batch)
                    loss = loss_fun(val_batch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=t_mask,freebits=freebits)
                    val_loss += loss.detach().item()
                    val_kl += lst[-1]
                    val_rl += lst[-2]
                
                # divide by all the batches of val set to get epoch metrics 
                val_loss = val_loss/len(val_loader)
                val_kl = val_kl/len(val_loader)
                val_rl = val_rl/len(val_loader)

                epoch_dict["Val total Loss"].append(val_loss)
                epoch_dict["Val KL Loss"].append(val_kl)
                epoch_dict["Val Rec Loss"].append(val_rl)

            ## Print out what's happening
            print(f"Train loss: {train_loss:.3f}|Train Rec: {train_rl:.3f} | Val loss: {val_loss:.3f}, Val Rec: {val_rl:.3f}\n")

    # save the model after training to the designated path
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

    # two dictionaries and the hyperparamstring grouped in a tuple     
    return batch_dict,epoch_dict,hyperparam_str



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
            np.min(minibatch_losses[num_losses:])*0.99,
            np.max(minibatch_losses[num_losses:])*1.01
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



# create a df from the epoch dictionary
def convert_epoch_df(epoch_dict: dict):
    """
    Converts the per epoch metrics dictionary of the training loop
    to a dataframe.
    Makes training - validation set comparisons by calculating the absolute
    difference of all the metrics per epoch for the two sets
    
    """
    epoch_df = pd.DataFrame(epoch_dict)
    epoch_df = (epoch_df
                .assign(
                    rec_dif = np.abs(epoch_df["Val Rec Loss"] - epoch_df["Train Rec Loss"]),
                    kl_dif = np.abs(epoch_df["Val KL Loss"] - epoch_df["Train KL Loss"]),
                    total_dif = np.abs(epoch_df["Val total Loss"] - epoch_df["Train total Loss"])

                ))
    print(epoch_df.head(2))
    return epoch_df



# based on the previous function: Gets the metrics of train and validation set
def get_train_vs_val_metrics(epoch_df: pd.DataFrame,
                              epoch: int,
                              model_id="",
                              savepath=""):
    """
    epoch_df:   Dataframe generated from the convert_epoch_df function
                contains the per epoch metrics of the training loop
    epoch:      number of epochs
    model_id:   the id of the model to put in the title
    savepath:   the path to save the model - preferably its own directory
    
    """
    
    if model_id == "":
        raise RuntimeError("Insert a model name.")
    
    if savepath == "":
        raise RuntimeError("Insert the model's path to save figures")
    
    if isinstance(epoch_df, pd.DataFrame):
        print("df inserted")
    else:
        raise TypeError("The input should be a dataframe")


    # plot validation error vs training error per epoch
    custom_label1 = "Reconstruction Error"
    custom_label2 = "KL Error"

    # create subfigures
    fig = plt.figure(layout="constrained", figsize=(14,7))
    subfigs = fig.subfigures(1,2,wspace=0.07)

    # first subfigure of the ax plot 
    ax = subfigs[0].subplots(2,1, sharex=True)

    ax1 = sns.lineplot(
        epoch_df,x=epoch_df["epoch"], y=epoch_df["Train Rec Loss"],
        lw = 2, color = "green", alpha = 0.6, label="Training Error",
        marker = "o",ax=ax[0]
    )
    ax1 = sns.lineplot(
        epoch_df,x=epoch_df["epoch"], y=epoch_df["Val Rec Loss"],
        lw = 2, color = "purple", alpha = 0.7, label="Validation Error",
        marker = "o",ax=ax[0]
    )

    # legend and tickmarks
    ax1.legend(frameon=False)
    ax1.set_title(f"{custom_label1}", fontsize = 14)
    new_labels = list(range(epoch + 1))
    new_positions = list(range(epoch + 1))
    ax1.set_xticks(new_positions[::4])
    ax1.set_xticklabels(new_labels[::4])

    # add second line - share x axis 
    # ax2 = ax1.twinx()
    ax2 = sns.lineplot(
        epoch_df, x=epoch_df["epoch"], y=epoch_df["rec_dif"],
        color = "black", ls="--", marker = "o", label = "Absol. Difference of Set Errors",
        ax=ax[1]
    )
    ax2.legend(frameon=False)


    # second subfigure of the ax plot 
    ax = subfigs[1].subplots(2,1, sharex=True)

    ax1 = sns.lineplot(
        epoch_df,x=epoch_df["epoch"], y=epoch_df["Train KL Loss"],
        lw = 2, color = "green", alpha = 0.6, label="Training Error",
        marker = "o",ax=ax[0]
    )
    ax1 = sns.lineplot(
        epoch_df,x=epoch_df["epoch"], y=epoch_df["Val KL Loss"],
        lw = 2, color = "purple", alpha = 0.7, label="Validation Error",
        marker = "o",ax=ax[0]
    )
    ax1.set_title(f"{custom_label2}", fontsize = 14)
    ax1.legend(frameon=False)
    ax1.set_xticks(new_positions[::4])
    ax1.set_xticklabels(new_labels[::4])

    # add second line - share x axis 
    # ax2 = ax1.twinx()
    ax2 = sns.lineplot(
        epoch_df, x=epoch_df["epoch"], y=epoch_df["kl_dif"],
        color = "black", ls="--", marker = "o", label = "Absol. Difference of Set Errors",
        ax=ax[1]
    )
    ax2.legend(frameon=False)

    fig.suptitle(f"Training vs Validation Error Curve:\n {model_id}",
                y=1.1,x=.54,
                fontsize=16)


    # save figures 
    plt.savefig(savepath+"train_val_epoch_curve.png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("Run the script locally")
else:
    print(f"Importing {__name__}, running in {device} with seed: {seed}" )
