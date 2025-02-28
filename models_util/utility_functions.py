# This script contains utility functions that I often used in this project:

# - spliting the training, validation and test set.
# - the training loop
# - graphs of the training and validation loss 
# etc, etc 

# import libraries vanila python
import numpy as np
import pandas as pd
import numpy.random as nrd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import plotly.graph_objects as go
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
    
    # Show only every 10th label 
    ax2.set_xticks(new_positions[::10])
    ax2.set_xticklabels(new_labels[::10])
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
    ax1.set_xticks(new_positions[::10])
    ax1.set_xticklabels(new_labels[::10])

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
    ax1.set_xticks(new_positions[::10])
    ax1.set_xticklabels(new_labels[::10])

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
    plt.savefig(savepath+"\\train_val_epoch_curve.png", dpi=600, bbox_inches="tight")
    plt.show()


def test_set_analysis(
        model,
        test_loader,
        loss_fun,
        freebits,
        model_id
        ):

        # use the trained model and its parameters 
        print(f"Using this model {model_id}")

        # the original and reconstructed tensors will be usefull for observations 
        test_iter_dict = {
        "iteration": [],
        "Test total Loss": [],
        "Test KL Loss": [], 
        "Test Rec Loss": [],
        "Test batch index": [],
        "x_orig tensors": [],
        "x_mu tensors": [],
        "masks": []
        }
        test_metrics = {
                "model_id": f"{model_id}",
                "bits": freebits,
                "avg_total_loss":0,
                "avg_kl_loss": 0,
                "avg_rl_loss": 0,
        }
        lst = []
        model.eval()
        test_loss, test_kl, test_rl = 0,0,0
        _iter = 1 
        with torch.inference_mode():
                for tbatch, tmask, tidx in test_loader:

                        x_mu, x_logvar, z_mu, z_logvar = model(tbatch) # forward step 
                        loss = loss_fun(tbatch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=tmask,freebits=freebits) #loss calculation 
                        
                        # sum the losses batch by batch 
                        test_loss += loss.detach().item()
                        test_kl += lst[-1]
                        test_rl += lst[-2]

                        # metrics per Batch-Iteration (append loss fn outputs as values)
                        test_iter_dict["iteration"].append(_iter)
                        test_iter_dict["Test total Loss"].append(loss.detach().item())
                        test_iter_dict["Test KL Loss"].append(lst[-1])
                        test_iter_dict["Test Rec Loss"].append(lst[-2])
                        # list of tensors with sample indices and the samples original, reconstructed 
                        test_iter_dict["Test batch index"].append(tidx.detach()) 
                        test_iter_dict["x_orig tensors"].append(tbatch.detach())
                        test_iter_dict["x_mu tensors"].append(x_mu.detach())
                        test_iter_dict["masks"].append(tmask.detach())
                        
                        # update batch
                        _iter += 1
                
        # divide by all the batches to get average loss for whole test set  
        test_loss = test_loss/len(test_loader)
        test_kl = test_kl/len(test_loader)
        test_rl = test_rl/len(test_loader)

        # metrics of the whole test set
        test_metrics["avg_total_loss"] = test_loss
        test_metrics["avg_kl_loss"] = test_kl
        test_metrics["avg_rl_loss"] = test_rl


        # return the results of the analysis 
        return test_iter_dict, test_metrics



def train_val_loop_v2(model: nn.Module,
                   loss_fun: Callable,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   model_name = "",
                   model_path = "",
                   epoch=20,
                   patience = 5,
                   learn_r=0.005,
                   freebits=0.1,
                   batch_size=128,
                   norm = 0):


    # initilize epoch run / checkpoint and patience
    checkpoint = 0
    pa_tience = patience
    best_loss = 2
    lrswitch = False  

    # set optimizer and learning rate
    optimizer = optim.Adam(model.parameters(), lr=learn_r)

    # create a string for the hyperparameters 
    hyperparam_str = f"ep{checkpoint}_norm{norm}_bits{freebits}_bs{batch_size}_lr{optimizer.param_groups[0]["lr"]}"

    # check for model name
    if model_name == "":
        raise RuntimeError("Insert a model name")
    
    # where to save the model
    if model_path == "":
        raise RuntimeError("Write a path directory to save the model")
    
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
        
        # begin training the model from epoch 1 and iteration 0 
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
            
            # check if the val loss is smaller than the best loss and check for early stopping 
            if (epoch > 40) & (val_rl < best_loss):
                best_loss = val_rl
                checkpoint = epoch
                hyperparam_str = f"ep{checkpoint}_norm{norm}_bits{freebits}_bs{batch_size}_lr{optimizer.param_groups[0]["lr"]}"
                final_name = model_name + "_" + hyperparam_str
                final_path = model_path + f"\\{final_name}.pth"
                torch.save(model.state_dict(), final_path) # save the model on the checkpoint epoch 
                # if exceed patient reduce learning rate
            elif (lrswitch is False) & (epoch > 40) & (epoch - checkpoint > pa_tience):
                print(f"Patience exceeded at {epoch} with last checkpoint saved at {checkpoint}")
                optimizer.param_groups[0]["lr"] = 0.001
                print(f"changed learning rate to {optimizer.param_groups[0]["lr"]}")
                checkpoint = epoch
                pa_tience = 10
                lrswitch = True
                # if reduced learning rate exceeds patience break the loop 
            elif (lrswitch is True) & (epoch > 40) & (epoch - checkpoint > pa_tience):
                print(f"Early stopping at epoch {epoch} with last checkpoint saved at {checkpoint}")
                break
    
    # if we run at low epochs then probably patience is not going to be exceeded
    if lrswitch is False:
        hyperparam_str = f"ep{epoch}_norm{norm}_bits{freebits}_bs{batch_size}_lr{optimizer.param_groups[0]["lr"]}"
    
    # save the model after training to the designated path
    # torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

    # two dictionaries and the hyperparamstring grouped in a tuple     
    return batch_dict,epoch_dict,hyperparam_str


def plot_test_losscurves(
        test_iter_dict,
        test_metrics,
        model_id,
        model_path
):
        selected_keys = ["iteration","Test total Loss","Test KL Loss","Test Rec Loss"]

        print(f"Using this model {model_id}")

        # list comprehension of the dictionary keys/columns i want to keep
        cols = [key for key in test_iter_dict.keys() if key in selected_keys]

        # dictionary comprehension and dataframe convertion = get the columns and the whole dataframe 
        testdf = pd.DataFrame({key: test_iter_dict[key] for key in cols})
        testdf

        # plot the test metrics per iteration and average values
        fig, axes = plt.subplots(3,1,figsize=(8,8),sharex=True)


        # Plot RL loss
        ax1 = sns.lineplot(
        testdf,x=testdf["iteration"], y=testdf["Test Rec Loss"],
        lw = 2, color = "royalblue", alpha = 0.7, label="Rec Error",
        marker = "o",ax=axes[0], markerfacecolor="black")

        ax1.set_ylim([
        np.min(testdf["Test Rec Loss"])*1.03,
        np.max(testdf["Test Rec Loss"])*0.92
        ])

        ax1.axhline(y=test_metrics["avg_rl_loss"], color='black', linestyle='--', linewidth=1, label=f'Average: {round(test_metrics["avg_rl_loss"],2)}')
        ax1.legend(frameon = False, ncol=2)


        # plot KL loss
        ax2 = sns.lineplot(
        testdf,x=testdf["iteration"], y=testdf["Test KL Loss"],
        lw = 2, color = "royalblue", alpha = 0.7, label="KL Error",
        marker = "o",ax=axes[1],markerfacecolor="black")

        ax2.set_ylim([
        np.min(testdf["Test KL Loss"])*0.999,
        np.max(testdf["Test KL Loss"])*1.001
        ])


        ax2.axhline(y=test_metrics["avg_kl_loss"], color='black', linestyle='--', linewidth=1, label=f'Average: {round(test_metrics["avg_kl_loss"],2)}')
        ax2.legend(frameon = False, ncol=2)

        # plot total loss
        ax3 = sns.lineplot(
        testdf,x=testdf["iteration"], y=testdf["Test total Loss"],
        lw = 2, color = "royalblue", alpha = 0.7, label="Total Error",
        marker = "o",ax=axes[2],markerfacecolor="black")

        ax3.set_ylim([
        np.min(testdf["Test total Loss"])*0.99,
        np.max(testdf["Test total Loss"])*1.01
        ])


        ax3.axhline(y=test_metrics["avg_total_loss"], color='black', linestyle='--', linewidth=1, label=f'Average: {round(test_metrics["avg_total_loss"],2)}')
        ax3.legend(frameon = False, ncol=2)


        ax3.set_xlabel("Iterations")
        plt.suptitle(f"Test Set Metrics\n{model_id}")
        plt.savefig(model_path + "\\testloss_curve.png", dpi=600, bbox_inches="tight",transparent=True)
        plt.show()



def get_matrix_rec(
        test_iter_dict,
        model_id,
        model_path
):
    # Reconstructions of the test set 

        # get a random tesnor from the test_iter dictionary, where the Xreconstrcuted are stored
        i = np.random.randint(len(test_iter_dict["x_orig tensors"]))

        # the tensors are not in the computation graph but the still need to be transfered to the cpu and then converted to numpy 
        len(test_iter_dict["x_orig tensors"])
        xorig, xrec, xmask = test_iter_dict["x_orig tensors"][i].cpu().numpy(), test_iter_dict["x_mu tensors"][i].cpu().numpy(), test_iter_dict["masks"][i].cpu().numpy()
        xdif = xorig-xrec

        # create the plot 
        fig, axes = plt.subplots(1,3,figsize=(18,6),sharey=True)

        ax1 = sns.heatmap(xorig, cmap="viridis",vmin=0,vmax=1,
                        mask=xmask,xticklabels=False, ax=axes[0],
                        cbar = False)

        ax2 = sns.heatmap(xrec, cmap="viridis",vmin=0,vmax=1,
                        mask=xmask,xticklabels=False,ax=axes[1],
                        cbar_ax=fig.add_axes([.17, 0, .4, .04]),
                        cbar_kws={"orientation": "horizontal"})


        ax3 = sns.heatmap(np.abs(xdif), cmap="Reds",
                        mask=xmask,xticklabels=False, ax=axes[2])
        ax1.set_title("Original matrix", fontsize = 14, y=1.05)
        ax2.set_title("Reconstructed matrix", fontsize = 14, y=1.05)
        ax3.set_title("Absolute difference", fontsize = 14, y=1.05)

        fig.suptitle(f"Protein Matrix Reconstruction of Test Set Batch:\n {model_id}",
                y=1.1,x=.54,
                fontsize=16)
        plt.savefig(model_path + "\\matrix_reconstruction.png", dpi=600, bbox_inches="tight",transparent=True)

        # save figures
        plt.show()

## scatter 3d umap for figures

def plot_umap3d(umap_df,
                savepath: str=""):
    
    
    if savepath == "":
        raise RuntimeError("Provide a path to save the figure")

    
    fig = plt.figure(figsize=(18, 12))

    # one figure - one dictionary
    fig_params = [{"view": (20,20), "xlabel":"UMAP1", "ylabel":"UMAP2","zlabel":"UMAP3"},
                  {"view": (20,110), "xlabel":"UMAP1", "ylabel":"UMAP2","zlabel":"UMAP3"},
                  {"view": (20,290), "xlabel":"UMAP1", "ylabel":"UMAP2","zlabel":"UMAP3"}]
    # data in x,y,z axis
    x, y, z = umap_df["dim1"], umap_df["dim2"], umap_df["dim3"]

    #color per protein
    colors = umap_df["color2"]

    # iterate over the # figures
    for n, params_dict in enumerate(fig_params):
    # Add 3D subplots
        ax = fig.add_subplot(1,3,n+1, projection='3d')
        if n < 2:
            scatter = ax.scatter(x, y, z, c=colors, marker='o',alpha=0.4, s=15)
        else:
            scatter = ax.scatter(z, y, x, c=colors, marker='o',alpha=0.4, s=15)
            ax.set_title('UMAP View 1',fontsize=14)
            ax.set_xlabel(params_dict["zlabel"])
            ax.set_ylabel(params_dict["ylabel"])
            ax.set_zlabel(params_dict["xlabel"])

    
        ax.set_title(f'UMAP View {n+1}',fontsize=14)
        ax.set_xlabel(params_dict["xlabel"])
        ax.set_ylabel(params_dict["ylabel"])
        ax.set_zlabel(params_dict["zlabel"])
        ax.view_init(elev=params_dict["view"][0], azim=params_dict["view"][1])  # Adjust the view angle for better clarity

        # Set gridlines only for the XY plane (grounded)
        ax.grid(False)
        # ax1.xaxis.gr
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = True


    comp_col_dict = sorted(dict(zip(umap_df["Compartments"],umap_df["color2"])).items())

    categories = {
        'Cytosol Neighborhood': '#ADD8E6',  # lightblue
        'Mitochondria': '#8B7355',  # burlywood4
        'Nucleosol': '#7F7F7F',  # gray
        'Secretory': '#FFD700'  # gold
    }
    # set y axis of legend
    yleg = 0.2
    yanot = yleg - 0.03
    ytext = yanot - 0.02
    fig.patches.extend([plt.Rectangle((0.15,yanot),0.22,0.01,
                                    fill=True, color='#ADD8E6', alpha=1, zorder=1000,
                                    transform=fig.transFigure, figure=fig)])

    fig.text(.25,ytext, "Cytosol", fontsize=11, color="black")

    fig.patches.extend([plt.Rectangle((0.38,yanot),0.09,0.01,
                                    fill=True, color='#8B7355', alpha=1, zorder=1000,
                                    transform=fig.transFigure, figure=fig)])
    fig.text(.395,ytext, "Mitochondria", fontsize=11, color='black')

    fig.patches.extend([plt.Rectangle((0.48,yanot),0.185,0.01,
                                    fill=True, color='lightgray', alpha=1, zorder=1000,
                                    transform=fig.transFigure, figure=fig)])
    fig.text(.55,ytext, "Nucleosol", fontsize=11, color="black")


    fig.patches.extend([plt.Rectangle((0.67,yanot),0.18,0.01,
                                    fill=True, color='gold', alpha=1, zorder=1000,
                                    transform=fig.transFigure, figure=fig)])
    fig.text(.746,ytext, "Secretory", fontsize=11, color="black")

    # create the legends for the clusters using the colormap dictionary

    legend_handles = [mpatches.Patch(color=color, label=comp) for comp, color in comp_col_dict]
    fig.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.5, yleg), ncol=15, title="Clusters",
            frameon=False, fontsize=10)
    plt.suptitle("Localization-based 3D UMAP Plot of the Latent Variables",y=.8,fontsize=16)
    plt.tight_layout()
    
    plt.savefig(savepath + "\\umap3d_latent.png", dpi=600, bbox_inches="tight",transparent=True)

    
    plt.show()



## plotly umap
def plot_umap3d_plotly(umap_df, savepath: str = ""):
    if savepath == "":
        raise RuntimeError("Provide a path to save the figure")
    
# Create an empty figure
    fig = go.Figure()
    
    compartments = sorted(umap_df["Compartments"].unique())
    for comp in compartments:
        df_subset = umap_df[umap_df["Compartments"] == comp]
        # Use the color from the 'color2' column (assuming it is consistent per compartment)
        color = df_subset["color2"].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=df_subset["dim1"],
            y=df_subset["dim2"],
            z=df_subset["dim3"],
            mode='markers',
            marker=dict(size=4, color=color, opacity=0.4),
            name=comp
        ))
    
    # Update scene (3D axes) labels and overall layout
    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            zaxis_title="UMAP3",
            # You can set a custom camera view if desired:
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        title="Localization-based 3D UMAP Plot of the Latent Variables",
        width=1000,
        height=800,
        # Define the legend style (horizontal legend placed at a specific paper coordinate)
        legend=dict(
            title="Clusters",
            orientation="h",
            yanchor="middle",
            y=0.2,      # similar to your yleg value (0.2)
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # Add custom rectangle shapes (using paper coordinates) to mimic your manual patches.
    # Here the rectangles are drawn in the overall figure area.
    shapes = [
    
        dict(type="rect", xref="paper", yref="paper",
             x0=0.12, y0=0.17, x1=0.38, y1=0.17+0.01,
             fillcolor="#ADD8E6", line=dict(width=0)),
        
        dict(type="rect", xref="paper", yref="paper",
             x0=0.39, y0=0.17, x1=0.39+0.12, y1=0.17+0.01,
             fillcolor="#8B7355", line=dict(width=0)),
        
        dict(type="rect", xref="paper", yref="paper",
             x0=0.52, y0=0.17, x1=0.56+0.185, y1=0.17+0.01,
             fillcolor="lightgray", line=dict(width=0)),
        # Secretory: starting at (0.67, 0.17) with width 0.18
        dict(type="rect", xref="paper", yref="paper",
             x0=0.76, y0=0.17, x1=0.76+0.22, y1=0.17+0.01,
             fillcolor="gold", line=dict(width=0))
    ]
    fig.update_layout(shapes=shapes)
    
    annotations = [
        dict(x=0.24, y=0.15, xref="paper", yref="paper",
             text="Cytosol", showarrow=False, font=dict(size=11, color="black")),
        dict(x=0.45, y=0.15, xref="paper", yref="paper",
             text="Mitochondria", showarrow=False, font=dict(size=11, color="black")),
        dict(x=0.63, y=0.15, xref="paper", yref="paper",
             text="Nucleosol", showarrow=False, font=dict(size=11, color="black")),
        dict(x=0.9, y=0.15, xref="paper", yref="paper",
             text="Secretory", showarrow=False, font=dict(size=11, color="black"))
    ]
    fig.update_layout(annotations=annotations)
    
    # Save the interactive figure as an HTML file.
    fig.write_html(savepath)
    print(f"Figure saved as HTML to: {savepath}")

if __name__ == "__main__":
    print("Run the script locally")
else:
    print(f"Importing {__name__}, running in {device} with seed: {seed}" )
