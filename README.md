# Project Structure 
```
proteomics_in_latent_space/                 # Project Root (Main Working Directory)
│
├── 00_hyperparam_optimization.ipynb        # Script to optimze hyperparam 
├── 01_run_model_SCBC.ipynb                 # Script to run the subcell proteomics data 
├── 02_run_model_ABMS.ipynb                 # Script to run the total-cell ptoteomics data
├── 03_ppi_predictions.ipynb                # Script to the ML pipeline for ppi predictions 
│
├── models_util/                            # Package for all python modules 
│   ├── __init__.py                        
│   ├── configs.py                          #  Handles seeds  & device configuration
│   ├── cost_functions.py                   #  VAE cost functions 
│   ├── custom_dataset.py                   #  Handles dataset loading in pytorch
│   ├── utility_functions.py                #  Usefull functions for the DL part of the project 
|   ├── ml_utility_funct.py                 #  Useful functions for ML part of the project  
│   ├── VAE1.py                             #  Variational autoencoder model
│   ├── VAE2.py                             #  VAE with deeper architecture 
│
├── r_util/                                 #  R scripts for analysis & figures 
│   ├──                                     # 
│   ├──                                     #
│
├── data/                                   #  Folder to store datasets
│   ├── processed/
│       ├── prot_abms_norm.txt              # Total Cell proteomics
│       ├── protein_quant_merged.txt        # Subcellular proteomics   
│   ├── raw/
│       ├──                        
│
├── models/                                 # Stores trained models & related figures
│   ├── model_name/                         # Example
│       ├── model_name.pth                  # Model checkpoint saved as pth
│       ├── example_figure_1.jpg            # Figure from the running script 
│                
├── figures/                                # Final figures from specified models and analysis 
│
├── enviroment.yml                          # Dependencies for python enviroment
├── notes.md                                # 
├── .gitignore                              # 
```


## Table of Contents



### Making the Project Reproducible (seed and device check)
Since Jupyter **does not reset** the RNG state between cells, we need to explicitly call set_seed() inside each module in the `model_util` folder.<br>
1. First we set a device and a seed number using the **configs.py** functions `set_seed()` and `set_device()`. These variables are stored as global variables. <br>
2. Next, when we import each module, configs.py is imported too. The seed and device variables are assigned to each module with the `get_seed()` and `get_device()` functions of the configs.py. These functions return the variables we used in **step 1**. <br>
3. Then to finally reproduce randomness, in each module, it automatically runs `set_seed(get_seed())`, so the same seed is used for the random states of all the scripts. 
<br>
4. We tried to reproduce shuffling, weight initialization, and the Dataloader object of pytorch. 

### Runnign the VAE from Root Directory 