# Project Structure 
```
proteomics_in_latent_space/                 # 🔹 Project Root (Main Working Directory)
│
├── running_script.py                       # 🔹 Main script to train the models 
│
├── models_util/                            # 🔹 Package for all python modules 
│   ├── __init__.py                        
│   ├── configs.py                           # ⚙️  Handles seeds  & device configuration
│   ├── cost_functions.py                   # 📊  VAE cost functions 
│   ├── custom_dataset.py                   # 🗂️  Handles dataset loading in pytorch
│   ├── utility_functions.py                # ⚙️  Useful functions VAE training  
│   ├── VAE1.py                             #      Variational autoencoder 
│
├── r_util/                                 #  R scripts & utilities for analysis 
│   ├──                                     # 
│   ├──                                     #
│
├── data/                                   # 🔹 Folder to store datasets
│   ├── processed/
│       ├── prot_abms_norm.txt              # 📜 Total Cell proteomics
│       ├── protein_quant_merged.txt        # 📜 Subcellular proteomics   
│   ├── raw/
│       ├──                        
│
├── outputs/                                # 📁 Stores trained models & logs
│   ├── trained_model.pth                   # 🎯 Saved PyTorch model checkpoint
│   ├── training_logs.txt                    # 📄 Training logs & results
│
├── requirements.txt                        # Dependencies

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