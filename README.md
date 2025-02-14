## Table of Contents


### Making the Project Reproducible (seed and device check)
Since Jupyter **does not reset** the RNG state between cells, we need to explicitly call set_seed() inside each module in the `model_util` folder.<br>
1. First we set a device and a seed number using the **configs.py** functions `set_seed()` and `set_device()`. These variables are stored as global variables. <br>
2. Next, when we import each module, configs.py is imported too. The seed and device variables are assigned to each module with the `get_seed()` and `get_device()` functions of the configs.py. These functions return the variables we used in **step 1**. <br>
3. Then to finally reproduce randomness, in each module, it automatically runs `set_seed(get_seed())`, so the same seed is used for the random states of all the scripts. 
<br>
4. We tried to reproduce shuffling, weight initialization, and the Dataloader object of pytorch. 

### Runnign the VAE from Root Directory 