# experimental code to save the tran-val loop

epoch = 20
model = model3
optimizer = optimizer
freebits = 1

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


for epoch in tqdm(range(epoch)):
	print(f"Epoch {epoch + 1}\n--------------------")
	train_loss, train_kl, train_rl = 0,0,0
	lst = [] # this list stores the averaged losses/batch that are computed from the loss
	iter = 0			
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
		loss.backward()
		optimizer.step()

		# update the batch dictionary - no val since #iterations are not the same 
		batch_dict["iteration"].append(iter)
		batch_dict["Train total Loss"].append(train_loss)
		batch_dict["Train KL Loss"].append(train_kl)
		batch_dict["Train Rec Loss"].append(train_rl)

		iter +=1

		# print every round of 10 batches the losses - smooths the results 
		if batch % 10 == 0:
			print(f"Batch {batch} and a total {batch*64}/{len(train_loader.dataset)} proteins have passed.")
			print(f"Current Loss: {train_loss/(batch+1)} | KL Loss: {train_kl/(batch+1)}| Rec Loss: {train_rl/(batch+1)}")


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
		for val_batch, t_mask, tidx in test_loader:

			x_mu, x_logvar, z_mu, z_logvar = model3(val_batch)
			loss = loss_fun(val_batch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=t_mask,freebits=freebits)
			val_loss += loss.detach().item()
			val_kl += lst[-1]
			val_rl += lst[-2]
		
		# divide by all the batches of val set to get epoch metrics 
		val_loss = val_loss/len(test_loader)
		val_kl = val_kl/len(test_loader)
		val_rl = val_rl/len(test_loader)

		epoch_dict["Val total Loss"].append(val_loss)
		epoch_dict["Val KL Loss"].append(val_kl)
		epoch_dict["Val Rec Loss"].append(val_rl)

	## Print out what's happening
	print(f"\nTrain loss: {train_loss:.5f}|Train Rec: {train_rl:.5f} | Val loss: {val_loss:.3f}, Val Rec: {val_rl:.3f}\n")



# plot with sebastians code 
def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}',
             color="green", alpha=0.8)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    if np.min(minibatch_losses) > 0:
        ax1.set_ylim([
            np.min(minibatch_losses[num_losses:])*0.98,
            np.max(minibatch_losses[num_losses:])*1.02
        ])

    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}', color="purple")
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()




### The train-val loop backup
model = model1
loss_fun = cf.loss_fun
epoch = 10
learn_r = 0.005
freebits = 1.25
batch_size = 128
norm = 0 

# set optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=learn_r)

# create a string for the hyperparameters 
hyperparam_str = f"norm{norm}_bits{freebits}_bs{batch_size}_lr{optimizer.param_groups[0]["lr"]}"


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
            for val_batch, t_mask, tidx in test_loader:
                x_mu, x_logvar, z_mu, z_logvar = model(val_batch)
                loss = loss_fun(val_batch, x_mu, x_logvar, z_mu, z_logvar,lst,mask=t_mask,freebits=freebits)
                val_loss += loss.detach().item()
                val_kl += lst[-1]
                val_rl += lst[-2]
            
            val_loss = val_loss/len(test_loader)
            val_kl = val_kl/len(test_loader)
            val_rl = val_rl/len(test_loader)
            
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
        print("Iter initialized before loop")			
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
            if batch % 10 == 0:
                print(f"Iter {batch} and a total {batch*batch_size}/{len(train_loader.dataset)} proteins have passed.")
                print(f"Current Loss: {train_loss/(batch+1)} | KL Loss: {train_kl/(batch+1)}| Rec Loss: {train_rl/(batch+1)}")


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
        print(f"\nTrain loss: {train_loss:.3f}|Train Rec: {train_rl:.3f} | Val loss: {val_loss:.3f}, Val Rec: {val_rl:.3f}\n")