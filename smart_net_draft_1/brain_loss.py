import torch
import torch.nn as nn
import torch.optim as optim


def brain_loss(output_c, output_s, target, criterion_, running_loss_, device, skipable_layer,update_weights, loss ):
	out , predicted = torch.max(output_c, 1)
	loss_ = torch.rand_like(out)
	if target == predicted :
		temp_target = torch.randint(0,1,(1, len(output_s))).type(torch.float32).to(device)
		if output_s[-1] > 0.5 :
			temp_target[0][-1] = 1
		for i in range(len(output_s)):
			loss_ = (len(output_s))*criterion_(output_s[i], temp_target[0][i])
			loss_ += loss_
			running_loss_ += loss_.item()
	elif target != predicted :
		if output_s[-1] < 0.5 :
			temp_target = torch.randint(1,2,(1,1)).type(torch.float32).to(device)
			loss_ = 0.2*criterion_(output_s[-1], temp_target[0])
			loss_ += loss_
			running_loss_ += loss_.item()
		else:
			temp_target = torch.randint(0,1,(1,1)).type(torch.float32).to(device)
			loss_ = 0.2*criterion_(output_s[-1], temp_target[0])
			loss_ += loss_
			running_loss_ += loss_.item()

	loss_ = 0*loss_ + loss
	
	return loss_, running_loss_
