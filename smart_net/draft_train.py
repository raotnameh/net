import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from utils import get_param_size
from draft_data import input_data
from class_ import draft
from sub_class import sub_draft
from brain_loss import brain_loss
from weight_load import load_weight
from torchsummary import summary
from torchvision import models
from time import time
import os


parser = argparse.ArgumentParser(description = "1st draft of NIPS")
parser.add_argument("--epochs", default = 10, help = "enter the no. of epochs")
parser.add_argument("--batch-size", type=int, default = 1, help = "enter the batch size")
parser.add_argument("--update-weights", type = int, default = 128, help = "enter the no. of steps after which to update the weight")
parser.add_argument("--learning-rate", type = float, default = 0.001, help = "enter the learning rate")
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--model-weights-path', default=None,help='Location of the  best model validation model weight')
parser.add_argument('--brain-weights-path', default=None,help='Location of the  best brain validation model weight')
parser.add_argument('--gpu-rank', default=0,help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--data-valid',help='enter the directory where the validation data of different classes is saved')
parser.add_argument('--data-train',help='enter the directory where the train data of different classes is saved')
parser.add_argument('--skipable-layer',default = 5,help='number of skipable layers')
parser.add_argument("--random", action='store_true',help='if skipping is random')
parser.add_argument("--brain", action='store_true',help='use brain or no')
parser.add_argument('--save-path',help='enter the directory to save weights')
parser.add_argument("--baseline", action='store_true',help='baseline architecture')

args = parser.parse_args()

input_train = input_data(root_dir = args.data_train, type = "train")
train_dl =  DataLoader(input_train, batch_size=args.batch_size,shuffle=True, num_workers=4)

input_valid = input_data(root_dir = args.data_valid, type = "valid")
valid_dl =  DataLoader(input_valid, batch_size=2*args.batch_size,shuffle=False, num_workers=4)


if __name__ == '__main__':
	update_weights = args.update_weights
	random = args.random

	model = draft(skipable_layers = int(args.skipable_layer), out_classes = input_train[0][3], random = random)
	brain = sub_draft()
	
	# else: model = nn.DataParallel(model)

	if args.model_weights_path == None or args.brain_weights_path == None:
		if args.brain == True: load_weight(model,args.model_weights_path,args.brain_weights_path, brain)
		else: load_weight(model,args.model_weights_path,args.brain_weights_path,)
	device = torch.device("cuda" if args.cuda else "cpu")
	if args.gpu_rank:
		torch.cuda.set_device(int(args.gpu_rank))
	model.to(device)
	if args.brain == True: brain.to(device)

	# for param in model.parameters():
	# 	param.requires_grad = False

	# summary(model, input_size = (3,224,224))

	print("no of parameters in the main class is : ", (get_param_size(model))/1000000," Million")
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) 

	if args.brain == True:
		criterion_ = nn.BCELoss()
		optimizer_ = optim.Adam(brain.parameters(), lr = 0.1*args.learning_rate)
	else: brain == None
	
	acc = 0
	out_l = []
	print("number of skipable_layers : ", args.skipable_layer)

	for j in range(int(args.epochs)):

		#Training

		model.train()
		running_loss = 0
		running_loss_ = 0
		print("start of epoch: ", j+1)
		m = 0
		optimizer.zero_grad()
		if args.brain == True: optimizer_.zero_grad()
		temp_ = np.random.randint(int(args.skipable_layer)+1, size=1)
		for i, data in enumerate(train_dl, 0):
			
			input, target, img_name, number_of_class = data
			
			input, target = (input.type(torch.float32)).to(device), target.to(device)

			if input.shape[1] != 3 : continue	
			output_c, output_s = model(input, device, args.baseline, temp_, brain )

			loss = criterion(output_c, target)

			if args.brain == True:
				loss_, running_loss_ = brain_loss(output_c, output_s, target, criterion_, running_loss_, device, args.skipable_layer,update_weights, loss.clone().detach())
				(loss_/update_weights).backward()

			(loss/update_weights).backward()

			running_loss += loss.item()

			# print every 50 mini-batches
			if i % 50 == 49: 	
				print('[%d, %5d] dumb_loss: %.3f' %(j + 1, i + 1, running_loss / update_weights))
				if args.brain == True: print('[%d, %5d] smart_loss: %.3f' %(j + 1, i + 1, running_loss_ / (update_weights)))
				running_loss = 0
				running_loss_ = 0

			# update weights as defined
			if i % update_weights == update_weights - 1:    	
				optimizer.step()
				optimizer.zero_grad()
				if args.brain == True:
					optimizer_.step()
					optimizer_.zero_grad()


		# Validation

		print("validation is : ", random)
		accuracy = 0

		if args.baseline == True or random == False:
			with torch.no_grad():
				num = 0
				length = 0
				start = time()
				for i, data in enumerate(valid_dl, 0):	
					model.eval()
					input, target, img_name, number_of_class = data
					input= (input.type(torch.float32)).to(device)

					output_c, output_s = model(input, device, args.baseline, temp_, brain)
					out , predicted = torch.max(output_c, 1)

					for i in range(len(target)):
						if target[i] == predicted[i].item():
							num = num + 1
					length = length + len(target)
				accuracy = (num/length)*100
				end = time()

				print("accuracy is : ",accuracy, " AND ", " It took : ", (end - start), " seconds ")
		
		elif random == True:
			with torch.no_grad():
				for temp_ in range(int(args.skipable_layer)+1):
					num = 0
					length = 0 
					start = time()
					for i, data in enumerate(valid_dl, 0):	
						model.eval()
						input, target, img_name, number_of_class = data
						input= (input.type(torch.float32)).to(device)

						output_c, output_s = model(input, device, args.baseline, temp_, brain)

						out , predicted = torch.max(output_c, 1)

						for i in range(len(target)):
							if target[i] == predicted[i].item():
								num = num + 1
						length = length + len(target)
					accuracy = (num/length)*100
					end = time()
					print("accuracy at ", temp_, " is : ",accuracy, " AND ", " It took : ", (end - start), " seconds ")		

		if accuracy >= acc :
			acc = accuracy
			torch.save(model.state_dict(),"/home/hemant/hem/nips/weights/model_layers_" + str(acc) + "_nips.pth")
			try: torch.save(brain.state_dict(),"/home/hemant/hem/nips/weights/brain_layers_" + str(acc) + "_nips.pth")
			except: pass
	print("training is finished")
