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
parser.add_argument('--skipable-layer',default = 0,help='number of skipable layers')
parser.add_argument("--random", action='store_true',help='if skipping is random or no')
parser.add_argument('--save-path',help='enter the directory to save weights')

args = parser.parse_args()

input_train = input_data(root_dir = args.data_train, type = "train")
train_dl =  DataLoader(input_train, batch_size=args.batch_size,shuffle=True, num_workers=4)
# print(input_train[0][3])
# exit()

input_valid = input_data(root_dir = args.data_valid, type = "valid")
valid_dl =  DataLoader(input_valid, batch_size=2*args.batch_size,shuffle=False, num_workers=4)


if __name__ == '__main__':
	update_weights = args.update_weights
	random = args.random

	model = draft(skipable_layers = int(args.skipable_layer), out_classes = input_train[0][3], random = random)
	# model = nn.DataParallel(model)
	brain = sub_draft()

	
	if args.model_weights_path or args.brain_weights_path:
		try:
			model.load_state_dict(torch.load(args.model_weights_path), strict = True)
			brain.load_state_dict(torch.load(args.brain_weights_path), strict = True)
			print("trying")
		except:
			try:
				model_pretrained_dict = torch.load(args.model_weights_path)
				bain_rpretrained_dict = torch.load(args.brain_weights_path)
				for param_tensor in model.state_dict():
					model.state_dict()[param_tensor][:] = model_pretrained_dict["state_dict"][param_tensor][:]
				for param_tensor in brain.state_dict():
					brain.state_dict()[param_tensor][:] = brain_pretrained_dict["state_dict"][param_tensor][:]
			except: pass
			print("except")

	device = torch.device("cuda" if args.cuda else "cpu")
	if args.gpu_rank:
		torch.cuda.set_device(int(args.gpu_rank))
	model.to(device)
	brain.to(device)

	# for param in model.parameters():
	# 	param.requires_grad = False

	# summary(model, (3,224,224))

	print("no of parameters: %d" % (get_param_size(model)+get_param_size(brain)))
	criterion = nn.CrossEntropyLoss()
	criterion_ = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) 
	optimizer_ = optim.Adam(brain.parameters(), lr = 0.01*args.learning_rate)
	acc = 0
	out_l = []
	print("number of skipable_layers : ", args.skipable_layer)

	for j in range(int(args.epochs)):
		model.train()
		running_loss = 0
		running_loss_ = 0
		print("start of epoch: ", j+1)
		m = 0
		optimizer.zero_grad()
		optimizer_.zero_grad()
		temp_ = np.random.randint(int(args.skipable_layer)+1, size=1)
		for i, data in enumerate(train_dl, 0):
			
			input, target, img_name, number_of_class = data
			
			input, target = (input.type(torch.float32)).to(device), target.to(device)

			output_c, output_s = model(input,temp_, brain, device)

			loss = criterion(output_c, target)

			if len(output_s) !=0 and random == False:
				loss_, running_loss_ = brain_loss(output_c, output_s, target, criterion_, running_loss_, device, args.skipable_layer,update_weights, loss.clone().detach())
				(loss_/update_weights).backward()

			(loss/update_weights).backward()

			running_loss += loss.item()

			if i % 50 == 49: # print every 50 mini-batches	
				print('[%d, %5d] dumb_loss: %.3f' %(j + 1, i + 1, running_loss / update_weights))
				print('[%d, %5d] smart_loss: %.3f' %(j + 1, i + 1, running_loss_ / (update_weights)))
				running_loss = 0
				running_loss_ = 0
			
			if i % update_weights == update_weights - 1:    # update weights as defined	
				optimizer.step()
				optimizer.zero_grad()
				optimizer_.step()
				optimizer_.zero_grad()

		print("testing is : ", random)

		if random == True:
			accuracy = 0
			with torch.no_grad():
				for temp_ in range(int(args.skipable_layer)+1):
					num = 0
					length = 0 
					start = time()
					for i, data in enumerate(valid_dl, 0):	
						model.eval()
						input, target, img_name, number_of_class = data
						input= (input.type(torch.float32)).to(device)

						output_c, output_s = model(input,temp_, brain, device)

						out , predicted = torch.max(output_c, 1)
						# print("out", target)
						# print("predicted", predicted)
						for i in range(len(target)):
							if target[i] == predicted[i].item():
								num = num + 1
						length = length + len(target)
					accuracy = (num/length)*100
					end = time()
					print("accuracy at ",temp_," : ",accuracy, "  ", "time : ", (end - start))
					out_l.append([temp_,accuracy,(end-start)])

		elif random == False:
			accuracy = 0
			with torch.no_grad():
				num = 0
				length = 0 
				pred = []
				len_ = []
				start = time()
				for i, data in enumerate(valid_dl, 0):	
					model.eval()
					input, target, img_name, number_of_class = data
					input= (input.type(torch.float32)).to(device)

					output_c, output_s = model(input,temp_, brain, device)
					len_.append(len(output_s))

					out , predicted = torch.max(output_c, 1)
					pred.append(predicted)
					# print(i,predicted)
					# exit()
					for i in range(len(target)):
						if target[i] == predicted[i].item():
							num = num + 1
					length = length + len(target)
				accuracy = (num/length)*100
				end = time()
				print("accuracy : ",accuracy, "  ", "time : ", (end - start))
				# print(pred)
				print(len_)
				# exit()

		if accuracy >= acc :
			acc = accuracy
			torch.save(model.state_dict(),"/home/hemant/hem/nips/weights/model_layers_" + str(args.skipable_layer)+"_" + str(acc) + "_nips.pth")
			torch.save(brain.state_dict(),"/home/hemant/hem/nips/weights/brain_layers_" + str(args.skipable_layer)+"_" + str(acc) + "_nips.pth")
		np.save("out_l",out_l)
	print("training is finished")
