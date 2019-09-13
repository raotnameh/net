import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from utils import get_param_size
from draft_data import input_data
from torchsummary import summary
from torchvision import models
from time import time
import os

parser = argparse.ArgumentParser(description = "1st draft of NIPS")
parser.add_argument("--epochs", default = 25, help = "enter the no. of epochs")
parser.add_argument("--batch-size", type=int, default = 16, help = "enter the batch size")
parser.add_argument("--learning-rate", type = float, default = 0.001, help = "enter the learning rate")
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--model-weights-path', default=None,help='Location of the  best model validation model weight')
parser.add_argument('--gpu-rank', default=0,help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--data-valid',help='enter the directory where the validation data of different classes is saved')
parser.add_argument('--data-train',help='enter the directory where the train data of different classes is saved')

args = parser.parse_args()

input_train = input_data(root_dir = args.data_train, type = "train")
train_dl =  DataLoader(input_train, batch_size=args.batch_size,shuffle=True, num_workers=4)
# print(input_train[0][3])
# exit()

input_valid = input_data(root_dir = args.data_valid, type = "valid")
valid_dl =  DataLoader(input_valid, batch_size=2*args.batch_size,shuffle=False, num_workers=4)


if __name__ == '__main__':

	model = models.resnet18(pretrained=True)
	# for param in model.parameters():
	#     param.requires_grad = False

	# Parameters of newly constructed modules have requires_grad=True by default
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, input_train[0][3])
		
	try: model.load_state_dict(torch.load(args.model_weights_path), strict = True)
	except: pass	
	device = torch.device("cuda" if args.cuda else "cpu")
	if args.gpu_rank:
		torch.cuda.set_device(int(args.gpu_rank))
	model.to(device)
	
	summary(model, (3,224,224))

	print("no of parameters: ", (get_param_size(model)/1000000, " Million"))
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) 

	# exit()


	for j in range(int(args.epochs)):
		model.train()
		running_loss = 0
		print("start of epoch: ", j+1)
		m = 0
		optimizer.zero_grad()
		for i, data in enumerate(train_dl, 0):
			
			input, target, img_name, number_of_class = data
			
			input, target = (input.type(torch.float32)).to(device), target.to(device)

			output = model(input)

			loss = criterion(output, target)
			running_loss += loss.item()

			if i % 50 == 49: # print every 50 mini-batches	
				print('[%d, %5d] dumb_loss: %.3f' %(j + 1, i + 1, running_loss ))
				running_loss = 0
				
			optimizer.step()
			optimizer.zero_grad()
			
		accuracy = 0
		with torch.no_grad():
			num = 0
			length = 0
			start = time()
			for i, data in enumerate(valid_dl, 0):	
				model.eval()
				input, target, img_name, number_of_class = data
				input= (input.type(torch.float32)).to(device)

				output = model(input)
				out , predicted = torch.max(output, 1)

				for i in range(len(target)):
					if target[i] == predicted[i].item():
						num = num + 1
				length = length + len(target)
			accuracy = (num/length)*100
			end = time()

			print("accuracy is : ",accuracy, " AND ", " It took : ", (end - start), " seconds ")
		torch.save(model.state_dict(),"/media/data_dump/hemant/records/baseline/weights_0/model_layers_"+str(j)+"_" + str(accuracy) + "_nips.pth")
	