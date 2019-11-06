
import numpy as np
import argparse
import os
import json
import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from time import time

from utils import *
from data import input_data
from model import *


parser = argparse.ArgumentParser(description = "1st draft of SMART_NET")
parser.add_argument("--epochs", default = 10, help = "enter the no. of epochs")
parser.add_argument("--batch-size", type=int, default = 32, help = "enter the batch size")
parser.add_argument("--learning-rate", type = float, default = 0.001, help = "enter the learning rate")
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--less-than', type=float, default=0.75,help='less than this probability ')
parser.add_argument('--gpu-rank', default=0,help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--data-valid',help='enter the directory where the validation data of different classes is saved')
parser.add_argument('--data-train',help='enter the directory where the train data of different classes is saved')
parser.add_argument('--save-path',help='enter the directory to save weights')
parser.add_argument("--pll", action='store_true',help='use multi-gpu or no')
parser.add_argument("--weights", action='store_true',help='use weights or no')
args = parser.parse_args()

input_train = input_data(root_dir = args.data_train, type = "train")
train_dl =  DataLoader(input_train, batch_size=args.batch_size,shuffle=True, num_workers=4)

input_valid = input_data(root_dir = args.data_valid, type = "valid")
valid_dl =  DataLoader(input_valid, batch_size=2,shuffle=False, num_workers=4)

less_than = args.less_than
# os.system("rm prob.txt")

if __name__ == '__main__':
	
	feature_1 = feature_b()
	feature_2 = feature_r()
	decision_ = decision(out_classes = input_train[0][3])
	models = [feature_1, feature_2, decision_]

	if args.pll == True : 
		for i  in models:
			i = nn.DataParallel(i)	

	device = torch.device("cuda" if args.cuda else "cpu")
	if args.gpu_rank and args.pll == False:
		torch.cuda.set_device(int(args.gpu_rank))

	if args.weights == True:
		print("pre-trained weights are used")
		feature_1.load_state_dict(torch.load("../weights/feature_1.pth",map_location="cuda:"+str(args.gpu_rank)), strict = True)
		feature_2.load_state_dict(torch.load("../weights/feature_2.pth",map_location="cuda:"+str(args.gpu_rank)), strict = True)
		try:decision_.load_state_dict(torch.load("../weights/decision.pth",map_location="cuda:"+str(args.gpu_rank)), strict = True)
		except:pass
	# for param in model.parameters():
	# 	param.requires_grad = False

	for i in models:
		i.to(device)

	# summary(decision_,(512,224,224))
	print("no of parameters is : ", (get_param_size([feature_1,feature_2,decision_]))/1000000," Million")

	optimizer_1 = optim.Adam(feature_1.parameters(), lr=args.learning_rate*10)
	optimizer_2 = optim.Adam(feature_2.parameters(), lr=args.learning_rate*10)
	optimizer_d = optim.Adam(decision_.parameters(), lr=args.learning_rate)	
	optimizer_1.zero_grad()
	optimizer_2.zero_grad()
	optimizer_d.zero_grad()
	
	criterion_d = nn.CrossEntropyLoss()
	
	accuracy = 0
	for j in range(int(args.epochs)):
		print("start of epoch: ", j+1)
		#Training
		for k in models:
			k.train()
		running_loss, switch_ = 0, 0
		less_than = less_than + (1-less_than)*0.03
		print("less than this :",less_than)
		start = time()
		for i, data in enumerate(train_dl, 0):
			
			input, target, img_name, number_of_class = data
			input, target = (input.type(torch.float32)).to(device), target.to(device)
			
			out_1 = feature_1(input)
			out = decision_(out_1)
			

			if switch(out, less_than, target,"train") == True:
				switch_ +=1
				
				loss_1 = criterion_d(out, target)
				loss_1.backward()

				out_2 = feature_2(out_1.clone().detach())
				out = decision_(out_2)
				loss = criterion_d(out, target)
				loss.backward()
				optimizer_1.step()
				optimizer_2.step()
				optimizer_d.step()

				running_loss += loss.item() + loss_1.item()
				

			else: 
				loss = criterion_d(out, target)
				loss.backward()
				optimizer_1.step()
				optimizer_d.step() 
				running_loss += loss.item()
			

			optimizer_1.zero_grad()
			optimizer_2.zero_grad()
			optimizer_d.zero_grad()

			# print every 25 mini-batches
			if i % 25 == 24: 	
				print('[%d, %5d] loss: %.3f' %(j + 1, i + 1, running_loss))
				running_loss = 0
		end = time()
		print(" It took : ", (end - start)/60, " mins for the last training epoch")
		print("in training the model switched ",(switch_/(i+1))*100, " % times in the previous epoch")

		# Validation
		running_loss, acc, num, length, switch_ = 0, 0, 0, 0, 0
		with torch.no_grad():
			start = time()
			for i, data in enumerate(valid_dl, 0):	
				for j in models:
					j.eval()

				input, target, img_name, number_of_class = data
				input, target = (input.type(torch.float32)).to(device), target.to(device)

				out_1 = feature_1(input)
				out = decision_(out_1)

				if switch(out, less_than) == True:
					switch_ +=1
					out_2 = feature_2(out_1)
					out = decision_(out_2)

				loss = criterion_d(out, target)
				running_loss += loss.item()

				out , predicted = torch.max(out, 1)
				for k in range(len(target)):
					if target[k] == predicted[k].item():
						num = num + 1
				length = length + len(target)
			acc = (num/length)*100
			end = time()
			print("accuracy and val loss is : ",acc,",",running_loss/(i+1), " --AND-- ", " It took : ", (end - start), " seconds ")
			print("in validation the model switched ",(switch_/(i+1))*100, " % times in the previous epoch")

		if acc > accuracy :
			accuracy = acc
			try: 
				feature_1_state_dict = feature_1.module.state_dict()
				feature_2_state_dict = feature_2.module.state_dict()
				decision_state_dict = decision_.module.state_dict()
			except : 
				feature_1_state_dict = feature_1.state_dict()
				feature_2_state_dict = feature_2.state_dict()
				decision_state_dict = decision_.state_dict()
			torch.save(feature_1_state_dict,str(args.save_path) + "feature_1" + ".pth")
			torch.save(feature_2_state_dict,str(args.save_path) + "feature_2" + ".pth")
			torch.save(decision_state_dict,str(args.save_path) + "decision" + ".pth")

	print("training is finished")
