import numpy as np
import argparse
import os
import json
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time

from utils import *
from data import input_data
from model import feature_b, feature_r, decision_r, gru_layer
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]= "4,3"

parser = argparse.ArgumentParser(description = "1st draft of SMART_NET")
parser.add_argument("--epochs", default = 50, help = "enter the no. of epochs")
parser.add_argument("--batch-size", type=int, default = 64, help = "enter the batch size")
parser.add_argument("--learning-rate", type = float, default = 0.01, help = "enter the learning rate")
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--steps', type=int, default=1,help='number of recursive layers')
parser.add_argument('--gpu-rank', default=0,help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--data-valid',default='data/val/',help='enter the directory where the validation data of different classes is saved')
parser.add_argument('--data-train',default='data/train/',help='enter the directory where the train data of different classes is saved')
parser.add_argument('--save-path',help='enter the directory to save weights')
parser.add_argument("--pll", action='store_true',help='use multi-gpu or no')
parser.add_argument("--weights", action='store_true',help='use weights or no')
args = parser.parse_args()

input_train = input_data(root_dir = args.data_train, type = "train")
train_dl =  DataLoader(input_train, batch_size=args.batch_size,shuffle=True, num_workers=4)

input_valid = input_data(root_dir = args.data_valid, type = "valid")
valid_dl =  DataLoader(input_valid, batch_size=args.batch_size,shuffle=False, num_workers=4)

# data sample: image, landmarks, img_name, self.number_of_class

if __name__ == '__main__':
	
	base_feature = feature_b()
	base_decision = decision_r()
	model = [[base_feature, base_decision]]
	recursive_layers = [[f"feature_{i}", f"decision_{i}"] for i in range(1,args.steps+1)]
	for i in recursive_layers:
		model.append([feature_r(),decision_r()])

	device = torch.device("cuda" if args.cuda else "cpu")
	if args.gpu_rank == True and args.pll == False:
		torch.cuda.set_device(int(args.gpu_rank))

	# if args.weights == True:
	# 	feature_1.load_state_dict(torch.load("../weights/final_feature_1.pth",map_location="cuda:"+str(args.gpu_rank)), strict = True)

	for layer in model:
		for sub_layer in layer:
			for params in sub_layer.parameters():
				params.requires_grad = True

	for i in model:
		for j in i:
			j.to(device)
	
	hidden_size = 512
	rnn_layer = gru_layer(4096, hidden_size,batch_first=False, classes = input_train.classes()).to(device)

	optimizers = []
	for rank, layers in enumerate(model):
		optimizers.append(optim.Adam(list(layers[0].parameters()) + list(layers[1].parameters()), lr=args.learning_rate))
		optimizers[rank].zero_grad()
	
	criterion = nn.CrossEntropyLoss()
	
	# print(model)
	print("no of parameters is : ", (get_param_size([j for i in model for j in i]))/1000000," Million")
	accuracy = 0
	for j in range(int(args.epochs)):
		for q in model:
			for w in q:
				w.train()
		running_loss = [0.0, 0.0]
		start = time()
		print("start of epoch: ", j+1)
		#Training
		for i, data in enumerate(train_dl, 0):
			
			input, target, img_name, number_of_class = data
			input, target = (input.type(torch.float32)).to(device), target.to(device)
			
			f_1 = model[0][0](input)
			d_1 = model[0][1](f_1)

			f_2 = model[1][0](f_1.clone().detach())
			d_2 = model[1][1](f_2)
			# print(d_2.shape)
			# exit()	
			# break
			out = torch.cat((d_1,d_2)).view(2,d_2.shape[0],-1)
			
			hidden_layer = torch.zeros(1,d_2.shape[0], hidden_size,dtype=torch.float32).to(device)
			out = rnn_layer(out,hidden_layer)

			
			# print(out[:,-1,:])
			# print(out[:,0,:].shape)
			loss = []
			for k in range(args.steps+1):
				loss.append(criterion(out[:,k,:],target))
			
			for r, l in enumerate(loss):
				l.backward(retain_graph=True)
				running_loss[r] += l.item()

			for m in optimizers:
				m.step()
			for jk in optimizers:
				jk.zero_grad()
		
			# print every 25 mini-batches
			if i % 10 == 10 - 1:	
				print('[%d, %5d] loss1: %.3f, loss2: %.3f' %(j + 1, i + 1, running_loss[0]/10, running_loss[1]/10))
				running_loss = [0.0, 0.0]
		end = time()
		print(" It took : ", (end - start)/60, " mins for the last training epoch")
		# print("in training the model switched ",(switch_/(i+1))*100, " % times in the previous epoch")
		# continue
		# Validation
		running_loss, acc, num, length =  [0, 0], [0, 0],  [0, 0], 0
		with torch.no_grad():
			start = time()
			for i, data in tqdm(enumerate(valid_dl, 0)):	
				for q in model:
					for w in q:
						w.eval()
				input, target, img_name, number_of_class = data
				input, target = (input.type(torch.float32)).to(device), target.to(device)

				f_1 = model[0][0](input)
				d_1 = model[0][1](f_1)

				f_2 = model[1][0](f_1.clone().detach())
				d_2 = model[1][1](f_2)
				print()

				out = torch.cat((d_1,d_2)).view(2,d_2.shape[0],-1)
				hidden_layer = torch.zeros(1,d_2.shape[0], hidden_size,dtype=torch.float32).to(device)
				out = rnn_layer(out,hidden_layer)

				loss = []
				for k in range(args.steps+1):
					l = criterion(out[:,k,:],target)
					loss.append(l)
					running_loss[k] += l.item()

					_, predicted = torch.max(out[:,k,:], 1)
					# print(predicted)
					# print(out[:,k,:])
					for df in range(len(target)):
						if target[df] == predicted[df].item():
							num[k] += 1
				length = length + len(target)
				# break
				
			for i in range(len(acc)):
				acc[i] = (num[i]/length)*100
				print(f"accuracy is: {acc[i]}")
			end = time()



			# print("accuracy and val loss is : ",acc,",",running_loss/(i+1), " --AND-- ", " It took : ", (end - start), " seconds ")
			# print("in validation the model switched ",(switch_/(i+1))*100, " % times in the previous epoch")

		# if acc > accuracy :
		# 	accuracy = acc
		# 	try: 
		# 		feature_1_state_dict = feature_1.module.state_dict()
		# 		feature_2_state_dict = feature_2.module.state_dict()
		# 		decision_state_dict = decision_.module.state_dict()
		# 	except : 
		# 		feature_1_state_dict = feature_1.state_dict()
		# 		feature_2_state_dict = feature_2.state_dict()
		# 		decision_state_dict = decision_.state_dict()
		# 	torch.save(feature_1_state_dict,str(args.save_path) + "final_feature_1" + ".pth")
		# 	torch.save(feature_2_state_dict,str(args.save_path) + "final_feature_2" + ".pth")
		# 	torch.save(decision_state_dict,str(args.save_path) + "final_decision" + ".pth")

	print("training is finished")
