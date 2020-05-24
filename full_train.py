from time import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import input_data

from utils import *
from data import input_data
from model import *


parser = argparse.ArgumentParser(description = "1st draft of SMART_NET")
parser.add_argument("--epochs", default = 10, help = "enter the no. of epochs")
parser.add_argument("--batch-size", type=int, default = 32, help = "enter the batch size")
parser.add_argument("--learning-rate", type = float, default = 0.001, help = "enter the learning rate")
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--gpu-rank', default=2,help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--data-valid',default = "/home/hemant/net/easy_net/data/val/",help='enter the directory where the validation data of different classes is saved')
parser.add_argument('--data-train',default = "/home/hemant/net/easy_net/data/train/",help='enter the directory where the train data of different classes is saved')
parser.add_argument("--pll", action='store_true',help='use multi-gpu or no')
parser.add_argument("--weights", action='store_true',help='use weights or no')
args = parser.parse_args()


input_train = input_data(root_dir = args.data_train, type = "train")
train_dl =  DataLoader(input_train, batch_size=args.batch_size,shuffle=True, num_workers=4)
input_valid = input_data(root_dir = args.data_valid, type = "valid")
valid_dl =  DataLoader(input_valid, batch_size=args.batch_size*2,shuffle=False, num_workers=4)

if __name__ == '__main__':

	model = nn.Sequential(  feature_b(),
							feature_r(),
							decision(out_classes = input_train[0][3]))

	rank = args.gpu_rank
	device = torch.device("cuda:"+str(rank) if args.cuda else "cpu")
	if args.gpu_rank == True and args.pll == False:
		torch.cuda.set_device(int(args.gpu_rank))

	model.to(device)
	if args.weights == True:
		model.load_state_dict(torch.load("../weights/full.pth",map_location="cuda:"+str(rank)),strict=True)

	optimizer = optim.Adam(model.parameters(), lr=0.001)
	criterion_d = nn.CrossEntropyLoss()

	accuracy = 0
	for j in range(200):
		print("start of epoch: ", j+1)
		#Training
		model.train()
		running_loss = 0
		start = time()
		for i, data in enumerate(train_dl, 0):

			input, target, img_name, number_of_class = data
			input, target = (input.type(torch.float32)).to(device), target.to(device)

			out = model(input)

			loss = criterion_d(out, target)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()


			optimizer.zero_grad()

			# print every 25 mini-batches
			if i % 25 == 24:
				print('[%d, %5d] loss: %.3f' %(j + 1, i + 1, running_loss))
				running_loss = 0
		end = time()
		print(" It took : ", (end - start)/60, " mins for the last training epoch")

		running_loss, acc, num, length = 0, 0, 0, 0
		with torch.no_grad():
			start = time()
			for i, data in enumerate(valid_dl, 0):
				model.eval()

				input, target, img_name, number_of_class = data
				input, target = (input.type(torch.float32)).to(device), target.to(device)

				out = model(input)

				loss = criterion_d(out, target)

				out , predicted = torch.max(out, 1)
				for k in range(len(target)):
					if target[k] == predicted[k].item():
						num = num + 1
				length = length + len(target)
			acc = (num/length)*100
			end = time()
			if acc>accuracy:
				torch.save(model.state_dict(),"../weights/full.pth")
				accuracy = acc
			print("accuracy and val loss is : ",acc,",",running_loss/(i+1), " --AND-- ", " It took : ", (end - start), " seconds ")