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


parser = argparse.ArgumentParser(description = "1st draft of NIPS")
parser.add_argument("--epochs", default = 10, help = "enter the no. of epochs")
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--weights-path', default=None,help='Location of the  best validation model weight')
parser.add_argument('--gpu-rank', default=0,help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--data-test',help='enter the directory where the validation data of different classes is saved')
parser.add_argument('--skipable-layer',default = 0,help='number of skipable layers')

args = parser.parse_args()
print(args.skipable_layer)

input_test = input_data(root_dir = args.data_test, type = "train")
test_dl =  DataLoader(input_test, batch_size=128,shuffle=True, num_workers=4)

if __name__ == '__main__':
	model = draft(skipable_layers = int(args.skipable_layer), out_classes = input_test[0][3],random = True)
	# model = nn.DataParallel(model)
	
	
	if args.weights_path:
		try:
			model.load_state_dict(torch.load(args.weights_path), strict = False)
			print("trying")
		except:
			pretrained_dict = torch.load(args.weights_path)
			try:
				for param_tensor in model.state_dict():
					model.state_dict()[param_tensor][:] = pretrained_dict["state_dict"][param_tensor][:]
			except: pass
			print("except")

	device = torch.device("cuda:0" if args.cuda else "cpu")
	if args.gpu_rank:
		torch.cuda.set_device(int(args.gpu_rank))
	model.to(device)

	# summary(model, (3,224,224))


	print("no of parameters: %d" % get_param_size(model)) 


	with torch.no_grad():
		num = 0
		length = 0
		for temp_ in range(int(args.skipable_layer)+1):
			for i, data in enumerate(test_dl, 0):	
				model.eval()
				input, target, img_name, number_of_class = data
				input= (input.type(torch.float32)).to(device)

				print(temp_)
				output_c, output_s = model(input,temp_)

				out , predicted = torch.max(output_c, 1)
				for i in range(len(target)):
					if target[i] == predicted[i].item():
						num = num + 1
				length = length + len(target)
			accuracy = (num/length)*100
			print(accuracy)

	print("testing is finished")
