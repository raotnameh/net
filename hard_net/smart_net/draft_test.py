import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from utils import get_param_size
from draft_data import input_data
from class_ import draft
from sub_class import sub_draft
from weight_load import load_weight
from torchvision import models
from time import time
import os



parser = argparse.ArgumentParser(description = "1st draft of NIPS")
parser.add_argument("--epochs", default = 10, help = "enter the no. of epochs")
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--data-test',help='enter the directory where the validation data of different classes is saved')
parser.add_argument('--skipable-layer',default = 0,help='number of skipable layers')
parser.add_argument("--batch-size", type=int, default = 1, help = "enter the batch size")
parser.add_argument('--model-weights-path', default=None,help='Location of the  best model validation model weight')
parser.add_argument('--brain-weights-path', default=None,help='Location of the  best brain validation model weight')
parser.add_argument('--gpu-rank', default=0,help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument("--brain", action='store_true',help='use brain or no')
parser.add_argument("--baseline", action='store_true',help='baseline architecture')
parser.add_argument("--random", action='store_true',help='if skipping is random')


args = parser.parse_args()

args = parser.parse_args()
print("number of skipable_layers : ",args.skipable_layer)

input_test = input_data(root_dir = args.data_test, type = "train")
test_dl =  DataLoader(input_test, batch_size=128,shuffle=True, num_workers=4)

if __name__ == '__main__':
	model = draft(skipable_layers = int(args.skipable_layer), out_classes = input_test[0][3], random = args.random)

	brain = None
	if args.brain == True:
		brain = sub_draft()
		criterion_ = nn.BCELoss()
		optimizer_ = optim.Adam(brain.parameters(), lr = 0.1*args.learning_rate)
	else: brain == None
	
	model.load_state_dict(torch.load(args.model_weights_path), strict = True)
	if args.brain == True: brain.load_state_dict(torch.load(args.brain_weights_path), strict = True)

	
	device = torch.device("cuda" if args.cuda else "cpu")
	if args.gpu_rank:
		torch.cuda.set_device(int(args.gpu_rank))
	model.to(device)
	if args.brain == True: brain.to(device)

	# summary(model, (3,224,224))


	print("no of parameters: %d" % get_param_size(model)) 
	print("Testing is random : ", args.random)
	accuracy = 0

	temp_ = np.random.randint(int(args.skipable_layer)+1, size=1)
	if args.baseline == True or random == False:
		with torch.no_grad():
			num = 0
			length = 0
			start = time()
			for i, data in enumerate(test_dl, 0):	
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
