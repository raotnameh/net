import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import conv_layer, get_param_size

from collections import OrderedDict


class feature_b(nn.Module):
	def __init__(self):
		super(feature_b,self).__init__()

		#input 224*224
		self.layerb1 = conv_layer(3, 64, kernel_size=3, stride = (1,1), padding=1)
		self.layerb2 = conv_layer(64, 64, kernel_size=3, stride = (2,2), padding=1)
		self.drop_layer1 = nn.Dropout(0.1)
		#input 112*112
		self.layerb3 = conv_layer(64, 128, kernel_size=3, stride = (1,1), padding=1)
		self.layerb4 = conv_layer(128, 128, kernel_size=3, stride = (2,2), padding=1)
		self.drop_layer2 = nn.Dropout(0.1)
		#input 56*56
		self.layerb5 = conv_layer(128, 256, kernel_size=3, stride = (1,1), padding=1)
		self.layerb6 = conv_layer(256, 512, kernel_size=3, stride = (1,1),padding=1)
		self.drop_layer3 = nn.Dropout(0.1)
		#input 56*56
		
	def forward(self,input):
		x = self.layerb1(input)
		x = self.layerb2(x)
		x = self.drop_layer1(x)

		x = self.layerb3(x)
		x = self.layerb4(x)
		x = self.drop_layer2(x)
		
		x = self.layerb5(x)
		x = self.layerb6(x)
		x = self.drop_layer3(x)

		return x

class feature_r(nn.Module):
	def __init__(self):
		super(feature_r,self).__init__()

		#input 56*56
		self.layerb1 = conv_layer(512, 512, kernel_size=3, stride = (1,1), padding=1)
		self.layerb2 = conv_layer(512, 512, kernel_size=3, stride = (1,1), padding=1)
		self.drop_layer1 = nn.Dropout(0.1)
		#input 56*56
		self.layerb3 = conv_layer(512, 512, kernel_size=3, stride = (1,1), padding=1)
		self.layerb4 = conv_layer(512, 512, kernel_size=3, stride = (1,1), padding=1)
		self.drop_layer2 = nn.Dropout(0.1)
		#input 56*56

		
	def forward(self, input):
		x = self.layerb1(input)
		x = self.layerb2(x)
		x = self.drop_layer1(x)

		x = self.layerb3(x)
		x = self.layerb4(x)
		x = self.drop_layer2(x)

		return x

class decision(nn.Module):
	def __init__(self,out_classes):
		super(decision,self).__init__()

		#input 56*56
		self.layerb1 = conv_layer(512, 1024, kernel_size=3, stride = (2,2), padding=1)
		#input 28*28
		self.layerb2 = conv_layer(1024, 1024, kernel_size=3, stride = (2,2), padding=1,max_pool = True)
		#input 7*7

		self.drop_layer = nn.Dropout(0.5)
		self.fc1 = nn.Linear(1024, 2048)
		self.fc2 = nn.Linear(2048, 2048)
		self.fc3 = nn.Linear(2048, out_classes)
		
	def forward(self, input):
		x = self.layerb1(input)
		x = self.layerb2(x)

		m =  nn.AvgPool2d((x.size()[2],x.size()[3]),stride = 1)
		x = m(x)
		x = x.view(-1, self.num_flat_features(x))
		
		x = self.fc1(x)
		x = self.drop_layer(x)
		x = self.fc2(x)
		x = self.drop_layer(x)
		x = self.fc3(x)


		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
		    num_features *= s
		return num_features