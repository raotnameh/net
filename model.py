import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import conv_layer, get_param_size

from collections import OrderedDict


class feature_b(nn.Module): #base feature layer
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
		self.layerb6 = conv_layer(256, 512, kernel_size=3, stride = (2,2),padding=1)
		self.drop_layer3 = nn.Dropout(0.1)
		#input 28*28
		
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

class feature_r(nn.Module): #recursive feature layer
	def __init__(self):
		super(feature_r,self).__init__()

		#input 28*28
		self.layerb1 = conv_layer(512, 512, kernel_size=3, stride = (1,1), padding=1)
		self.layerb2 = conv_layer(512, 512, kernel_size=3, stride = (1,1), padding=1)
		self.drop_layer1 = nn.Dropout(0.1)
		#input 28*28
		self.layerb3 = conv_layer(512, 512, kernel_size=3, stride = (1,1), padding=1)
		self.layerb4 = conv_layer(512, 512, kernel_size=3, stride = (1,1), padding=1)
		self.drop_layer2 = nn.Dropout(0.1)
		#input 28*28

		
	def forward(self, input):
		x = self.layerb1(input)
		x = self.layerb2(x)
		x = self.drop_layer1(x)

		x = self.layerb3(x)
		x = self.layerb4(x)
		x = self.drop_layer2(x)

		return x


class decision_r(nn.Module): #recursice decision layer
	def __init__(self):
		super(decision_r,self).__init__()

		#input 28*28
		self.layerb1 = conv_layer(512, 1024, kernel_size=3, stride = (2,2), padding=1)
		#input 14*14
		self.layerb2 = conv_layer(1024, 4096, kernel_size=3, stride = (2,2), padding=1)
		#input 7*7
		
	def forward(self, input):
		x = self.layerb1(input)
		x = self.layerb2(x)

		m =  nn.AvgPool2d((x.size()[2],x.size()[3]),stride = 1)
		x = m(x)
		x = x.view(-1, self.num_flat_features(x))
		
		return x #4096 features

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
		    num_features *= s
		return num_features

class gru_layer(nn.Module): #gru layer
	def __init__(self,in_,hidden_, batch_first,classes):
		super(gru_layer,self).__init__()

		self.hidden = hidden_
		self.gru = nn.GRU(in_, self.hidden, num_layers=1, bias=True, batch_first=batch_first, dropout=0, bidirectional=False)
		self.linear1 = nn.Linear(self.hidden, classes, bias=False)
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, input,hidden):
		out, _ = self.gru(input, hidden)
		out = self.linear1(out.view(out.shape[1],out.shape[0],-1))
		out = self.sigmoid(out)
		
		return out #output classes

	def init_hidden(self, batch_size):
		return torch.zeros(1,batch_size, self.hidden,dtype=torch.float32)