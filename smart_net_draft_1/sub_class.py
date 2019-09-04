import torch
import torch.nn as nn
from utils import conv_layer, get_param_size, rnn
import numpy as np


class sub_draft(nn.Module):
	def __init__(self,input_dim,input_size,hidden_size,output_size):
		super(sub_draft, self).__init__()

		self.input_dim = input_dim
		if input_ = True: self.layer_1 = conv_layer(self.input_dim, 32, kernel_size=3, stride = (2,2), padding=1) #[112,112] 
		else: self.layer_1 = conv_layer(self.input_dim, 32, kernel_size=3, stride = (1,1), padding=1) #[112,112]
		self.layer_2 = conv_layer(32, 128, kernel_size=3, stride = (2,2), padding=1) # [56,56]
		self.layer_3 = conv_layer(128, 512, kernel_size=3, stride = (2,2), padding=1) #[28,28]
		self.layer_4 = conv_layer(512, 1024, kernel_size=3, stride = (2,2), padding=1) #[14,14]

		self.fc1 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
								 nn.ReLU(True))
		self.fc1_drop = nn.Dropout(0.5)
		self.last_layer = rnn(input_size,hidden_size,output_size)


	def forward(self, current,hidden):

		x = self.layer_1(current) # [112,112]
		x = self.layer_2(x) # [56,56]
		x = self.layer_3(x) # [28,28]
		x = self.layer_4(x) # [14,14]

		m =  nn.AvgPool2d((x.size()[2],x.size()[3]),stride = 1)
		x = m(x)
		x = x.view(-1, self.num_flat_features(x))
		x = self.fc1(x)
		x = self.fc1_drop(x)
		print(x)
		out, hn = self.last_layer()

		# print(x)
		return out, hn

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
		    num_features *= s
		return num_features
