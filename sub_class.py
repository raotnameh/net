import torch
import torch.nn as nn
from utils import conv_layer, get_param_size
import numpy as np

class sub_draft(nn.Module):
	def __init__(self):
		super(sub_draft, self).__init__()

		self.layer_1 = conv_layer(259, 512, kernel_size=3, stride = (2,2), padding=1) #[56,56]
		self.layer_1_1 = conv_layer(512, 128, kernel_size=1, stride = (2,2), padding=0) # [28,28]
		self.layer_2 = conv_layer(128, 512, kernel_size=3, stride = (2,2), padding=1) #[14,14]
		self.layer_2_1 = conv_layer(512, 128, kernel_size=1, stride = (1,1), padding=0) #[14,14]
		self.layer_3 = conv_layer(128, 512, kernel_size=3, stride = (2,2), padding=1) #[7,7]

		self.fc1 = nn.Sequential(nn.Linear(in_features=512, out_features=512),
								 nn.ReLU(True))
		self.fc1_drop = nn.Dropout(0.5)
		self.fc2 = nn.Linear(512,1)
		self.sigmoid = nn.Sigmoid()


	def forward(self, past, current):
		# current shape : [batch_size, 256,112,112]
		# past shape : [batch_size, 3,112,112]
		x = torch.cat((past,current),1)#[112,112]
		# x shape : [batch_size, 259,112,112]
		x = self.layer_1(x) # [56,56]
		x = self.layer_1_1(x) # [28,28]
		x = self.layer_2(x) # [14,14]
		x = self.layer_2_1(x) # [14,14]
		x = self.layer_3(x) # [7,7]

		m =  nn.AvgPool2d((x.size()[2],x.size()[3]),stride = 1)
		x = m(x)
		x = x.view(-1, self.num_flat_features(x))
		x = self.fc1(x)
		x = self.fc1_drop(x)
		x = self.fc2(x)
		x = self.sigmoid(x)

		# print(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
		    num_features *= s
		return num_features

