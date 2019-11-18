import torch
import torch.nn as nn
import torch.optim as optim

def get_param_size(z):
        params = 0
        for model in z:
	        for p in model.parameters():
	            tmp = 1
	            for x in p.size():
	                tmp *= x
	            params += tmp
        return params

def plot(input_,num):
	fig=plt.figure(figsize=(16, 16))
	for i in range(1, num + 1):
	    fig.add_subplot(num/2, num/2, i)
	    plt.imshow(input_[0,i-1])

	plt.savefig('books_read.png')
	plt.show()

def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

        
class conv_layer(nn.Module):
	def __init__(self, in_, out_, kernel_size, stride, padding, max_pool = False):
		super(conv_layer,self).__init__()

		self.in_ = in_
		self.out_ = out_
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.max_pool = max_pool

		if max_pool:
		    self.conv = nn.Sequential(nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding),
		                                    nn.BatchNorm2d(out_),
		                                    nn.ReLU6(True),
		                                    nn.MaxPool2d(2))

		else:
		    self.conv = nn.Sequential(nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding),
		                                    nn.BatchNorm2d(out_),
		                                    nn.ReLU6(True))


	def forward(self, x):
		x = self.conv(x) 
		return x

def switch(out, less_than = 0.75,target = None,set = "val"):
	soft = nn.functional.softmax(out,dim = 1)
	max_, _ = torch.max(soft,1)
	sum_ = max_.clone().detach().mean().item() # surety
	with open("prob.txt" ,"a") as f:
		f.write(str(sum_)+"\n")

	if set == "train":
		num = 0.0
		out , predicted = torch.max(out, 1)
		for k in range(len(target)):
			if target[k] == predicted[k].item():
				num = num + 1

		if  sum_*0.95 >= less_than  and num/len(target) >= less_than : # surety and accuracy
			return False
		elif sum_*0.95 <= less_than  and num/len(target) <= less_than :
			return True
		else: return False

	else:
		if  sum_ <= less_than :
			return True
		else: return False 
