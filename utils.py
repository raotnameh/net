import torch
import torch.nn as nn

def get_param_size(model):
        params = 0
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
