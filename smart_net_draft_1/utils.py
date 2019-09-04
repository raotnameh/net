#supporting classes and fucntions
#for any queries contact, raotnameh@gmail.com or +91 8285207072

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

class rnn_layer(nn.Module):
    def __init__(self,input_size,hidden_size=512,num_layers=1,dropout_=0.5,bidirectional=False):
        super(rnn_layer, self).__init__()

        self.rnn = nn.RNN(input_size,hidden_size,num_layers,dropout=dropout_,bidirectional=bidirectional) 

    def forward(self,input,h0):
        out,hn = self.rnn(input,h0)
        return out,hn



#input shape: seq_len, batch, input_size
#h_0 shape num_layers * num_directions, batch, hidden_size

#sequence length means = length of the input i.e. to roll into how many time steps.
#input size = list of list of input


class rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

