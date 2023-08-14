import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=6)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=7)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=5)

    def forward(self,t):
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = F.relu(self.conv3(t))
        t = self.conv4(t)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]      

        activation = x
        activations = [x] # sigmoid输出
        zs = [] # 卷积层输出
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # z = wa + b
            zs.append(z)
            activation = sigmoid(z) # a = s(z)
            activations.append(activation)
        '''
        推导看本子
        '''
        # 为什么要先求一个delta，因为这些都是可复用的，相当于赋初值
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)