import mnist_loader
import network

net = network.Network([784,30,10])
net.SGD(mnist_loader.train_data,30,50,1.0,mnist_loader.test_data)