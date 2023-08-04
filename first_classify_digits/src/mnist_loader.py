import os
import urllib.request
import gzip
import numpy as np

def download_mnist_data(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for file in files:
        file_path = os.path.join(save_path, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(f"{url}/{file}", file_path)
            print("Download complete!")

def load_mnist_data(save_path):
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    data = {}
    for file in files:
        file_path = os.path.join(save_path, file)
        with gzip.open(file_path, 'rb') as f:
            if 'images' in file:
                data[file.split('-')[0]+'-images'] = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
            else:
                data[file.split('-')[0]+'-labels'] = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data

# 下载并保存MNIST数据集
url = 'http://yann.lecun.com/exdb/mnist'
save_path = '../data'
download_mnist_data(url, save_path)

# 加载训练和测试数据集
data = load_mnist_data(save_path)
train_data, test_data = list(zip(data['train-images'].reshape(60000,-1,1), data['train-labels'].reshape(60000,1))), list(zip(data['t10k-images'].reshape(10000,-1,1), data['t10k-labels'].reshape(10000,1)))


