import sys, os
import torch
import torchvision
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from sklearn import preprocessing
import csv
import random
import numpy as np
import struct
import pandas as pd

# Most of this is from:
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/conv_net_py_torch.py

# HYPERPARAMETERS
NUM_EPOCHS = 6
NUM_CLASSES = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.001
MNIST_DATA_PATH = './mnist_data/'
MODEL_STORE_PATH = './mnist_model/'
TESTING = False

# EXPERIMENT NUMBERS
QUANTIZE_BITS_CUT = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]


def main(a):
    # transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=MNIST_DATA_PATH, train=True, transform=trans)
    test_dataset = torchvision.datasets.MNIST(root=MNIST_DATA_PATH, train=False, transform=trans)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvNet()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Experiment
    exp_results = []

    for bits in QUANTIZE_BITS_CUT:
        # Load weights
        if os.path.isfile(MODEL_STORE_PATH + 'conv_net_model.ckpt'):
            model.load_state_dict(torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt'))
            print('Model loaded.')
        else:
            print('No model found.')

        # Test the model
        model.eval(testing=True, bits=bits)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if total % 100 == 0:
                    print(
                        'Predicted {}/{} correctly {}'.format(correct, total, ':)' if correct / total > 0.95 else ':('))

            exp_results.append(correct)
            print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # Save data to csv file
    df = pd.DataFrame([exp_results])
    df.columns = QUANTIZE_BITS_CUT
    df.to_csv('./results/mnist_quantize_RESULTS.csv')


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.testing = False
        self.bits = -1

    def eval(self, testing=False, bits=-1):
        self.testing = testing
        self.bits = bits

        # quantize all weights
        if self.bits != -1:
            for i_v in range(len(self.fc1.weight)):
                for i_w in range(len(self.fc1.weight[i_v])):
                    s = struct.pack('>f', self.fc1.weight[i_v][i_w].data)
                    i = struct.unpack('>l', s)[0]
                    i = i >> bits
                    i = i << bits
                    i = struct.pack('>l', i)
                    self.fc1.weight[i_v][i_w] = struct.unpack('>f', i)[0]
            for i_v in range(len(self.fc2.weight)):
                for i_w in range(len(self.fc2.weight[i_v])):
                    s = struct.pack('>f', self.fc2.weight[i_v][i_w].data)
                    i = struct.unpack('>l', s)[0]
                    i = i >> bits
                    i = i << bits
                    i = struct.pack('>l', i)
                    self.fc2.weight[i_v][i_w] = struct.unpack('>f', i)[0]

        super(ConvNet, self).eval()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))