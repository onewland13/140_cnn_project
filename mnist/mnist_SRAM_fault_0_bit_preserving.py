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
FAULT_RATES = [1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.000001] 

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
	all_exp_results = []

	for fault_rate in FAULT_RATES:
		# Load weights
		if os.path.isfile(MODEL_STORE_PATH + 'conv_net_model.ckpt'):
			model.load_state_dict(torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt'))
			print('Model loaded.')
		else:
			print('No model found.')
	
		exp_results = []

		# Test the model
		model.eval(testing=True, fault_rate=fault_rate)
		with torch.no_grad():
			correct = 0
			total = 0
			for images, labels in test_loader:
				outputs = model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				if total % 100 == 0:
					print('Predicted {}/{} correctly {}'.format(correct, total, ':)' if correct / total > 0.95 else ':('))
				exp_results.append(correct)

			print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
		all_exp_results.append(exp_results)

	# Save data to csv file
	df = pd.DataFrame(all_exp_results)
	df = df.transpose()
	df.columns = FAULT_RATES
	df.to_csv('./results/mnist_SRAM_fault_0_bit_preserving_RESULTS.csv')

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
		self.fault_rate = 0

	def eval(self, testing=False, fault_rate=0):
		self.testing = testing
		self.fault_rate = fault_rate
		super(ConvNet, self).eval()

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1) 
		out = self.drop_out(out)

		# Change weights at some probability
		if self.testing:
			weights_changed = []
			for i_vector in range(len(self.fc1.weight)):
				for i_w in range(len(self.fc1.weight[i_vector])):
					if random.random() < self.fault_rate:
						weight = self.fc1.weight[i_vector][i_w].data
						weights_changed.append((i_vector, i_w, weight))
						self.fc1.weight[i_vector][i_w] = self._zero_random_bit(weight)
			# evaluate weights
			out = self.fc1(out)
			# change weights back 
			for i_v, i_w, w in weights_changed:
				self.fc1.weight[i_v][i_w] = w

			weights_changed = []
			# change weights at some probability
			for i_vector in range(len(self.fc2.weight)):
				for i_w in range(len(self.fc2.weight[i_vector])):
					if random.random() < self.fault_rate:
						weight = self.fc2.weight[i_vector][i_w].data
						weights_changed.append((i_vector, i_w, weight))
						self.fc2.weight[i_vector][i_w] = self._zero_random_bit(weight)
			# evaluate weights
			out = self.fc2(out)
			# change weights back 
			for i_v, i_w, w in weights_changed:
				self.fc2.weight[i_v][i_w] = w
		else:
			out = self.fc1(out)
			out = self.fc2(out)

		return out

	# here our fault rate handling procedure is to turn the bit to 0 no matter what
	def _zero_random_bit(self, w):
		rep = struct.pack('>f', w)
		integer_w = struct.unpack('>l', rep)[0]

		bit_loc = random.randint(0, 30) # changing bit[31] seems to break this idk why
		mask = ~(1 << bit_loc)

		rep = struct.pack('>l', mask & integer_w)

		w_altered_float = struct.unpack('>f', rep)[0]
		return w_altered_float


if __name__ == '__main__': 
	sys.exit(main(sys.argv[1:]))