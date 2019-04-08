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
FAULT_RATE = 0.01

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

	if len(a) > 0 and a[0] == 'load':
		if os.path.isfile(MODEL_STORE_PATH + 'conv_net_model.ckpt'):
			model.load_state_dict(torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt'))
		else:
			print('No model found.')
	else:
		# Train the model
		total_step = len(train_loader)
		loss_list = []
		acc_list = []
		for epoch in range(NUM_EPOCHS):
			for i, (images, labels) in enumerate(train_loader):
				# Run the forward pass
				outputs = model(images)
				loss = criterion(outputs, labels)
				loss_list.append(loss.item())

				# Backpropagate and perform Adam optimization
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Track the accuracy
				total = labels.size(0)
				_, predicted = torch.max(outputs.data, 1)
				correct = (predicted == labels).sum().item()
				acc_list.append(correct / total)

				if (i + 1) % 100 == 0:
					print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
						  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item(), (correct / total) * 100))

	# Test the model
	model.eval(testing=True)
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

		print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

	# Save the model and plot
	torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

	# p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
	# p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
	# p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
	# p.line(np.arange(len(loss_list)), loss_list)
	# p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
	# show(p)

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

	def eval(self, testing=False):
		self.testing = testing
		super(ConvNet, self).eval()

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1) 
		out = self.drop_out(out)

		# change weights at some probability
		if self.testing:
			for i_vector in range(len(self.fc1.weight)):
				for i_w in range(len(self.fc1.weight[i_vector])):
					if random.random() < FAULT_RATE:
						# SRAM Fault handling strategy goes here

						# In this example a bit just flips to a 0
						weight = self.fc1.weight[i_vector][i_w].data
						self.fc1.weight[i_vector][i_w] = self._zero_random_bit(weight)

						# This would just 0 out the whole weight
						# self.fc1.weight[i_vector][i_w] = 0.0

			# change weights at some probability
			for i_vector in range(len(self.fc2.weight)):
				for i_w in range(len(self.fc2.weight[i_vector])):
					if random.random() < FAULT_RATE:
						# SRAM Fault handling strategy goes here

						# In this example a bit just flips to a 0
						weight = self.fc2.weight[i_vector][i_w].data
						self.fc2.weight[i_vector][i_w] = self._zero_random_bit(weight)

						# This would just 0 out the whole weight
						# self.fc2.weight[i_vector][i_w] = 0.0

			# create modified layers
			# fc1_temp = nn.Linear(7 * 7 * 64, 1000)
			# fc2_temp = nn.Linear(1000, 10)
			# fc1_temp.weight = Parameter(torch.Tensor(fc1_weights))
			# fc2_temp.weight = Parameter(torch.Tensor(fc2_weights))
			
			# predict accordingly
			out = self.fc1(out)
			out = self.fc2(out)

			# replace fc1 and fc2 with modified weights (optional)
			# self.fc1 = fc1_temp
			# self.fc2 = fc2_temp
		else: 
			out = self.fc1(out)
			out = self.fc2(out)

		return out

	def _zero_random_bit(self, w):
		# print('FLOAT BEFORE:', w)
		rep = struct.pack('>f', w)
		integer_w = struct.unpack('>l', rep)[0]
		# print('INT EQUIVALENT:', '{0:b}'.format(integer_w))

		bit_loc = random.randint(0, 30) # changing bit 31 seems to break this idk why
		mask = ~(1 << bit_loc)

		# print('INT MODIFIED:  ', '{0:b}'.format(integer_w & mask))
		rep = struct.pack('>l', mask & integer_w)

		w_altered_float = struct.unpack('>f', rep)[0]
		# print('FLOAT AFTER:', w_altered_float)
		return w_altered_float


if __name__ == '__main__': 
	sys.exit(main(sys.argv[1:]))