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
	model.eval()
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

	# Save the model
	torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

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

	def eval(self):
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