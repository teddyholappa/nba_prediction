import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#final_merge = pd.read_pickle("./merged.pkl")
final_merge = pd.read_pickle("./merged_5_day.pkl")
labels = final_merge['WL']
final_merge = final_merge.drop(columns=['WL','MATCHUP'])
#Normalizing the dataset
final_merge = (final_merge-final_merge.mean())/(final_merge.std())
train_xs = torch.tensor(final_merge[0:8000].astype(float).values)
train_labels = labels[0:8000]
train_labels = train_labels.mask(train_labels == 'W',1)
train_labels = train_labels.mask(train_labels == 'L',0)
train_labels = torch.tensor(train_labels[0:8000].astype(int).values).unsqueeze(1)
test_xs = torch.tensor(final_merge[8000:].astype(float).values)
test_labels = labels[8000:]
test_labels = test_labels.mask(test_labels == 'W',1)
test_labels = test_labels.mask(test_labels == 'L',0)
test_labels = torch.tensor(test_labels.astype(int).values)


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(40,30)
		self.fc2 = nn.Linear(30,10)
		self.fc3 = nn.Linear(10,10)
		self.fc4 = nn.Linear(10,1)
		self.dropout = nn.Dropout(0.25)

	def forward(self,x):
		x = F.relu(self.fc1(x.float()))
		x = self.dropout(x)
		x = F.relu(self.fc2(x))
		x = self.dropout(x)
		x = F.relu(self.fc3(x))
		x = torch.sigmoid(self.fc4(x))
		return x

class LogisticRegression(nn.Module):
	def __init__(self):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(40,1)
	def forward(self,x):
		pred = F.sigmoid(self.linear(x.float()))
		return pred

net = Net()
#net = LogisticRegression()

#criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=.9)
for epoch in range(10):
	running_loss = 0.0
	for i in range(train_xs.shape[0]):
		optimizer.zero_grad()
		outputs = net(train_xs)
		loss = criterion(outputs, train_labels.float())
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print("Training over")

print("--Testing--")
output = net(test_xs)
predicted = (output>0.5).float()
test_labels = test_labels.unsqueeze(1)
correct = (predicted == test_labels).float()
print("Correct: ")
print(correct)
correct_sum = correct.sum()
print("Correct_sum: ", correct_sum)
print("Test accuracy")
print(correct_sum / test_xs.shape[0])



#We can probably change the features here, eliminate linearly dependent features!




