import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from utilityScript import standardizeData
import pdb

class NeuralNet(nn.Module):
	def __init__(self,input_size,hidden_layer,num_classes,hiddenActivation=torch.relu):
		super(NeuralNet, self).__init__()
		self.hidden = nn.ModuleList()
		self.hiddenActivation=hiddenActivation
		self.epochError=[]
		self.trMu,self.trSd = [],[]
		# Hidden layers
		if len(hidden_layer)==1:
			self.hidden.append(nn.Linear(input_size,hidden_layer[0]))
		elif(len(hidden_layer)>1):
			for i in range(len(hidden_layer)-1):
				if i==0:
					self.hidden.append(nn.Linear(input_size, hidden_layer[i]))
					self.hidden.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
				else:
					self.hidden.append(nn.Linear(hidden_layer[i],hidden_layer[i+1]))
		# Output layer
		self.out = nn.Linear(hidden_layer[-1], num_classes)

	def forward(self, x):
		# Feedforward
		for layer in self.hidden:
			x=self.hiddenActivation(layer(x))
		output= F.softmax(self.out(x), dim=1)
		return output

	def setHiddenWeight(self,W,b):
		for i in range(len(self.hidden)):
			self.hidden[i].bias.data=b[i].float()
			self.hidden[i].weight.data=W[i].float()

	def setOutputWeight(self,W,b):
		self.out.bias.data=b.float()
		self.out.weight.data=W.float()

	def fit(self,trData,trLabels,standardizeFlag,batchSize,optimizationFunc='Adam',learningRate=0.001,m=0,numEpochs=100,cudaDeviceId=0,verbose=False):
		
		if standardizeFlag:
		#standardize data
			mu,sd,trData = standardizeData(trData)
			self.trMu = mu
			self.trSd = sd

		# prepare data for fine tuning
		torchData = Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(trLabels.flatten().astype(int)))
		dataLoader = Data.DataLoader(dataset=torchData,batch_size=batchSize,shuffle=True)

		# Device configuration
		device = torch.device('cuda:'+str(cudaDeviceId))
		criterion = nn.CrossEntropyLoss()
		if optimizationFunc.upper()=='ADAM':
			optimizer = torch.optim.Adam(self.parameters(), lr=learningRate,amsgrad=True)
		elif optimizationFunc.upper()=='SGD':
			optimizer = torch.optim.SGD(self.parameters(), lr=learningRate,momentum=m)
		total_step = len(dataLoader)
		self.to(device)
		for epoch in range(numEpochs):
			error=[]
			for i, (sample, labels) in enumerate(dataLoader):  
				# Move tensors to the configured device
				sample = sample.to(device)
				labels = labels.to(device)

				# Forward pass
				outputs = self.forward(sample)
				loss = criterion(outputs, labels)
				error.append(loss.item())

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				#if (i+1) % 100 == 0:
				#	print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, numEpochs, i+1, total_step, loss.item()))
			self.epochError.append(np.mean(error))
			if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
				print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs,self.epochError[-1]))
	'''
	def predict(self,dataLoader,device=''):
		if device =='':
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		with torch.no_grad():
			correct = 0
			total = 0
			pVals=[]
			pLabels=[]
			for sample, labels in dataLoader:
				sample = sample.to(device)
				labels = labels.to(device)
				outputs = self.forward(sample)
				_, predicted = torch.max(outputs.data, 1)
				pLabels.append(predicted)
				pVals.append(outputs)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		pVals=np.vstack((pVals))
		pLabels=np.hstack((pLabels))
		print('Accuracy of the network : {} %'.format(100 * correct / total))
		return pVals,pLabels
		'''
	def predict(self,x):

		if len(self.trMu) != 0 and len(self.trSd) != 0:#standarization has been applied on training data so apply on test data
			x = standardizeData(x,self.trMu,self.trSd)
		x = torch.from_numpy(x).float().to('cpu')

		with torch.no_grad():
			fOut = self.forward(x)
		fOut = fOut.to('cpu').numpy()
		predictedVal = np.max(fOut,axis=1)
		predictedLabels = (np.argmax(fOut,axis=1)).reshape(-1,1)
		return predictedVal,predictedLabels
