import pdb
from copy import deepcopy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
#from CustomizedLinear import CustomizedLinear
from OneToOneLinear import OneToOneLinear
from utilityDBN import createOutputAsCentroids,standardizeData

class FeatureSelectingSCE(nn.Module):
	def __init__(self, netConfig={}):
		super(FeatureSelectingSCE, self).__init__()
		if len(netConfig.keys()) != 0:		
			self.inputDim,self.outputDim = netConfig['inputDim'],netConfig['inputDim']
			self.hLayer,self.hLayerPost = deepcopy(netConfig['hL']),deepcopy(netConfig['hL'])			
			self.l1Penalty,self.l2Penalty = 0.0,0.0
			self.weightCutoff = None
			self.oActFunc,self.errorFunc = 'linear','MSE'
			self.linearDecoder=False
			
			self.splWs = None
			self.fWeight,self.fIndices = [],[]
			#pdb.set_trace()
			if 'weightCutoff' in netConfig.keys(): self.weightCutoff = netConfig['weightCutoff']
			if 'l1Penalty' in netConfig.keys(): self.l1Penalty = netConfig['l1Penalty']
			if 'l2Penalty' in netConfig.keys(): self.l2Penalty = netConfig['l2Penalty']
			if 'errorFunc' in netConfig.keys(): self.errorFunc = netConfig['errorFunc']
			if 'oActFunc' in netConfig.keys(): self.oActFunc = netConfig['oActFunc']
			if 'linearDecoder' in netConfig.keys(): self.linearDecoder=netConfig['linearDecoder']
			
			self.hActFunc,self.hActFuncPost=deepcopy(netConfig['hActFunc']),deepcopy(netConfig['hActFunc'])

		else:#for default set up
			self.hLayer=[2]
			self.oActFunc,self.errorFunc='linear','MSE'
			self.hActFunc,self.hActFuncPost='tanh','tanh'

		#internal variables
		self.epochError=[]
		self.trMu=[]
		self.trSd=[]
		self.tmpPreHActFunc=[]
		self.preTrW,self.preTrB = [],[]
		self.runningPostTr = False
		self.device = None
		self.preTrItr = None

	def initNet(self,input_size,hidden_layer):
		self.hidden=nn.ModuleList()
		# Hidden layers
		if not(self.runningPostTr):
			if len(hidden_layer)==1:
				self.hidden.append(nn.Linear(input_size,hidden_layer[0]))
				
			elif(len(hidden_layer)>1):
				for i in range(len(hidden_layer)-1):
					if i==0:
						self.hidden.append(nn.Linear(input_size, hidden_layer[i]))
						self.hidden.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
					else:
						self.hidden.append(nn.Linear(hidden_layer[i],hidden_layer[i+1]))
		else:
			#pdb.set_trace()
			for i in range(len(hidden_layer)-1):
				if i==0:
					self.hidden.append(OneToOneLinear(self.inputDim))
					self.hidden.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
				else:
					self.hidden.append(nn.Linear(hidden_layer[i],hidden_layer[i+1]))
		
		if not(self.runningPostTr):
			self.reset_parameters(hidden_layer)
		# Output layer
		self.out = nn.Linear(hidden_layer[-1], input_size)
		
	def reset_parameters(self,hidden_layer):

		hL = 0
		while True:
			if self.hActFunc[hL].upper() in ['SIGMOID','TANH']:
				torch.nn.init.xavier_uniform_(self.hidden[hL].weight)
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			elif self.hActFunc[hL].upper() == 'RELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			elif self.hActFunc[hL].upper() == 'LRELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='leaky_relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			if hL == len(hidden_layer)-1:
				break
			hL += 1

	def forwardPost(self, x):
		# Feedforward		
		for l in range(len(self.hidden)):
			if self.hActFuncPost[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='RELU':
				x = F.relu(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
			else:#default is linear				
				x = self.hidden[l](x)

		if self.oActFunc.upper()=='SIGMOID':
			return torch.sigmoid(self.out(x))
		else:
			return self.out(x)

	def forwardPre(self, x):
		# Feedforward
		for l in range(len(self.hidden)):
			if self.tmpPreHActFunc[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
			else:#default is linear
				x = self.hidden[l](x)
		if self.oActFunc.upper()=='SIGMOID':
			return torch.sigmoid(self.out(x))
		else:
			return self.out(x)

	def setHiddenWeight(self,W,b):
		for i in range(len(self.hidden)):
			self.hidden[i].bias.data=b[i].float()
			self.hidden[i].weight.data=W[i].float()

	def setOutputWeight(self,W,b):
		self.out.bias.data=b.float()
		self.out.weight.data=W.float()

	def returnTransformedData(self,x):
		fOut=[x]
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			for layer in self.hidden:
				fOut.append(self.hiddenActivation(layer(fOut[-1])))
			if self.output_activation.upper()=='SIGMOID':
				fOut.append(torch.sigmoid(self.out(fOut[-1])))
			else:
				fOut.append(self.out(fOut[-1]))
		return fOut[1:]#Ignore the original input

	def preTrain(self,dataLoader,optimizationFunc,learningRate,m,batchSize,numEpochs,verbose):

		#variable to store weight and bias temporarily
		preW=[]
		preB=[]

		#loop to do layer-wise pre-training
		for d in range(len(self.hLayer)):
			#pdb.set_trace()	
			#set the hidden layer structure
			hidden_layer=self.hLayer[:d+1]
			self.tmpPreHActFunc=self.hActFunc[:d+1]
			
			if verbose:
				print('Pre-training hidden layer:',d+1,'No. of units:',self.hLayer[d])

			#initialize the network weight and bias. Initialization will be done randomly.
			self.initNet(self.inputDim,hidden_layer)

			# reset weights and biases by pretrained layers.
			if d>0:				
				for l in range(d):
					# initialize the net
					self.hidden[l].weight.data=preW[l]
					self.hidden[l].bias.data=preB[l]
					self.hidden[l].weight.requires_grad=False
					self.hidden[l].bias.requires_grad=False

			# set loss function
			if self.errorFunc.upper() == 'CE':
				criterion = nn.CrossEntropyLoss()
			elif self.errorFunc.upper() == 'BCE':
				criterion = nn.BCELoss()
			elif self.errorFunc.upper() == 'MSE':
				criterion = nn.MSELoss()

			# set optimization function
			if optimizationFunc.upper()=='ADAM':
				optimizer = torch.optim.Adam(self.parameters(),lr=learningRate,amsgrad=True)
			elif optimizationFunc.upper()=='SGD':
				optimizer = torch.optim.SGD(self.parameters(),lr=learningRate,momentum=m)

			# Load the model to device
			self.to(self.device)

			# Start training
			tmpEpochError = []
			self.preTrItr = numEpochs
			for epoch in range(numEpochs):
				error=[]
				for i, (trInput, trOutput) in enumerate(dataLoader):  
					# Move tensors to the configured device
					trInput = trInput.to(self.device)
					trOutput = trOutput.to(self.device)

					# Forward pass
					outputs = self.forwardPre(trInput)
					loss = criterion(outputs, trOutput)
					
					error.append(loss.item())

					# Backward and optimize
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				tmpEpochError.append(np.mean(error))
				if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
					print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs, tmpEpochError[-1]))
			
			# store pre-trained weight and bias
			preW.append(self.hidden[d].weight)
			preB.append(self.hidden[d].bias)

		#now set requires_grad =True for all the layers
		for l in range(len(hidden_layer)):
			self.hidden[l].weight.requires_grad=True			
			self.hidden[l].bias.requires_grad=True
			self.preTrW.append(deepcopy(self.hidden[l].weight.data))
			self.preTrB.append(deepcopy(self.hidden[l].bias.data))
			
		self.out.weight.requires_grad=True
		self.out.bias.requires_grad=True
		self.preTrW.append(deepcopy(self.out.weight.data))
		self.preTrB.append(deepcopy(self.out.bias.data))
		
		if verbose:
			print('Pre-training is done.')

	def postTrain(self,dataLoader,optimizationFunc,learningRate,m,batchSize,numEpochs,verbose):

		# set device
		#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		#pdb.set_trace()
		# add 'SPL' between the input and the first hidden layer
		self.runningPostTr = True
		self.hLayerPost = np.insert(self.hLayerPost,0,self.inputDim)
		self.hActFuncPost.insert(0,'SPL')
		self.initNet(self.inputDim,self.hLayerPost)
		
		# use the preTrW and preTrB to set the weight and bias.
		# note weights of SPL are initialized to 1. There is no bias for this layer.
		#pdb.set_trace()
		#for l in np.arange(2,len(self.hLayerPost)):
		for l in np.arange(1,len(self.hLayerPost)):
			self.hidden[l].weight.data = deepcopy(self.preTrW[l-1])
			self.hidden[l].bias.data = deepcopy(self.preTrB[l-1])
		self.out.weight.data = deepcopy(self.preTrW[-1])
		self.out.bias.data = deepcopy(self.preTrB[-1])
		
		# set loss function
		if self.errorFunc.upper() == 'CE':
			criterion = nn.CrossEntropyLoss()
		elif self.errorFunc.upper() == 'BCE':
			criterion = nn.BCELoss()
		elif self.errorFunc.upper() == 'MSE':
			criterion = nn.MSELoss()

		# set optimization function
		if optimizationFunc.upper()=='ADAM':
			optimizer = torch.optim.Adam(self.parameters(),lr=learningRate,amsgrad=True)
		elif optimizationFunc.upper()=='SGD':
			optimizer = torch.optim.SGD(self.parameters(),lr=learningRate,momentum=m)

		# Load the model to device
		self.to(self.device)
		
		# do training to adjust the weight before applying L1 penalty.
		# The weights of SPL is set to 1. This training will adjust the weights of SPL along with other layers.
		if verbose:
			print('Readjusting the weights of SPL. L1 penalty will not be applied now.')

		tmpEpochs = self.preTrItr
		tmpEpochError = []
		for epoch in range(tmpEpochs):
			error=[]
			for i, (trInput, trOutput) in enumerate(dataLoader):  
				# Move tensors to the configured device
				trInput = trInput.to(self.device)
				trOutput = trOutput.to(self.device)

				# Forward pass
				outputs = self.forwardPost(trInput)
				loss = criterion(outputs, trOutput)
				error.append(loss.item())

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			tmpEpochError.append(np.mean(error))
			if verbose and ((epoch+1) % (tmpEpochs*0.1)) == 0:
				print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, tmpEpochs, tmpEpochError[-1]))		

		#pdb.set_trace()
		# Start training
		if verbose:
			print('Training network:',self.inputDim,'-->',self.hLayerPost,'-->',self.inputDim)

		if self.weightCutoff != None:
			epoch = 1
			error = []
			while True:
				for i, (trInput, trOutput) in enumerate(dataLoader):  
					# Move tensors to the configured device
					trInput = trInput.to(self.device)
					trOutput = trOutput.to(self.device)

					# Forward pass
					outputs = self.forwardPost(trInput)
					loss = criterion(outputs, trOutput)

					# Check for regularization
					if self.l1Penalty != 0 or self.l2Penalty != 0:
						l1RegLoss,l2RegLoss = torch.tensor([0.0],requires_grad=True).to(self.device), torch.tensor([0.0],requires_grad=True).to(self.device)
						if self.l1Penalty != 0 and self.l2Penalty == 0:
							l1RegLoss = torch.norm(self.hidden[0].weight, p=1)
							loss = loss + self.l1Penalty * l1RegLoss
						elif self.l1Penalty == 0 and self.l2Penalty != 0:
							l2RegLoss = torch.norm(self.hidden[0].weight, p=2)
							loss = loss + 0.5 * self.l2Penalty * l2RegLoss
						elif self.l1Penalty != 0 and self.l2Penalty != 0:
							l2RegLoss = torch.norm(self.hidden[0].weight, p=2)
							l1RegLoss = torch.norm(self.hidden[0].weight, p=1)
							loss = loss + self.l1Penalty * l1RegLoss + 0.5 * self.l2Penalty * l2RegLoss
					
					error.append(loss.item())

					# Backward and optimize
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				self.epochError.append(np.mean(error))
				epoch += 1

				# now check the L1 norm of the SPL to check stopping criteria.
				if torch.sum(torch.abs(self.hidden[0].weight.data)).item() <= self.weightCutoff:
					if verbose:
						print('L1 norm of SPL has reached the cut-off. Stopping training. Epochs elapsed:',epoch)
					break
				else:					
					if verbose and ((epoch) % 100) == 0:
						print ('Epoch {}, Loss: {:.6f}, L1 Norm of SPL: {:.6f}'.format(epoch, self.epochError[-1],torch.sum(torch.abs(self.hidden[0].weight.data)).item()))

		else:
			for epoch in range(numEpochs):
				error = []
				for i, (trInput, trOutput) in enumerate(dataLoader):  
					# Move tensors to the configured device
					trInput = trInput.to(self.device)
					trOutput = trOutput.to(self.device)

					# Forward pass
					outputs = self.forwardPost(trInput)
					loss = criterion(outputs, trOutput)

					# Check for regularization
					if self.l1Penalty != 0 or self.l2Penalty != 0:
						l1RegLoss,l2RegLoss = torch.tensor([0.0],requires_grad=True).to(self.device), torch.tensor([0.0],requires_grad=True).to(self.device)
						if self.l1Penalty != 0 and self.l2Penalty == 0:
							l1RegLoss = torch.norm(self.hidden[0].weight, p=1)
							loss = loss + self.l1Penalty * l1RegLoss
						elif self.l1Penalty == 0 and self.l2Penalty != 0:
							l2RegLoss = torch.norm(self.hidden[0].weight, p=2)
							loss = loss + 0.5 * self.l2Penalty * l2RegLoss
						elif self.l1Penalty != 0 and self.l2Penalty != 0:
							l2RegLoss = torch.norm(self.hidden[0].weight, p=2)
							l1RegLoss = torch.norm(self.hidden[0].weight, p=1)
							loss = loss + self.l1Penalty * l1RegLoss + 0.5 * self.l2Penalty * l2RegLoss
					
					error.append(loss.item())

					# Backward and optimize
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				self.epochError.append(np.mean(error))
				if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
					print ('Epoch [{}/{}], Loss: {:.6f}, L1 Norm of SPL: {:.6f}'.format(epoch+1, numEpochs, self.epochError[-1],torch.sum(torch.abs(self.hidden[0].weight.data)).item()))

		self.splWs = self.hidden[0].weight.data.to('cpu')
		self.fWeight,self.fIndices = torch.sort(torch.abs(self.splWs),descending=True)
		self.fWeight = self.fWeight.to('cpu').numpy()
		self.fIndices = self.fIndices.to('cpu').numpy()
		
	def fit(self,trData,trLabels,valData=[],valLabels=[],preTraining=True,optimizationFunc='Adam',learningRate=0.001,m=0,
			miniBatchSize=100,numEpochsPreTrn=25,numEpochsPostTrn=100,standardizeFlag=True,cudaDeviceId=0,verbose=True):

		# set device
		self.device = torch.device('cuda:'+str(cudaDeviceId))

		#standardize data
		if standardizeFlag:
			mu,sd,trData = standardizeData(trData)
			#_,_,target = standardizeData(target)
			self.trMu=mu
			self.trSd=sd

		#create target: centroid for each class
		target=createOutputAsCentroids(trData,trLabels)
		#pdb.set_trace()

		#create target of validation data for early stopping
		#if len(valData) != 0:
		#	valData = standardizeData(valData,self.trMu,self.trSd)
		

		#Prepare data for torch
		trDataTorch=Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(target).float())
		dataLoader=Data.DataLoader(dataset=trDataTorch,batch_size=miniBatchSize,shuffle=True)

		#layer-wise pre-training
		#pdb.set_trace()
		if preTraining:
			self.preTrain(dataLoader,optimizationFunc,learningRate,m,miniBatchSize,numEpochsPreTrn,verbose)
		else:
			#initialize the network weight and bias
			self.initNet(self.inputDim,self.hLayerPost)
		#post training
		self.postTrain(dataLoader,optimizationFunc,learningRate,m,miniBatchSize,numEpochsPostTrn,verbose)
		
	def predict(self,x):
		if len(self.trMu) != 0 and len(self.trSd) != 0:#standarization has been applied on training data so apply on test data
			x = standardizeData(x,self.trMu,self.trSd)
		#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		x=torch.from_numpy(x).float().to(device)
		fOut=[x]
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			for l in range(len(self.hidden)):
				if self.hActFuncPost[l].upper()=='SIGMOID':
					fOut.append(torch.sigmoid(self.hidden[l](fOut[-1])))
				elif self.hActFuncPost[l].upper()=='TANH':
					fOut.append(torch.tanh(self.hidden[l](fOut[-1])))
				elif self.hActFuncPost[l].upper()=='RELU':
					fOut.append(torch.relu(self.hidden[l](fOut[-1])))
				else:#default is linear				
					fOut.append(self.hidden[l](fOut[-1]))

			if self.oActFunc.upper()=='SIGMOID':
				fOut.append(torch.sigmoid(self.out(fOut[-1])))
			else:
				fOut.append(self.out(fOut[-1]))

		return fOut
