import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utilityDBN import getApplicationData
from FeatureSelectingSCEPyTorch import FeatureSelectingSCE
from utilityBiology import returnImpFeaturesElbow
import pickle
from sklearn.metrics import accuracy_score
from simpleANNClassifierPyTorch import *

def load_ALLAML_Data(flag = 'Trn',partition=0):
	
	#load data file
	part = 'Partition'+str(partition)
	if flag.upper() == 'TRN':
		lData = pickle.load(open('./Data/ALLAML/'+part+'/trnData.p','rb'))
	elif flag.upper() == 'TST':
		lData = pickle.load(open('./Data/ALLAML/'+part+'/tstData.p','rb'))
	
	return lData

def runSCE(trData,trLabels,l1Penalty,lrnRate,num_epochs_pre,num_epochs_post,standardizeFlag,preTrFlag):
	
	flag = False
	while not(flag):
	
		dict2 = {}
		dict2['inputL'] = np.shape(trData)[1]
		dict2['outputL'] = np.shape(trData)[1]
		dict2['hL'] = [np.shape(trData)[1],100]
		dict2['actFunc'] = ['SPL','tanh']
		dict2['outputActivation'] = 'linear'
		dict2['l1Penalty'] = l1Penalty
		dict2['nItrPre'] = 10
		dict2['nItrPost'] = 50
		dict2['errorFunc']='MSE'

		#initiate an object of the model and call it's training method 
		model = SparseCE(dict2)
		model.fit(trData,trLabels)
		featureList,featuresW = returnImpFeaturesElbow(model.splWs)
		
		model = FeatureSelectingSCE(dict2)
		model.fit(trData,trLabels,preTrFlag,optimizationFunc='Adam',learningRate=learning_rate,m=momentum,miniBatchSize=miniBatch_size,
		numEpochsPreTrn=num_epochs_pre,numEpochsPostTrn=num_epochs_post,standardizeFlag=standardizeFlag,verbose=True)

		if len(featureList) < 50:
			print('Too few feature were selecred. Discarding this run')
			continue
		else:
			flag = True
		#pdb.set_trace()
		print('L1 Penalty',l1Penalty,'CE cost:',np.round(model.trErrorTrace[-1],2),'L1 Cost:',np.round(np.sum(np.abs(model.splWs)),2),'No. of selected features',len(featureList))
		
		W = -1*np.sort(-1*np.abs(model.splWs))
		#plt.plot(W,label='L1 Penalty '+str(l1Penalty))
		#plt.xlabel('feature indices',size=15)
		#plt.ylabel('feature weight',size=15)
		#plt.legend(fontsize=15)
		#plt.show()
	return W,featureList,featuresW
	
def classifyALLAMLData(partition,featureSet,fCntList,gpuId):
	
	accuracyList = []
	display = True
	for feaCnt in fCntList:
		#load ALLAML data
		trnSet = load_ALLAML_Data('Trn',partition)
		tstSet = load_ALLAML_Data('Tst',partition)
		trData,trLabels = trnSet[:,:-1],trnSet[:,-1]
		tstData,tstLabels = tstSet[:,:-1],tstSet[:,-1]
		fea = featureSet[:feaCnt]
	
		#use the selected features
		trData,tstData = trData[:,fea],tstData[:,fea]
		if display:
			print('No. of training samples',len(trData),' No of test samples',len(tstData))
			display = False
		
		nClass = len(np.unique(trLabels))
		allACC = []
		for i in range(10):
			ann = NeuralNet(trData.shape[1], [500] , nClass)
			ann.fit(trData,trLabels,standardizeFlag=True,batchSize=200,optimizationFunc='Adam',learningRate=0.001, numEpochs=25,cudaDeviceId=gpuId)
			ann = ann.to('cpu')
			tstPredProb,tstPredLabel = ann.predict(tstData)
			accuracy = 100 * accuracy_score(tstLabels.flatten(), tstPredLabel)
			allACC.append(accuracy)
			#print('Accuracy after iteration',i+1,np.round(accuracy,2))
		allACC = np.hstack((allACC))
		#print('No. of features:',trData.shape[1],'.Average Accuracy:',np.round(np.mean(allACC),2),'+/',np.round(np.std(allACC),2))
		accuracyList.append(np.round(np.mean(allACC),2))
		#pdb.set_trace()
		print('Data partition:',partition,'Accuracy using',feaCnt,'no. of features:',np.round(np.mean(allACC),2),'+/',np.round(np.std(allACC),2))
	return accuracyList


if __name__ == "__main__":

	# hyper-parameters for Adam optimizer
	num_epochs_pre = 10
	num_epochs_post = 125
	#for high dimensional dataset use a single minibatch. I did this by setting the minibatch size large.
	miniBatch_size = 50000
	learning_rate = 0.01
	gpuId = 1
	pp = 1 #possible values 0,1,2
	momentum = 0.8
	fCntList = [10,50]
	standardizeFlag = False
	preTrFlag = True
	
	#load training data data
	trnSet = load_ALLAML_Data('Trn',pp)
	tstSet = load_ALLAML_Data('Tst',pp)
	trData,trLabels = trnSet[:,:-1],trnSet[:,-1]
	tstData,tstLabels = tstSet[:,:-1],tstSet[:,-1]

	# initialize network hyper-parameters
	dict2={}
	dict2['inputDim']=np.shape(trData)[1]
	dict2['hL']=[100]
	dict2['hActFunc']=['tanh']
	dict2['oActFunc']='linear'
	dict2['errorFunc']='MSE'
	dict2['l1Penalty']=0.0002

	model = FeatureSelectingSCE(dict2)
	model.fit(trData,trLabels,preTraining=preTrFlag,optimizationFunc='Adam',learningRate=learning_rate,m=momentum,miniBatchSize=miniBatch_size,
		numEpochsPreTrn=num_epochs_pre,numEpochsPostTrn=num_epochs_post,standardizeFlag=standardizeFlag,verbose=True)

	#fWeights = model.fWeight.to('cpu').numpy()
	#fIndices = model.fIndices.to('cpu').numpy()
	fWeights = model.fWeight
	fIndices = model.fIndices
	feaList,feaW = returnImpFeaturesElbow(fWeights)
	print('No of selected features',len(feaW))
	fCntList = [10,50]
	#using the selected features run classification
	accuracyList = classifyALLAMLData(pp,feaList,fCntList,gpuId)
