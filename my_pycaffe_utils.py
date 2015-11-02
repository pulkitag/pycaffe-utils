## @package my_pycaffe_utils
#  Miscellaneous Util Functions
#

import my_pycaffe as mp
import my_pycaffe_io as mpio
import numpy as np
import pdb
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import subprocess
import collections as co
import other_utils as ou
import shutil
import copy
#import h5py as h5
from pycaffe_config import cfg

CAFFE_PATH = cfg.CAFFE_PATH

def zf_saliency(net, imBatch, numOutputs, opName, ipName='data', stride=2, patchSz=11):
	'''
		Takes as input a network and set of images imBatch
		net: Instance of MyNet
		imBatch: the images for which saliency needs to be computed. (expects: N * ch * h * w)
		numOutputs: Number of output units in the blob named opName
		Produces the saliency map of im
		net is of type MyNet
	'''

	assert(np.mod(patchSz,2)==1), 'patchSz needs to an odd Num'
	p = int(np.floor(patchSz/2.0))
	N = imBatch.shape[0]

	if isinstance(opName, basestring):
		opName = [opName]
		numOutputs = [numOutputs]
	
	#Transform the image
	imT = net.preprocess_batch(imBatch)

	#Find the Original Scores
	dataLayer = {}
	dataLayer[ipName] = imT
	batchSz,ch,nr,nc = imT.shape
	dType     = imT.dtype
	nrNew     = len(range(p, nr-p-1, stride))
	ncNew     = len(range(p, nc-p-1, stride)) 

	origScore = np.copy(net.net.forward_all(**dataLayer))
	for op in opName:
		assert op in origScore.keys(), "Some outputs not found"

	imSalient = {}
	for (op, num) in zip(opName, numOutputs):
		imSalient[op] = np.zeros((N, num, nrNew, ncNew))
 
	for (imCount,im) in enumerate(imT[0:N]):
		count   = 0
		imIdx   = []
		ims     = np.zeros(imT.shape).astype(dType)
		for (ir,r) in enumerate(range(p, nr-p-1, stride)):
			for (ic,c) in enumerate(range(p, nc-p-1, stride)):
				imPatched = np.copy(im)
				#Make an image patch 0
				imPatched[:, r-p:r+p+1, c-p:c+p+1] = 0	
				ims[count,:,:,:] = imPatched
				imIdx.append((ir,ic))
				count += 1
				#If count is batch size compute the features
				if count==batchSz or (ir == nrNew-1 and ic == ncNew-1):
					dataLayer = {}
					dataLayer[ipName] = net.preprocess_batch(ims)
					allScores = net.net.forward(**dataLayer)
					for (op,num) in zip(opName, numOutputs):
						scores = origScore[op][imCount] - allScores[op][0:count]
						scores = scores.reshape((count, num))
						for idx,coords in enumerate(imIdx):
							y, x = coords
							imSalient[op][imCount, :, y, x] = scores[idx,:].reshape(num,)
					count = 0
					imIdx = []	
	
	return imSalient, origScore


def mapILSVRC12_labels_wnids(metaFile):
	dat    = sio.loadmat(metaFile, struct_as_record=False, squeeze_me=True)
	dat    = dat['synsets']
	labels, wnid = [],[]
	for dd in dat:
		labels.append(dd.ILSVRC2012_ID - 1)
		wnid.append(dd.WNID)
	labels = labels[0:1000]
	wnid   = wnid[0:1000]	
	return labels, wnid

##
# Read ILSVRC Data
class ILSVRC12Reader:
	def __init__(self, caffeDir=CAFFE_PATH):
		labelFile  = '/data1/pulkitag/ILSVRC-2012-raw/devkit-1.0/data/ILSVRC2012_validation_ground_truth.txt'
		metaFile      = '/data1/pulkitag/ILSVRC-2012-raw/devkit-1.0/data/meta.mat'
		self.imFile_     = '/data1/pulkitag/ILSVRC-2012-raw/256/val/ILSVRC2012_val_%08d.JPEG'
		self.count_      = 0

		#Load the groundtruth labels from Imagenet
		fid  = open(labelFile)
		data = fid.readlines()
		valLabels = [int(i)-1 for i in data] #-1 for python formatting 
		
		#Load the Synsets
		synFile = os.path.join(caffeDir, 'data/ilsvrc12/synsets.txt')
		fid     = open(synFile, 'r')
		self.synsets_ = [s.strip() for s in fid.readlines()]
		fid.close()

		#Align the Imagenet Labels to the Synset order on which the BVLC reference model was trained.
		modLabels, modWnid = mapILSVRC12_labels_wnids(metaFile)
		self.labels_ = np.zeros((len(valLabels),)).astype(np.int)
		for (i,lb) in enumerate(valLabels):
			lIdx = modLabels.index(lb)
			syn  = modWnid[lIdx]
			sIdx = self.synsets_.index(syn)
			self.labels_[i] = sIdx

		#Load the synset words
		synWordFile = os.path.join(caffeDir, 'data/ilsvrc12/synset_words.txt')
		fid         = open(synWordFile, 'r')
		data        = fid.readlines()
		self.words_ = {}
		for l in data:
			synNames = l.split()
			syn      = synNames[0]
			words    = [w for w in synNames[1:]]
			self.words_[syn] = words
		fid.close()


	def reset(self):
		self.count_ = 0

	def set_count(self, count):
		self.count_ = count

	def read(self):
		imFile = self.imFile_ % (self.count_ + 1)
		im     = mp.caffe.io.load_image(imFile)
		lb     = self.labels_[self.count_]	
		syn    = self.synsets_[lb]
		words  = self.words_[syn]
		self.count_ += 1
		return im, lb, syn, words

	def word_label(self, lb):
		return self.words_[self.synsets_[lb]]	

def modelname_2_solvername(modelName):
	#Remove .caffemodel
	solverName = modelName[0:-11] + '.solverstate'
	return solverName


def read_layerdefs_from_proto(fName):
	'''
		Reads the definitions of layers from a protoFile
	'''
	fid = open(fName,'r')
	lines = fid.readlines()
	fid.close()

	layerNames, topNames  = [], []
	layerDef   = []
	stFlag     = True
	layerFlag  = False
	tmpDef = []
	for (idx,l) in enumerate(lines):
		isLayerSt = 'layer' in l
		if isLayerSt:
			if stFlag:
				stFlag = False
				layerNames.append('init')
				topNames.append('')
			else:
				layerNames.append(layerName)
				topNames.append(topName)
			layerDef.append(tmpDef)
			tmpDef    = []
			layerName, topName = mp.find_layer_name(lines[idx:])

		tmpDef.append(l)
		
	return layerNames, topNames, layerDef

## Produces names for layers
class LayerNameGenerator: 
	def __init__(self):
		self.count_ = 0 # counts the total number of layers
		self.nc_ = {} #Naming convention 
		self.nc_['InnerProduct']  = 'fc'
		self.nc_['ReLU']          = 'relu'
		self.nc_['Sigmoid']       = 'sigmoid'
		self.nc_['Concat']        = 'concat'
		self.nc_['EuclideanLoss'] = 'loss' 
		self.nc_['SoftmaxWithLoss'] = 'loss' 
		self.nc_['Dropout']       = 'drop' 
		self.nc_['Convolution']   = 'conv'
		self.nc_['Accuracy']      = 'accuracy'
		self.nc_['Pooling']       = 'pool'
		self.nc_['RandomNoise']   = 'rn'
		self.lastType_ = None

	def next_name(self, layerType):
		assert layerType in self.nc_.keys(), 'layerType %s not found' % layerType
		prefix = self.nc_[layerType]
		if layerType in ['ReLU','Pooling','Sigmoid'] and self.lastType_ in ['InnerProduct', 'Convolution']:
			pass
		elif layerType in ['Pooling'] and self.lastType_ in ['ReLU', 'Sigmoid']:
			pass
		elif layerType == 'Concat':
			#Don't increment the counter for concatenation layer. 
			pass
		else:
			self.count_ += 1
		name = '%s%d' % (prefix, self.count_)
		self.lastType_ = layerType
		return name

##
# Get protos for different objects needed in netdef .prototxt
def get_proto_dict(protoType, key, **kwargs):
	'''
		protoType: Type of proto to generate
		key      : the key to look for in kwargs
	'''
	#Initt the proto
	if kwargs.has_key(key):
		pArgs = kwargs[key]
	else:
		pArgs = {}
	pDict = co.OrderedDict()
	#Make the proto
	if protoType in ['param_w', 'param_b']:
		if protoType == 'param_w':
			defVals = {'lr_mult': str(1), 'decay_mult': str(1)}
		else:
			defVals = {'lr_mult': str(2), 'decay_mult': str(0)}
		#Name is not necessary
		if pArgs.has_key('name'):
				pDict['name'] = '"%s"' % pArgs['name']
		for key in defVals:
			if key in pArgs:
				pDict[key] = pArgs[key]
			else:
				pDict[key] = defVals[key]
	else:	
		raise Exception('Prototype %s not recognized' % protoType)
	return copy.deepcopy(pDict)

##
# Generate String for writing down a protofile for a layer. 
def get_layerdef_for_proto(layerType, layerName, bottom, numOutput=1, **kwargs):
	'''
		I recommend the following strategy for making layers:
			a. Make the basic layer architecture using this function.
			b. Modify this architecture as needed to change the layers. 
		
		##numOutput is depreciated. Instead use num_output in kwargs
		##kwargs will override all other parameters

	'''
	layerDef = co.OrderedDict()
	layerDef['name']  = '"%s"' % layerName
	layerDef['type']  = '"%s"' % layerType
	layerDef['bottom'] =	'"%s"' % bottom
	#The prms for different layers. 
	if layerType == 'InnerProduct':
		layerDef['top']    = '"%s"' % layerName
		layerDef['param'] = get_proto_dict('param_w', 'param', **kwargs)
		paramDup = make_key('param', layerDef.keys())
		layerDef[paramDup] = get_proto_dict('param_b', paramDup, **kwargs)
		ipKey = 'inner_product_param'
		layerDef[ipKey]  = co.OrderedDict()
		if kwargs.has_key('num_output'):
			layerDef[ipKey]['num_output'] = kwargs['num_output']
		else:
			layerDef[ipKey]['num_output'] = str(numOutput)
		layerDef[ipKey]['weight_filler'] = {}
		layerDef[ipKey]['weight_filler']['type'] = '"gaussian"'
		layerDef[ipKey]['weight_filler']['std']  = str(0.005)
		layerDef[ipKey]['bias_filler'] = {}
		layerDef[ipKey]['bias_filler']['type'] = '"constant"'
		layerDef[ipKey]['bias_filler']['value']  = str(1.)

	elif layerType == 'Convolution':
		layerDef['top']    = '"%s"' % layerName
		layerDef['param'] = get_proto_dict('param_w', 'param', **kwargs)
		paramDup = make_key('param', layerDef.keys())
		layerDef[paramDup] = get_proto_dict('param_b', paramDup, **kwargs)
		ipKey = 'convolution_param'
		layerDef[ipKey]  = co.OrderedDict()
		if kwargs.has_key('num_output'):
			layerDef[ipKey]['num_output'] = kwargs['num_output']
		else:
			layerDef[ipKey]['num_output'] = str(numOutput)
		layerDef[ipKey]['kernel_size']  = kwargs['kernel_size']
		layerDef[ipKey]['stride']       = kwargs['stride']
		if kwargs.has_key('pad'):
			layerDef[ipKey]['pad'] = kwargs['pad']
		if kwargs.has_key('group'):
			layerDef[ipKey]['group'] = kwargs['group']	
		layerDef[ipKey]['weight_filler'] = {}
		layerDef[ipKey]['weight_filler']['type'] = '"gaussian"'
		layerDef[ipKey]['weight_filler']['std']  = str(0.01)
		layerDef[ipKey]['bias_filler'] = {}
		layerDef[ipKey]['bias_filler']['type'] = '"constant"'
		layerDef[ipKey]['bias_filler']['value']  = str(0.)

	elif layerType in ['ReLU', 'Sigmoid']:
		if kwargs.has_key('top'):
			topName = kwargs['top']
		else:
			topName = layerName
		layerDef['top'] = '"%s"' % topName

	elif layerType == 'Pooling':
		if kwargs.has_key('top'):
			topName = kwargs['top']
		else:
			topName = layerName
		layerDef['top'] = '"%s"' % topName
		layerDef['pooling_param'] = {}
		layerDef['pooling_param']['pool'] = 'MAX'
		layerDef['pooling_param']['kernel_size'] = kwargs['kernel_size']
		layerDef['pooling_param']['stride']      = kwargs['stride']

	elif layerType=='Silence':
		#Nothing to be done
		pass

	elif layerType=='Dropout':
		if kwargs.has_key('top'):
			layerDef['top']    = '"%s"' % kwargs['top']
		else:
			layerDef['top']   = '"%s"' % layerName
		layerDef['dropout_param'] = co.OrderedDict()
		layerDef['dropout_param']['dropout_ratio'] = str(kwargs['dropout_ratio'])

	elif layerType in ['Accuracy']:
		assert kwargs.has_key('bottom2')
		bottom2 = make_key('bottom', layerDef.keys())
		layerDef[bottom2] = '"%s"' % kwargs['bottom2']
		if kwargs.has_key('top'):
			layerDef['top']   = '"%s"' % kwargs['top']
		else:
			layerDef['top']   = '"%s"' % layerName

	elif layerType in ['EuclideanLoss', 'SoftmaxWithLoss']:
		assert kwargs.has_key('bottom2')
		bottom2 = make_key('bottom', layerDef.keys())
		layerDef[bottom2] = '"%s"' % kwargs['bottom2']
		if kwargs.has_key('top'):
			layerDef['top']   = '"%s"' % kwargs['top']
		else:
			layerDef['top']   = '"%s"' % layerName
		layerDef['loss_weight'] = 1

	elif layerType in ['ContrastiveLoss']:
		assert kwargs.has_key('bottom2') and kwargs.has_key('bottom3')
		bottom2 = make_key('bottom', layerDef.keys())
		layerDef[bottom2] = '"%s"' % kwargs['bottom2']
		bottom3 = make_key('bottom', layerDef.keys())
		layerDef[bottom3] = '"%s"' % kwargs['bottom3']
		if kwargs.has_key('top'):
			layerDef['top']   = '"%s"' % kwargs['top']
		else:
			layerDef['top']   = '"%s"' % layerName
		#Add the contrastive loss margin
		layerDef['contrastive_loss_param'] = co.OrderedDict()
		if kwargs.has_key('margin'):
			layerDef['constrastive_loss_param']['margin'] = kwargs['margin']
		else:
			layerDef['constrastive_loss_param']['margin'] = 1.0
		layerDef['loss_weight'] = 1

	elif layerType == 'Concat':
		assert kwargs.has_key('bottom2')
		assert kwargs.has_key('concat_dim')
		bottom2 = make_key('bottom', layerDef.keys())
		layerDef[bottom2] = '"%s"' % kwargs['bottom2']
		layerDef['concat_param'] = co.OrderedDict()
		layerDef['concat_param']['concat_dim'] = kwargs['concat_dim']
		layerDef['top']   = '"%s"' % layerName

	elif layerType in ['DeployData']:
		layerDef['input'] = '"%s"' % layerName
		ipDims = kwargs['ipDims'] #(batch_size, channels, h, w)
		layerDef['input_dim'] = ipDims[0]
		key  = make_key('input_dim', layerDef.keys())
		layerDef[key] = ipDims[1]
		layerDef[make_key('input_dim', layerDef.keys())] = ipDims[2]		
		layerDef[make_key('input_dim', layerDef.keys())] = ipDims[3]		

	elif layerType in ['RandomNoise']:
		layerDef['top']    = '"%s"' % layerName
		if kwargs.has_key('adaptive_sigma'):
			layerDef['random_noise_param'] = co.OrderedDict()
			layerDef['random_noise_param']['adaptive_sigma']  = kwargs['adaptive_sigma']
			layerDef['random_noise_param']['adaptive_factor'] = kwargs['adaptive_factor'] 		

	else:
		raise Exception('%s layer type not found' % layerType)
	return layerDef

##
# Get the layerdefs for siamese layers
def get_siamese_layerdef_for_proto(layerType, layerName, bottom, numOutput=1, **kwargs):
	if layerType in ['InnerProduct', 'Convolution']:
		paramKey    = 'param'
		paramDupKey = make_key('param', [paramKey])
		#Weight param name
		wName = ou.get_item_dict(kwargs, [paramKey] + ['name'])
		if wName is None:
			wName = layerName + '_w'
		#Bias param name
		bName = ou.get_item_dict(kwargs, [paramDupKey] + ['name']) 
		if bName is None:
			 bName = layerName + '_b'
		#Set the info
		kwargs['param'] = {}
		kwargs['param']['name'] = wName
		kwargs[paramDupKey] = {}
		kwargs[paramDupKey]['name'] = bName
	elif layerType in ['ReLU', 'Pooling']:
		pass
	else:
		raise Exception('Siamese layer not supported for %s' % layerType)
		
	#First Siamese layer
	def1 = get_layerdef_for_proto(layerType, layerName, bottom, numOutput=numOutput, **kwargs)
	#Second Siamese layer
	layerName = layerName + '_p'
	bottom    = bottom + '_p'
	def2 = get_layerdef_for_proto(layerType, layerName, bottom, numOutput=numOutput, **kwargs)
	return def1, def2


##
#Make a siamese counter part
def siamese_from_layerdef(lDef, botName=None):
	name, top, bot = lDef['name'], lDef['top'], lDef['bottom']
	suffix = '_p'
	smName  = '"%s"' % (name[1:-1] + suffix)
	smTop   = '"%s"' % (top[1:-1]  + suffix)
	if botName is not None:	
		smBot = '"%s"' % botName
	else:
		smBot = '"%s"' % (bot[1:-1] + suffix)

	smDef = co.OrderedDict(lDef)
	smDef['name'] = smName
	smDef['top']  = smTop
	smDef['bottom']  = smBot
	return smDef


##
#Helper fucntion for process_debug_log - helps find specific things in the line
def find_in_line(l, findType):
	if findType == 'iterNum':
		assert 'Iteration' in l, 'Iteration word must be present in the line'
		ss  = l.split()
		idx = ss.index('Iteration')
		return int(ss[idx+1][0:-1])
	elif findType == 'layerName':
		assert 'Layer' in l, 'Layer word must be present in the line'
		ss = l.split()
		idx = ss.index('Layer')
		return ss[idx+1][:-1]

	else:
		raise Exception('Unrecognized findType')

##
# Debug log that contatins how the parameters and features are changing during
# forward, backward and update stages.  
def process_debug_log(logFile, setName='train'):
	'''
		The debug file contains infromation in the following format
		Forward Pass
		Backward Pass
		#The loss of the the network (Since, backward pass has not been used to update,
		 the loss corresponds to previous iteration. 
		Update log. 
	'''

	assert setName in ['train','test'], 'Unrecognized set-name'

	fid = open(logFile, 'r')
	lines = fid.readlines()	

	#Find the layerNames	
	layerNames = []
	for (count, l) in enumerate(lines):
		if 'Setting up' in l and 'Setting up crop layers' not in l:
			name = l.split()[-1]
			if name not in layerNames:
				layerNames.append(name)
		if 'Solver scaffolding' in l:
			break
	lines = lines[count:] 

	#Start collecting the required stats. 
	iters  = []
	
	#Variable Intialization
	allBlobOp, allBlobParam, allBlobDiff, allBlobUpdate = {}, {}, {}, {}
	blobOp, blobParam, blobDiff, blobUpdate = {}, {}, {}, {}
	for name in layerNames:
		allBlobOp[name], allBlobParam[name], allBlobDiff[name], allBlobUpdate[name] = [], [], [], []
		blobOp[name], blobParam[name], blobDiff[name], blobUpdate[name] = [], [], [], []

	#Only when seeFlag is set to True, we will collect the required data stats.  
	seeFlag    = False
	updateFlag = False
	appendFlag = False
	writeFlag  = False
	for (count, l) in enumerate(lines):
		#Find the start of a new and relevant iteration 
		if 'Forward' in l and writeFlag == True:
			for name in layerNames:
				allBlobOp[name].append(blobOp[name])
				allBlobParam[name].append(blobParam[name])
				allBlobDiff[name].append(blobDiff[name])
				allBlobUpdate[name].append(blobUpdate[name])
				blobOp[name], blobParam[name], blobDiff[name], blobUpdate[name] = [], [], [], []
			writeFlag = False

		#The training log starts after the line Test Net Output
		if setName=='train' and ('Test' in l and 'output' in l):
			seeFlag = True
			print "Setting to true"

		#If there is a testing network then set the seeFlag to False	
		if setName=='train' and 'Testing' in l:
			seeFlag = False

		if seeFlag and 'Iteration' in l and 'lr' in l: 	
			iterNum = find_in_line(l, 'iterNum')
			iters.append(iterNum)	
		
		if seeFlag and 'Forward' in l:
			lName = find_in_line(l, 'layerName')
			print lName	
			if 'top blob' in l:	
				blobOp[lName].append(float(l.split()[-1]))
			if 'param blob' in l:
				blobParam[lName].append(float(l.split()[-1]))

		#Find the back propogated diffs
		if seeFlag and ('Backward' in l) and ('Layer' in l):
			lName = find_in_line(l, 'layerName')
			if 'param blob' in l:
				blobDiff[lName].append(float(l.split()[-1]))

		#Find the update
		if seeFlag and 'Update' in l and 'Layer' in l:
			#Forward after the update is the time when we need to write.
			writeFlag = True
			lName = find_in_line(l, 'layerName')
			assert 'diff' in l
			assert l.split()[-2] == 'diff:', 'I Found %s instead of diff:' % l.split()[-2]
			blobUpdate[lName].append(float(l.split()[-1]))	 

	fid.close()
	for name in layerNames:
		print name
		pdb.set_trace()
		data = [np.array(a).reshape(1,len(a)) for a in allBlobOp[name]]
		allBlobOp[name] = np.concatenate(data, axis=0)
		data = [np.array(a).reshape(1,len(a)) for a in allBlobParam[name]]
		allBlobParam[name] = np.concatenate(data, axis=0)
		data = [np.array(a).reshape(1,len(a)) for a in allBlobDiff[name]]
		allBlobDiff[name] = np.concatenate(data, axis=0)
		data = [np.array(a).reshape(1,len(a)) for a in allBlobUpdate[name]]
		allBlobUpdate[name] = np.concatenate(data, axis=0)
		
	#If there are K blobs in a layer, then the size will N * K where N is the number of samples
	# and K is the number of blobs. 
	return allBlobOp, allBlobParam, allBlobDiff, allBlobUpdate, layerNames, iters

##
# Helper function for plot_debug_log
def plot_debug_subplots_(data, subPlotNum, label, col, fig, 
					numPlots=3):
	'''
		data: N * K where N is the number of points and K 
					is the number of outputs
		subPlotNum: The subplot Number
		label     : The label to put
		col       : The color of the line
		fig       : The figure to use for plotting
		numPlots  : number of plots
	'''
	plt.figure(fig.number)
	if data.ndim > 0:
		data = np.abs(data)
		'''
		#print data.shape
		#if data.ndim > 1:
			data = np.sum(data, axis=tuple(range(1,data.ndim)))
		#print label, data.shape
		'''
		for i in range(data.shape[1]):
			plt.subplot(numPlots,1,subPlotNum + i)
			print label + '_blob%d' % i
			plt.plot(range(data.shape[0]), data[:,i], 
								label=label +  '_blob%d' % i, color=col)
			plt.legend()

##
# Plots the weight and features values from a debug enabled log file. 
def plot_debug_log(logFile, setName='train', plotNames=None):
	'''
		blobOp: The output of the blob
		blobParam : The weight of the blobs
		blobDiff  : The gradients of the blob
		blobUpdate: The update for the blob. 
	'''
	blobOp, blobParam, blobDiff, blobUpdate, layerNames, iterNum\
							 = process_debug_log(logFile, setName=setName)
	if plotNames is not None:
		for p in plotNames:
			assert p in layerNames, '%s layer not found' % p
	else:
		plotNames = [l for l in layerNames]

	colIdx = np.linspace(0,255,len(plotNames)).astype(int)
	plt.ion()
	print colIdx
	print layerNames
	for count,name in enumerate(plotNames):
		fig = plt.figure()
		col = plt.cm.jet(colIdx[count])
		#Number of plots
		numPlots = blobOp[name].shape[1] + blobParam[name].shape[1] +\
							 blobDiff[name].shape[1] + blobUpdate[name].shape[1]
		#Plot the data
		plot_debug_subplots_(blobOp[name], 1, name + '_data', col,
												 fig, numPlots=numPlots)
		#Plot the norms of the weights. 
		plot_debug_subplots_(blobParam[name], 1 + blobOp[name].shape[1],
										name + '_weights', col, fig, numPlots=numPlots)
		#Norms of the gradient
		plot_debug_subplots_(blobDiff[name],
							 1 + blobOp[name].shape[1] + blobParam[name].shape[1],
							 name + '_diff', col,  fig, numPlots=numPlots)
		#The ratio of update/weights
		lUpdate = blobUpdate[name]
		ratio   = np.zeros(lUpdate.shape)	
		if lUpdate.shape[1]>0:
			for d in range(lUpdate.shape[1]):
				idx = blobParam[name][:,d] == 0	
				ratio[:,d] = lUpdate[:,d] / blobParam[name][:,d] 
				ratio[idx,d] = 0
			plot_debug_subplots_(ratio,
				1 + blobOp[name].shape[1] + blobParam[name].shape[1] + blobDiff[name].shape[1],
				name + '_ration', col, fig, numPlots=numPlots)	
	
	plt.show()	


##
# Get the accuracy from the test log
def test_log2acc(logFile):
	acc = None
	with open(logFile, 'r') as fid:
		lines = fid.readlines()
		data  = lines[-2].split()
		assert data[-3] == 'accuracy', 'Something is wrong in the way the accuracy is being read'
		acc = float(data[-1])
	return acc


##
# Get useful caffe paths
# Set the paths over here for using the utils code. 
def get_caffe_paths():
	paths  = {}
	paths['caffeMain']   = CAFFE_PATH 
	paths['tools']       = os.path.join(paths['caffeMain'], 'build', 'tools')
	paths['pythonTest']  = os.path.join(paths['caffeMain'], 'python', 'test')
	return paths


def give_run_permissions_(fileName):
	args = ['chmod u+x %s' % fileName]
	subprocess.check_call(args,shell=True)


##
#Make a new key if there are duplicates.
def make_key(key, keyNames, dupStr='_$dup$'):
	'''
		The new string is made by concatenating the dupStr,
		until a unique name is found. 
	'''
	if key not in keyNames:
		return key
	else:
		key = key + dupStr
		return make_key(key, keyNames, dupStr)

##
# If the key has dupStr, strip it off. 
def strip_key(key, dupStr='_$dup$'):
	if dupStr in key:
		idx = key.find(dupStr)
		key = key[:idx]	
	return key 


##
# Extracts message parameters of a .prototxt from a list of strings. 
def get_proto_param(lines):
	#The String to use in case of duplicate names
	dupStr    = '_$dup$'
	data      = co.OrderedDict()
	braCount  = 0
	readFlag  = True
	i         = 0
	while readFlag:
		if i>=len(lines):
			break
		l = lines[i]
		#print l.strip()
		if l in ['', '\n']:
			#Continue if empty line encountered.
			print 'Skipping empty line: %s ' % l.strip()
		elif '#' in l:
			#In case of comments
			print 'Ignoring line: %s' % l.strip()
		elif '{' in l and '}' in l:
			raise Exception('Reformat the file, both "{" and "}" cannot be present on the same line %s' % l)
		elif '{' in l:
			name       = l.split()[0]
			if '{' in name:
				assert name[-1] == '{'
				name = name[:-1]
			name       = make_key(name, data.keys(), dupStr=dupStr)
			data[name], skipI = get_proto_param(lines[i+1:]) 
			braCount += 1
			i        += skipI
		elif '}' in l:
			braCount -= 1
		else:
			#print l
			splitVals = l.strip().split(':')
			if len(splitVals) > 2:
				raise Exception('Improper format for: %s, l.split() should only produce 2 values' % l)
			name, val  = splitVals
			name       = name.strip()
			val        = val.strip()
			name       = make_key(name, data.keys(), dupStr=dupStr)
			data[name] = val
		if braCount == -1:
			break
		i += 1
	return data, i

##
# Write the proto information into a file. 
def write_proto_param(fid, protoData, numTabs=0):
	'''
		fid :    The file handle to which data needs to be written.
		data:    The data to be written. 
		numTabs: Is for the proper alignment. 
	'''
	tabStr = '\t ' * numTabs 
	for (key, data) in protoData.iteritems():
		if data is None:
			continue
		key = strip_key(key)
		if isinstance(data, dict):
			line = '%s %s { \n' % (tabStr,key)
			fid.write(line)
			write_proto_param(fid, data, numTabs=numTabs + 1)
			line = '%s } \n' % (tabStr)
			fid.write(line)
		else:
			line = '%s %s: %s \n' % (tabStr, key, data)
			fid.write(line)

##
# Write proto param for a layer
def write_proto_param_layer(fid, protoData):
	'''
		fid      : File Handle
		protoData: The data that needs to be written 
	'''
	assert protoData.has_key('name') and protoData.has_key('type'),\
		'layer must have a name and a type'
	name, layerType = protoData['name'][1:-1], protoData['type'][1:-1]
	if layerType =='DeployData':
		layerProto = copy.deepcopy(protoData)
		del layerProto['name'], layerProto['bottom'], layerProto['type']
		write_proto_param(fid, layerProto, 0)
	else:	
		fid.write('layer { \n')
		write_proto_param(fid, protoData, numTabs=0)
		fid.write('} \n')


##
# Reads the architecture definition file and converts it into a nice, programmable format. 		
class ProtoDef():
	ProtoPhases = ['TRAIN', 'TEST'] 
	def __init__(self, defFile=None, layerDict=None):
		if layerDict is not None:
			assert layerDict.has_key('TRAIN') and layerDict.has_key('TEST')
			self.layers_   = co.OrderedDict(layerDict)
			self.initData_ = []
			return 
		#If initializing from a file
		self.layers_ = {}
		self.layers_['TRAIN'] = co.OrderedDict()	
		self.layers_['TEST']  = co.OrderedDict()
		self.siameseConvert_  = False  #If the def has been convered to siamese or not. 
		fid   = open(defFile, 'r')
		lines = fid.readlines()
		i     = 0
		layerInit = False
		#Lines that are there before the layers start. 
		self.initData_ = []
		while True:
			l = lines[i]
			if not layerInit:
				self.initData_.append(l)
			if ('layers' in l) or ('layer' in l):
				layerInit = True
				layerName,_ = mp.find_layer_name(lines[i:])
				layerData, skipI  = get_proto_param(lines[i+1:])
				if layerData.has_key('include'):
					phase = layerData['include']['phase']
					assert phase in ['TRAIN', 'TEST'], '%s phase not recognized' % phase
					assert layerName not in self.layers_[phase].keys(), 'Duplicate LayerName Found'
					self.layers_[phase][layerName] = layerData
				else:
					#Default Phase is Train
					assert layerName not in self.layers_['TRAIN'].keys(),\
																 'Duplicate LayerName: %s found' % layerName
					self.layers_['TRAIN'][layerName] = layerData
				i += skipI
			i += 1
			if i >= len(lines):
				break
		#The last line of iniData_ is going to be "layer {", so remove it. 
		self.initData_ = self.initData_[:-1]

	@classmethod
	def deploy_from_proto(cls, initDef, dataLayerNames=['data'], imSz=[[3,128,128]], **kwargs):
		if isinstance(initDef, ProtoDef):
			netDef = copy.deepcopy(initDef) 
		else:
			assert isinstance(initDef, str), 'Invalid format of netDef'
			netDef = cls(initDef)
		netDef.make_deploy(dataLayerNames=dataLayerNames, imSz=imSz, **kwargs)
		return netDef

	@classmethod
	def visproto_from_proto(cls, initDef, dataLayerNames=['window_data'],
													 imSz=[[3,101,101]], delAbove='conv1'):
		if isinstance(initDef, ProtoDef):
			netDef = copy.deepcopy(initDef) 
		else:
			assert isinstance(initDef, str), 'Invalid format of netDef'
			netDef = cls(initDef)
		for ph in ['TRAIN', 'TEST']:
			delFlag = False
			lNames = netDef.get_all_layernames(phase=ph)
			for l in lNames:
				if l in dataLayerNames or delFlag:
					netDef.del_layer(l)	
				if l == delAbove:
					delFlag = True
		#Make the approrpiate header
		assert (len(dataLayerNames) == 1)
		ch, h, w = imSz[0]
		netDef.initData_.append('input: "%s"\n' % 'data')
		netDef.initData_.append('input_dim: %d\n' % 10)
		netDef.initData_.append('input_dim: %d\n' % ch)
		netDef.initData_.append('input_dim: %d\n' % h)
		netDef.initData_.append('input_dim: %d\n' % w)
		return netDef

	##
	# Convert a network into a siamese network. 
	def get_siamese(self, firstName, lastName):
		'''
			Make Siamese from bot to top layers
		'''
		stList, enList = co.OrderedDict(), co.OrderedDict()
		sList1 = co.OrderedDict()
		sList2 = co.OrderedDict()
		for ph in ['TRAIN', 'TEST']:
			stList[ph], enList[ph] = co.OrderedDict(), co.OrderedDict()
			sList1[ph], sList2[ph] = co.OrderedDict(), co.OrderedDict()
			stFlag, enFlag = False, False
			for lKey, layer in self.layers_[ph].iteritems():
				if not stFlag and lKey == firstName:
					stFlag = True
				if not stFlag and not enFlag:
					print (lKey)
					stList[ph][lKey] = copy.deepcopy(layer)
				#At the end of the siamese net. 
				if enFlag:
					print(lKey)
					enList[ph][lKey] = copy.deepcopy(layer)
				#Copy the layers into the siamese streams
				if stFlag:
					sList1[ph][lKey] = copy.deepcopy(layer)
					siamLayer    = siamese_from_layerdef(layer)
					siamName     = siamLayer['name'][1:-1]
					print (lKey, siamName)
					sList2[ph][siamName] = siamLayer
				if lKey == lastName:
					stFlag = False
					enFlag = True
					#Make the Concat Layer
					#concatDef = get_layerdef_for_proto('concat')				
	
		#Combine the layers
		netDef = co.OrderedDict()
		for ph in ['TRAIN', 'TEST']:
			netDef[ph] = co.OrderedDict()
			for k, v in stList[ph].iteritems():
				netDef[ph][k] = v
			for k, v in sList1[ph].iteritems():
				netDef[ph][k] = v
			for k, v in sList2[ph].iteritems():
				netDef[ph][k] = v
			for k, v in enList[ph].iteritems():
				netDef[ph][k] = v
		protoDef =  ProtoDef(layerDict=netDef)	
		protoDef.initData_ = copy.deepcopy(self.initData_)
		return protoDef

	##
	#Make a deploy prototxt file
	def make_deploy(self, dataLayerNames=['data'], imSz=[[3,128,128]], 
									batchSz=None, delLayers=None):
		'''
			the deployNet will have only 1 phase, by default it will be set to TRAIN
			From the original network copy the data layers of the test phase and all
			other layers from the training phase. 
		'''
		deployNet = co.OrderedDict()
		for ph in ProtoDef.ProtoPhases:
			deployNet[ph] = co.OrderedDict()

		trPhase = 'TRAIN'
		tePhase = 'TEST'
		#Copy the data layers from the test phase
		for name,sz in zip(dataLayerNames, imSz):
			assert self.layers_[tePhase].has_key(name), '%s data layer not found' % name
			if batchSz is None:
				batchSzPath = ou.find_path_key(self.layers_[tePhase][name], 'batch_size')
				batchSz     = ou.get_item_recursive_key(self.layers_[tePhase][name], batchSzPath)
			layerName   = self.layers_[tePhase][name]['name']
			sz          = [batchSz] + sz
			szDict = {'ipDims': sz}
			deployNet[trPhase][name] = get_layerdef_for_proto('DeployData', name, None, **szDict)	
		#Copy all the other layers from the training network
		for name in self.layers_[trPhase].keys():
			if name in dataLayerNames:
				continue
			else:
				if delLayers is not None:
					if name in delLayers:
						continue
				deployNet[trPhase][name] = copy.deepcopy(self.layers_[trPhase][name])
		self.layers_ = copy.deepcopy(deployNet)	
	
	##
	# Write the prototxt architecture file
	def write(self, outFile):
		with open(outFile, 'w') as fid:
			#Write Init Data
			for l in self.initData_:
				fid.write(l)
			#Write TRAIN/TEST Layers
			for (key, data) in self.layers_['TRAIN'].iteritems():
				write_proto_param_layer(fid, data)
				#Write the test layer if it is present.  
				if self.layers_['TEST'].has_key(key):
					write_proto_param_layer(fid, self.layers_['TEST'][key])
			#Write the layers in TEST which were not their in TRAIN
			testKeys = self.layers_['TEST'].keys()
			for key in testKeys:
				if key in self.layers_['TRAIN'].keys():
					continue
				else:
					write_proto_param_layer(fid, self.layers_['TEST'][key], numTabs=0)

	##
	# Find the layers that have learning rates
	def find_learning_layers(self):
		layerNames = co.OrderedDict()
		for ph in ProtoDef.ProtoPhases:
			layerNames[ph] = []
			for lName in self.layers_[ph]:
				if self.layers_[ph][lName].has_key('param'):
					layerNames[ph].append(lName)
		return layerNames

	##
	# If the layer has learnable parameters set them to zero
	def set_no_learning(self, layerName, phase='TRAIN'):
		'''
			Generally the learning layers have weights and biases
			Currently this code will only work for these layers. 
		'''
		lType = self.get_layer_property(layerName, 'type', phase=phase)
		if lType in ['PReLU']:
			numParam = 1
		else:
			numParam = 2
		if numParam == 1:
			self.set_layer_property(layerName, ['param','lr_mult'], 0., phase=phase, propNum=[0, 0])
			self.set_layer_property(layerName, ['param','decay_mult'], 0., 
																				phase=phase, propNum=[0, 0])
		else:
			self.set_layer_property(layerName, ['param','lr_mult'], 0., phase=phase, propNum=[1, 0])
			self.set_layer_property(layerName, ['param','decay_mult'], 0.,
																			 phase=phase, propNum=[1, 0])

	##
	# Set std for all the layers
	def set_std_all(self, std):
		layerNames = self.find_learning_layers()
		for ph in layerNames.keys():
			for name in layerNames[ph]:
				keyPath = ou.find_path_key(self.layers_[ph][name], 'weight_filler')
				if len(keyPath) > 0:
					fillerType = ou.get_item_recursive_key(self.layers_[ph][name], 
												keyPath + ['type'])
					if fillerType=='"gaussian"':
						keyPath = keyPath + ['std']
						ou.set_recursive_key(self.layers_[ph][name], keyPath, std)
					else:
						print "Filler type for %s is not gaussian" % name
 
	##
	#Make all the layers uptil a given layer as non-learnable
	#Since the layers are stored in form of an ordered-dict this will work. 
	def set_no_learning_until(self, layerName):
		'''
			Useful when finetuning only a few layers in the network.
			layerName: The last layer that shall have the non-zero learning rate
		'''
		allLayers = self.find_learning_layers()
		foundFlag = False
		for ph in ProtoDef.ProtoPhases:
			doFlag = True
			lNames = allLayers[ph]
			count  = 0
			if len(lNames)==0:
				continue
			while doFlag:
				if lNames[count] == layerName:
					doFlag = False
					foundFlag = True
				else:
					#print 'No-Learning: %s' % lNames[count]
					self.set_no_learning(lNames[count], phase=ph)
				count += 1
				if count == len(lNames):
					doFlag = False

		#pdb.set_trace()
		if not foundFlag:
			raise Exception('Layer Name %s is not learnable or not found'
					% layerName)

	##
	def get_layer_property(self, layerName, propName, phase='TRAIN', propNum=0):
		'''
			See documentation of set_layer_property for meaning of variable names. 
		'''
		if not isinstance(propName, list):
			#Modify the propName to account for duplicates
			propName = propName + '_$dup$' * propNum
			propName = [propName]
		else:
			if isinstance(propNum, list):
				assert len(propNum)==len(propName), 'Lengths mismatch'
				propName = [p + i * '_$dup$' for (p,i) in zip(propName, propNum)]
			else:
				assert propNum==0,'propNum is not appropriately specified'
		return ou.get_item_dict(self.layers_[phase][layerName], propName)	

	def rename_layer(self, oldLayerName, newLayerName, phase='TRAIN'):
		assert oldLayerName in self.layers_[phase].keys(), '%s layer not found' % layerName
		layerTemp = co.OrderedDict()
		for k,v in self.layers_[phase].iteritems():
			if k == oldLayerName:
				layerTemp[newLayerName] = copy.deepcopy(self.layers_[phase][oldLayerName])
				layerTemp[newLayerName]['name'] = '"%s"' % newLayerName
			else:
				layerTemp[k] = copy.deepcopy(self.layers_[phase][k])
		#Delete the old keyrs
		for k in self.layers_[phase].keys():
			del self.layers_[phase][k]
		#Copy the values
		for k in layerTemp.keys():
			self.layers_[phase][k] = copy.deepcopy(layerTemp[k])
		print ('WARNING RENAME %s LAYER - DOESNOT CHANGES THE TOP NAME or PARAM NAMES' % oldLayerName)

	##					
	def set_layer_property(self, layerName, propName, value, phase='TRAIN',  propNum=0): 
		'''
			layerName: Name of the layer in which the property is present.
			propName : Name of the property.
								 If there is a recursive property like, layer['data_param']['source'],
								 then provide a list.   
			value    : The value of the property. 
			phase    : The phase in which to make edits. 
			propNum  : Some properties like top can duplicated mutliple times in a layer, so which one.
		'''
		assert phase in ProtoDef.ProtoPhases, 'phase name not recognized'
		assert layerName in self.layers_[phase].keys(), '%s layer not found' % layerName
		if not isinstance(propName, list):
			#Modify the propName to account for duplicates
			propName = propName + '_$dup$' * propNum
			propName = [propName]
		else:
			if isinstance(propNum, list):
				assert len(propNum)==len(propName), 'Lengths mismatch'
				propName = [p + i * '_$dup$' for (p,i) in zip(propName, propNum)]
			else:
				assert propNum==0,'propNum is not appropriately specified'
		#Set the value
		ou.set_recursive_key(self.layers_[phase][layerName], propName, value)
		#If the name of the layer is changing
		if len(propName) ==1 and propName[0]=='name':
			self.rename_layer(layerName, value, phase=phase)	
	
	##					
	def del_layer_property(self, layerName, propName, phase='TRAIN',  propNum=0): 
		'''
			layerName: Name of the layer in which the property is present.
			propName : Name of the property.
								 If there is a recursive property like, layer['data_param']['source'],
								 then provide a list.   
			phase    : The phase in which to make edits. 
			propNum  : Some properties like top can duplicated mutliple times in a layer, so which one.
		'''
		assert phase in ProtoDef.ProtoPhases, 'phase name not recognized'
		assert layerName in self.layers_[phase].keys(), '%s layer not found' % layerName
		if not isinstance(propName, list):
			#Modify the propName to account for duplicates
			propName = propName + '_$dup$' * propNum
			propName = [propName]
		else:
			if isinstance(propNum, list):
				assert len(propNum)==len(propName), 'Lengths mismatch'
				propName = [p + i * '_$dup$' for (p,i) in zip(propName, propNum)]
			else:
				assert propNum==0,'propNum is not appropriately specified'
		#Set the value
		ou.del_recursive_key(self.layers_[phase][layerName], propName)


	##
	#Get layer from name
	def get_layer(self, layerName, phase='TRAIN'):
		assert layerName in self.layers_[phase].keys(), 'Layer doesnot exists'
		return copy.deepcopy(self.layers_[phase][layerName])

	##
	def add_layer(self, layerName, layer, phase='TRAIN'):
		assert layerName not in self.layers_[phase].keys(), 'Layer already exists'
		lProtoName = layer['name'][1:-1]
		assert layerName == lProtoName, 'Inconsistency in names, (%s,%s)' % (layerName, lProtoName)
		self.layers_[phase][layerName] = layer	

	##
	def del_layer(self, layerName):
		for phase in self.layers_.keys():
			if not isinstance(layerName, list):
				layerName = [layerName]
			for l in layerName:
				if self.layers_[phase].has_key(l):
					del self.layers_[phase][l]

	##
	def del_all_layers_above(self, layerName):
		'''
			Delete all layers above layerName
			layerName will not be deleted. 	
		'''
		delLayers = co.OrderedDict()
		for ph in ProtoDef.ProtoPhases:
			delLayers[ph] = []
			encounterFlag = False
			#Find the layers to delete
			for name in self.layers_[ph]:
				if not encounterFlag and name != layerName:
					continue
				if encounterFlag:
					delLayers[ph].append(name)
				if name==layerName:
					encounterFlag = True
			#Delete the layers
			for name in delLayers[ph]:
				self.del_layer(name)
	
	##
	# Get the name of the top of the last layer. 
	def get_last_top_name(self, phase='TRAIN'):
		'''
			This needs to be imrpoved - right now it assumes that the last
			layer only has a single top blob
		'''
		lastKey = next(self.layers_[phase].__reversed__())
		#Ensure there is only one top
		topKey      = make_key('top', self.layers_[phase][lastKey])
		topExpected = make_key('top', ['top'])
		assert topKey==topExpected, 'There are multiple tops in %s' % lastKey
		topName = self.layers_[phase][lastKey]['top'][1:-1]
		return topName

	##
	# Get the layerNames from the type of the layer. 
	def get_layernames_from_type(self, layerType, phase='TRAIN'):
		layerNames = []
		for lName in self.layers_[phase].keys():
			lType = self.layers_[phase][lName]['type'][1:-1]
			if lType == layerType:
				layerNames.append(lName)
		return layerNames

	##
	#Get all the layernames
	def get_all_layernames(self, phase='TRAIN'):
		names = [l for l in self.layers_[phase].keys()]
		return names

	##
	#Append layers from protodef
	

##
# Class for making the solver_prototxt
class SolverDef:
	def __init__(self):
		self.data_ = co.OrderedDict()

	@classmethod
	def from_file(cls, inFile):
		self = cls()
		with open(inFile,'r') as f:
			lines = f.readlines()
			self.data_,_ = get_proto_param(lines)
		return self

	##
	# Add a property if not there, modifies if already exists
	def add_property(self, propName, value):
		if propName in self.data_.keys():
			self.set_property(propName, value)
		else:
			self.data_[propName] = value

	##
	# Delete Property
	def del_property(self, propName):
		assert propName in self.data_.keys(), '%s not found' % propName
		del self[propName]

	##
	# Get property
	def get_property(self, propName):
		assert propName in self.data_.keys(), '%s not found' % propName
		return self.data_[propName]

	##
	# Set property
	def set_property(self, propName, value): 
		assert propName in self.data_.keys()
		if not isinstance(propName, list):
			propName = [propName]
		ou.set_recursive_key(self.data_, propName, value)

	##
  # Write the solver file
	def write(self, outFile):
		with open(outFile, 'w') as fid:
			fid.write('# Autotmatically generated solver prototxt \n')
			if self.data_.has_key('device_id'):
				if type(self.data_['device_id']) == list:
					#self.data_['device_id'] = ''.join('%d,' % d for d in self.data_['device_id'])[0:-1]
					self.data_['device_id'] = self.data_['device_id'][0]
			write_proto_param(fid, self.data_, numTabs=0)

##
# Get the defaults
def get_defaults(setArgs, defArgs):
	for key in setArgs.keys():
		assert defArgs.has_key(key), 'Key not found: %s' % key
		defArgs[key] = copy.deepcopy(setArgs[key])
	return defArgs


##
# Programatically make a solver. 
def make_solver(**kwargs):
	defArgs = co.OrderedDict([('net', '""'), ('test_iter', 100), ('test_interval', 1000),
						 ('base_lr', 0.01), ('momentum', 0.9), ('weight_decay', 0.0005),
						 ('gamma', 0.1), ('stepsize', 100000), ('lr_policy', '"step"'),
						 ('display', 20),('max_iter', 310000), ('snapshot', 10000),
						 ('snapshot_prefix', '""'), ('solver_mode', 'GPU'), ('device_id', 1),
						 ('debug_info', 'false'), ('clip_gradients', -1)])

	defArgs = get_defaults(kwargs, defArgs)  
	sol = SolverDef()
	for key in defArgs.keys():	
		sol.add_property(key, defArgs[key])
	return sol

	
##
# Programatically write Caffe Experiment Files. 
class ExperimentFiles:
	def __init__(self, modelDir, defFile='caffenet.prototxt', 
							 solverFile='solver.prototxt',
							 logFileTrain='log_train.txt', logFileTest='log_test.txt', 
							 runFileTrain='run_train.sh', runFileTest='run_test.sh', 
							 deviceId=0, repNum=None, isTest=False):
		'''
			modelDir   : The directory where model will be stored. 
			defFile    : The relative (to modelDir) path of architecture file.
			solverFile : The relative path of solver file
			logFile    : Relative path of log file 
			deviceId   : The GPU ID to be used.
								 	 If multiple GPUS are used make it a list
			repNum     : If none - then no repeats, otherwise use repeats. 
		'''
		self.modelDir_ = modelDir
		if not os.path.exists(self.modelDir_):
			os.makedirs(self.modelDir_)
		self.solver_   = ou.chunk_filename(os.path.join(self.modelDir_, solverFile))
		self.logTrain_ = ou.chunk_filename(os.path.join(self.modelDir_, logFileTrain))
		self.logTest_  = ou.chunk_filename(os.path.join(self.modelDir_, logFileTest))
		self.def_      = ou.chunk_filename(os.path.join(self.modelDir_, defFile))
		self.runTrain_ = ou.chunk_filename(os.path.join(self.modelDir_, runFileTrain))
		self.runTest_  = ou.chunk_filename(os.path.join(self.modelDir_, runFileTest))
		self.paths_    = get_caffe_paths()
		self.deviceId_ = deviceId
		if not(type(deviceId) == list):
			self.deviceId_ = [self.deviceId_]
		self.deviceStr_ = ''.join('%d,' % d for d in self.deviceId_)[0:-1]
		self.repNum_   = repNum
		self.isTest_   = isTest
		#To Prevent the results from getting lost I will copy over the log files
		#into a result folder. 
		self.resultDir_ = os.path.join(self.modelDir_,'result_store')
		if not os.path.exists(self.resultDir_):
			os.makedirs(self.resultDir_)
		self.resultLogTrain_ = ou.chunk_filename(os.path.join(self.resultDir_, logFileTrain))
		self.resultLogTest_  = ou.chunk_filename(os.path.join(self.resultDir_, logFileTest))
		self.isResume_       = False


	##
	# Write script for training.  
	def write_run_train(self, modelFile=None):
		with open(self.runTrain_,'w') as f:
			f.write('#!/usr/bin/env sh \n \n')
			f.write('TOOLS=%s \n \n' % self.paths_['tools'])
			f.write('GLOG_logtostderr=1 $TOOLS/caffe train')
			if self.isResume_:
				assert modelFile == None
				f.write('\t --snapshot=%s' % self.resumeSolver_)
			else:
				if modelFile is not None:
					f.write('\t --weights=%s' % modelFile)
			f.write('\t --solver=%s' % self.solver_)
			f.write('\t -gpu %s' % self.deviceStr_)
			f.write('\t 2>&1 | tee %s \n' % self.logTrain_)
		give_run_permissions_(self.runTrain_)

	##
	# Write test script
	def write_run_test(self, modelIterations, testIterations):
		'''
			modelIterations: Number of iterations of the modelFile. 
											 The modelFile Name is automatically extracted from the solver file.
			testIterations:  Number of iterations to use for testing.   
		'''
		snapshot = self.extract_snapshot_name() % modelIterations
		snapshot = ou.chunk_filename(snapshot)
		with open(self.runTest_,'w') as f:
			f.write('#!/usr/bin/env sh \n \n')
			f.write('TOOLS=%s \n \n' % self.paths_['tools'])
			f.write('GLOG_logtostderr=1 $TOOLS/caffe test')
			f.write('\t --weights=%s' % snapshot)
			f.write('\t --model=%s ' % self.def_)
			f.write('\t --iterations=%d' % testIterations)
			#For testing use one GPU only
			f.write('\t -gpu %d' % self.deviceId_[0])
			f.write('\t 2>&1 | tee %s \n' % self.logTest_)
		give_run_permissions_(self.runTest_)

	##
	# Initialiaze a solver from the file/SolverDef instance. 
	def init_solver_from_external(self, inFile):
		if isinstance(inFile, SolverDef):
			self.solDef_ = inFile
		else:
			self.solDef_ = SolverDef.from_file(inFile)
		self.solDef_.add_property('device_id', self.deviceId_)
		#Modify the name of the net
		self.solDef_.set_property('net', '"%s"' % self.def_)		

	##
	# Intialize the net definition from the file/ProtoDef Instance.
	def init_netdef_from_external(self, inFile):
		if isinstance(inFile, ProtoDef):
			self.netDef_ = inFile
		else: 
			self.netDef_ = ProtoDef(inFile)

	##
	# Write a solver file for making reps.
	def write_solver(self):
		'''
			Modifies the inFile to make it appropriate for running repeats
		'''
		if self.repNum_ is not None:
			#Modify snapshot name
			snapName   = self.solDef_.get_property('snapshot_prefix')
			snapName   = snapName[:-1] + '_rep%d"' % repNum
			snapName   = ou.chunk_filename(snapshot)
			self.solDef_.set_property('snapshot_prefix', snapName)
		
		self.solDef_.write(self.solver_)	

	##		
	def extract_snapshot_name(self):
		'''
			Find the name with which models are being stored. 
		'''
		snapshot   = self.solDef_.get_property('snapshot_prefix')
		#_iter_%d.caffemodel is added by caffe while snapshotting. 
		snapshot = snapshot[1:-1] + '_iter_%d.caffemodel'
		snapshot = ou.chunk_filename(snapshot)
		return snapshot

	##
	def write_netdef(self):
		self.netDef_.write(self.def_)

	##
	# Run the Experiment
	def run(self):
		cwd = os.getcwd()
		subprocess.check_call([('cd %s && ' % self.modelDir_) + self.runTrain_] ,shell=True)
		os.chdir(cwd)
		shutil.copyfile(self.logTrain_, self.resultLogTrain_)		
		if self.isTest_:
			subprocess.check_call([('cd %s && ' % self.modelDir_) + self.runTest_] ,shell=True)
			os.chdir(cwd)
			shutil.copyfile(self.logTest_, self.resultLogTest_)		

	def setup_resume(self, resumeIter):
		modelFile  = self.extract_snapshot_name() % resumeIter
		solverFile = modelname_2_solvername(modelFile)
		self.isResume_     = True
		self.resumeModel_  = modelFile	
		self.resumeSolver_ = solverFile
		if np.mod(resumeIter,1000)==0:
			resumeStr = '_resume%dK' % int(resumeIter/1000)
		else:
			resumeIter = '_resume%d' % resumeIter
		self.logTrain_       = self.logTrain_[:-4] + resumeStr + '.txt'
		self.resultLogTrain_ = self.resultLogTrain_[:-4] + resumeStr + '.txt'
		self.runTrain_       = self.runTrain_[:-3] + resumeStr + '.sh'


##
# Programatically make a Caffe Experiment. 
class CaffeExperiment:
	def __init__(self, dataExpName, caffeExpName, expDirPrefix, snapDirPrefix,
							 defPrefix = 'caffenet', solverPrefix = 'solver',
							 logPrefix = 'log', runPrefix = 'run', deviceId = 0,
							 repNum = None, isTest=False):
		'''
			experiment directory: expDirPrefix  + dataExpName
			snapshot   directory: snapDirPrefix + dataExpName
			solver     file     : expDir + solverPrefix + caffeExpName
			net-def    file     : expDir + defPrefix    + caffeExpName
			log        file     : expDir + logPrefix    + caffeExpName
			run        file     : expDir + runPrefix    + caffeExpName 
		'''
		self.dataExpName_  = dataExpName
		self.caffeExpName_ = caffeExpName
		#Relevant directories. 
		self.dirs_  = {}
		self.dirs_['exp']  = os.path.join(expDirPrefix,  dataExpName)
		self.dirs_['snap'] = os.path.join(snapDirPrefix, dataExpName)  

		#Relevant files.
		tmpDirName,_  = os.path.split(caffeExpName)
		if tmpDirName == '': 
			solverFile    = solverPrefix + '_' + caffeExpName + '.prototxt'
			defFile       = defPrefix    + '_' + caffeExpName + '.prototxt'
			defDeployFile = defPrefix    + '_' + caffeExpName + '_deploy.prototxt'
			logFile       = logPrefix + '_' + '%s' + '_' + caffeExpName + '.txt'
			runFile       = runPrefix + '_' + '%s' + '_' + caffeExpName + '.sh'
			snapPrefix    = defPrefix + '_' + caffeExpName 
		else:
			solverFile    = caffeExpName + '_' + solverPrefix + '.prototxt'
			defFile       = caffeExpName + '_' + defPrefix    + '.prototxt'
			defDeployFile = caffeExpName + '_' + defPrefix    + '_deploy.prototxt'
			logFile       = caffeExpName + '_' + '%s' + '_' + logPrefix + '.txt'
			runFile       = caffeExpName + '_' + '%s' + '_' + logPrefix + '.sh'
			snapPrefix    = caffeExpName + '_' + defPrefix 

		self.files_   = {}
		self.files_['solver'] = os.path.join(self.dirs_['exp'], solverFile) 
		self.files_['netdef'] = os.path.join(self.dirs_['exp'], defFile)
		self.files_['netdefDeploy'] = os.path.join(self.dirs_['exp'], defDeployFile) 
		self.files_['logTrain'] = os.path.join(self.dirs_['exp'], logFile % 'train')
		self.files_['logTest']  = os.path.join(self.dirs_['exp'], logFile % 'test')
		self.files_['runTrain'] = os.path.join(self.dirs_['exp'], runFile % 'train')
		self.files_['runTest']  = os.path.join(self.dirs_['exp'], runFile % 'test')

		#snapshot
		self.files_['snap'] = os.path.join(snapDirPrefix, dataExpName,
													snapPrefix + '_iter_%d.caffemodel')  
		self.snapPrefix_    = '"%s"' % os.path.join(snapDirPrefix, dataExpName, snapPrefix)		
		self.snapPrefix_    = ou.chunk_filename(self.snapPrefix_, maxLen=242)

		#Chunk all the filnames if needed
		for key in self.files_.keys():
			self.files_[key] = ou.chunk_filename(self.files_[key])

		#Setup the experiment files.
		self.expFile_ = ExperimentFiles(self.dirs_['exp'], defFile = defFile,
											solverFile = solverFile, 
											logFileTrain = logFile % 'train', logFileTest = logFile % 'test', 
											runFileTrain = runFile % 'train', runFileTest = runFile % 'test', 
											deviceId = deviceId, repNum = repNum, isTest=isTest)
		self.isTest_  = isTest
		self.net_     = None

	##
	#initalize from solver file/SolverDef and netdef file/ProtoDef
	def init_from_external(self, solverFile, netDefFile):
		self.expFile_.init_solver_from_external(solverFile)
		self.expFile_.init_netdef_from_external(netDefFile)	
		#Set the correct snapshot prefix. 
		self.expFile_.solDef_.set_property('snapshot_prefix', self.snapPrefix_)

	##
	# init from self
	def init_from_self(self):
		self.expFile_.init_solver_from_external(self.files_['solver'])
		self.expFile_.init_netdef_from_external(self.files_['netdef'])	

	##
	def del_layer(self, layerName):
		self.expFile_.netDef_.del_layer(layerName) 

	##
	def del_all_layers_above(self, layerName):
		self.expFile_.netDef_.del_all_layers_above(layerName)

	## Get layer property
	def get_layer_property(self, layerName, propName, **kwargs):
		return self.expFile_.netDef_.get_layer_property(layerName, propName, **kwargs)

	## Set the property. 	
	def set_layer_property(self, layerName, propName, value, **kwargs):
		self.expFile_.netDef_.set_layer_property(layerName, propName, value, **kwargs)

	##
	def add_layer(self, layerName, layer, phase):
		self.expFile_.netDef_.add_layer(layerName, layer, phase)
	##
	# Get the layerNames from the type of the layer. 
	def get_layernames_from_type(self, layerType, phase='TRAIN'):
		return self.expFile_.netDef_.get_layernames_from_type(layerType, phase=phase)

	##
	def get_snapshot_name(self, numIter=10000):
		snapName = self.expFile_.extract_snapshot_name() % numIter
		snapName = ou.chunk_filename(snapName)
		return snapName

	## Only finetune the layers that are above ( and including) layerName
	def finetune_above(self, layerName):
		self.expFile_.netDef_.set_no_learning_until(layerName)	

	## All layernames
	def get_all_layernames(self, phase='TRAIN'):
		return self.expFile_.netDef_.get_all_layernames(phase=phase)

	## Get the top name of the last layer
	def get_last_top_name(self):
		return self.expFile_.netDef_.get_last_top_name()

	## Set init std of all layers that are gaussian
	def set_std_gaussian_weight_init(self, stdVal):
		self.expFile_.netDef_.set_std_all(stdVal)

	##Setup the network
	def setup_net(self, **kwargs):
		if self.net_ is None:
			self.make(**kwargs)
			snapName  = self.get_snapshot_name(kwargs['modelIter'])
			self.net_ = mp.MyNet(self.files_['netdef'], snapName)

	##Get weights from a layer.
	def get_weights(self, layerName, **kwargs):
		self.setup_net(**kwargs)
		return self.net_.net.params[layerName][0].data 


	# Make the experiment. 
	def make(self, modelFile=None, writeTest=False, testIter=None, modelIter=None,
								 resumeIter=None):
		'''
			modelFile - file to finetune from if needed.
			writeTest - if the test file needs to be written. 
			if writeTest is True:
				testIter :  For number of iterations the test needs to be run.
				modelIter:  Used for estimating the model used for running the tests. 
			resumeIter: If the experiment needs to be resumed. 
		'''
		if not os.path.exists(self.dirs_['exp']):
			os.makedirs(self.dirs_['exp'])
		if not os.path.exists(self.dirs_['snap']):
			os.makedirs(self.dirs_['snap'])
		
		if resumeIter is not None:
			self.expFile_.setup_resume(resumeIter)	
			assert modelFile is None, "Model file cannot be specified with resume\
							just specify the number of iterations"
 
		self.expFile_.write_netdef()
		self.expFile_.write_solver()
		print "MODEL: %s" % modelFile
		self.expFile_.write_run_train(modelFile)
		if writeTest:
			assert testIter is not None and modelIter is not None, 'Missing variables'
			self.expFile_.write_run_test(modelIter, testIter)		

	## Make the deploy file. 
	def make_deploy(self, dataLayerNames, imSz, **kwargs):
		self.deployProto_ = ProtoDef.deploy_from_proto(self.expFile_.netDef_,
									 dataLayerNames=dataLayerNames, imSz=imSz, **kwargs)
		self.deployProto_.write(self.files_['netdefDeploy'])
	
	## Get the deploy proto
	def get_deploy_proto(self):	
		return self.deployProto_

	## Get deploy file
	def get_deploy_file(self):
		return self.files_['netdefDeploy']		

	##
	# Run the experiment
	def run(self):
		self.expFile_.run()
	
	def get_test_accuracy(self):
		return test_log2acc(self.files_['logTest'])


##
# Test a Caffe Made using a lmdb
class CaffeTest:
	##
	# Intialize using an instance of CaffeExperiment and a lmdb that needs
	# to be used for testing
	@classmethod
	def from_caffe_exp_lmdb(cls, caffeExp, lmdbTest, doubleDb=False, deviceId=0):
		self      = cls()
		self.exp_ = caffeExp
		self.db_  = mpio.DbReader(lmdbTest)
		self.device_ = deviceId
		self.ipMode_    = 'lmdb'
		self.testCount_ = 0
		return self

	##
	def setup_network(self, opNames, dataLayerNames=['data'], labelBlob=['label'], 
						 imH=128, imW=128, cropH=112, cropW=112, channels=3,
						 modelIterations=10000, 
						 batchSz=100, delLayers=['accuracy', 'loss'],
						 maxClassCount=None, maxLabel=None):
		'''
			This will simply store the gt and predicted labels
			opName         : The layer from which the predicted features need to be taken.
											 should be a list. 
			dataLayerName  : Layer to which feed the data
			labelBlob      : The name of the label blob
			modelIterations: The number of iterations for which caffe-model needs to be used. 
			maxClassCount  : If we want to test so that only maxClassCount examples of each
											 class are considered.
			maxLabel       : Used in conjuction with maxClassCount, labels are assumed to be from	
											 [0, maxLabel) 
		'''
		if not isinstance(opNames, list):
			opNames = [opNames]
		assert len(dataLayerNames)==1
		#Make the deploy file
		self.exp_.make_deploy(dataLayerNames = dataLayerNames, imSz = [[channels, cropH, cropW]], 
								batchSz = batchSz, delLayers = delLayers)
		self.netdef_ = self.exp_.get_deploy_file()
		#Name of the model
		self.model_  = self.exp_.get_snapshot_name(numIter=modelIterations)
		#Setup the network
		self.net_    = mp.MyNet(self.netdef_, self.model_, deviceId=self.device_,
											testMode=True)
		#Set the data pre-processing
		meanFile = self.exp_.get_layer_property(dataLayerNames[0], 'mean_file')
		if meanFile is not None:
			meanFile = meanFile[1:-1]
			print 'Using mean file: %s' % meanFile
		else:
			print 'No mean file found'

		scale = self.exp_.get_layer_property(dataLayerNames[0], 'scale')
		if scale is not None:
			scale = float(scale)
			print 'Setting scale as :%f' % scale
		else:
			scale = None
		self.net_.set_preprocess(ipName = dataLayerNames[0], isBlobFormat=True,
										imageDims = (imH, imW, channels),
										cropDims  = (cropH, cropW), chSwap=None,
										rawScale = scale, meanDat = meanFile)  
		self.ip_      = dataLayerNames[0]
		self.op_      = opNames
		self.batchSz_ = batchSz
	
		self.maxClsCount_ = maxClassCount
		if maxClassCount is not None:
			assert maxLabel is not None, 'Specify maxLabel'
			self.clsCount_  = np.zeros((maxLabel,))


	def get_data(self):
		'''
			Returns: data, label, isEnd
		'''
		if self.ipMode_ == 'lmdb':
			readFlag = True
			readCount = 0
			allData, allLabel = [], []
			#Read the data
			while readFlag:
				data, label = self.db_.read_next()
				if data is None:
					break
				if self.maxClsCount_ is not None:
					if self.clsCount_[label] < self.maxClsCount_:
						self.clsCount_[label] += 1
					else:
						continue	
				#print data.shape
				allData.append(data.reshape(((1,) + data.shape)))
				allLabel.append(label)
				readCount += 1
				if readCount == self.batchSz_:
					readFlag = False
			#Accumalate the data
			if len(allData) > 0:
				allData  = np.concatenate(allData)
				allLabel = np.array(allLabel) 					
		else:
			raise Exception('Data Model not recognized')

		if len(allData) == 0:
			return None, None, True
		elif len(allData) < self.batchSz_:
			return allData, allLabel, True
		else:
			return allData, allLabel, False

	## 
	# Run the test
	def run_test(self):
		self.gtLb_   = []
		self.pdFeat_ = []
		runFlag = True
		while runFlag:
			data, label, isEnd = self.get_data()
			if data is None:
					print 'data is none'
					break
			if isEnd:
				runFlag = False
			#print 'running: ', data.shape, label.shape
			#Get the features
			op = self.net_.forward_all(blobs=self.op_, **{self.ip_: data})
			#If there are multiple outputs - then concatenate them along the channel dim.
			opDat = []
			for i,key in enumerate(self.op_):
				if i==0:
					opDat = op[key]
				else:
					opDat = np.concatenate((opDat, op[key]), axis=1)
			self.pdFeat_.append(opDat.squeeze())		
			self.gtLb_.append(label.squeeze())
			self.testCount_ += len(label.squeeze())
		self.pdFeat_ = np.concatenate(self.pdFeat_)
		self.gtLb_   = np.concatenate(self.gtLb_)

	##
	# Compute Accuracy
	def compute_performance(self, accType='accClassMean'):
		'''
			accType: 
				accClassMean: Compute accuracy of each class and then take the mean
				acc         : Overall accuracy	
		'''
		if accType in ['accClassMean', 'acc']:
			self.pdLb_ = np.argmax(self.pdFeat_, axis=1)
			if accType == 'acc':
				tp        = np.sum(self.pdLb_ == self.gtLb_)
				self.acc_ = float(tp)/self.pdLb_.shape[0]
				return self.acc_
			else:
				cls   = np.unique(self.gtLb_)
				clAcc = []
				for cl in cls:
					tp = np.sum((self.pdLb_ == cl) & (self.gtLb_ == cl))
					N  = np.sum(self.gtLb_ == cl)
					clAcc.append(float(tp)/N)
				clAcc = np.array(clAcc)
				self.accClassMean_ = np.mean(clAcc)
				return self.accClassMean_  
		else:
			raise Exception('Accuracy type %s not recognized' % accType)	
	
	##
	# Save Accuracy
	def save_performance(self, accTypes, outFile):
		print outFile
		fid = h5.File(outFile, 'w')
		print "Saving accuracies over %d examples" % self.testCount_
		for key in accTypes:
			acc  = np.array([self.compute_performance(accType=key)]) 
			dat  = fid.create_dataset(key, acc.shape, dtype='f')
			dat[:] = acc[:]
			print "%s: %.3f" % (key, acc[:])
		fid.close()


	def close(self):
		self.db_.close()	

##
# Useful when some layers need to be deleted and output needs to be compared
# to some other output
class CaffeDebug:
	@classmethod
	def from_caffe_exp(cls, caffeExp, modelIterations, deviceId=0):
		self         = cls()
		self.exp_    = caffeExp
		self.device_ = deviceId
		self.modelIters_ = modelIterations
		self.setup_net()
		return self

	##
	def setup_net(self):
		self.model_  = self.exp_.get_snapshot_name(numIter=self.modelIters_)
		self.netdef_ = self.exp_.files_['netdef']
		#Setup the network
		self.net_    = mp.MyNet(self.netdef_, self.model_, deviceId=self.device_,
											testMode=True)

	##
	def set_debug_output_name(self, name):
		assert isinstance(name, list)
		assert len(name)==1
		self.op_ = name
		#self.exp_.del_all_layers_above(name)	

	## The data corresponding to the next batch
	def next(self):
		op = self.net_.forward_all(blobs=self.op_, noInputs=True)
		return op[self.op_[0]]	
	
 
def make_experiment_repeats(modelDir, defPrefix,
									 solverPrefix='solver', repNum=0, deviceId=0, suffix=None, 
										defData=None, testIterations=None, modelIterations=None):
	'''
		Used to run an experiment multiple times.
		This is useful for testing different random initializations. 
		repNum       : The repeat. 
		modelDir     : The directory containing the defFile and solver file
		defPrefix    : The prefix of the architecture prototxt file. 
		solverPrefix : The prefix of the solver file. 
		deviceId     : The GPU device to use.
		suffix       : If a suffix is present it is added to all the files.
								   For eg solver.prototxt will become, solver_suffix.prototxt
		defData      : None or Instance of class ProtoDef which defined a protoFile
		testIterations: Number of test iterations to run
		modelIterations: The number of iterations of training for which model should be loaded 
	'''
	assert os.path.exists(modelDir), 'ModelDir %s not found' % modelDir

	#Make the directory for storing rep data. 
	repDir          = modelDir + '_reps'
	if not os.path.exists(repDir):
		os.makedirs(repDir)
	#Make the definition file. 
	if suffix is not None:
		defFile    = defPrefix + '_' + suffix + '.prototxt'
		solverFile = solverPrefix + '_' + suffix + '.prototxt'
		repSuffix  = '_rep%d_%s' % (repNum, suffix) 
	else:
		defFile    = defPrefix + '.prototxt'
		solverFile = solverPrefix + '.prototxt' 
		repSuffix  = '_rep%d' % repNum
 
	#Get Solver File Name
	solRoot, solExt = os.path.splitext(solverFile)

	repSol   = solRoot + repSuffix + solExt
	repDef   = defPrefix + repSuffix + '.prototxt'
	repLog   = 'log%s.txt' % (repSuffix)
	repRun   = 'run%s.sh'  % (repSuffix)
	repLogTest   = 'log_test%s.txt' % (repSuffix)
	repRunTest   = 'run_test%s.sh'  % (repSuffix)
			
	#Training Phase
	trainExp = ExperimentFiles(modelDir=repDir, defFile=repDef, solverFile=repSol,
						 logFileTrain=repLog, runFileTrain=repRun, deviceId=deviceId, repNum=repNum)
	trainExp.init_solver_from_external(os.path.join(modelDir, solverFile))
	if defData is None:
		trainExp.init_netdef_from_external(os.path.join(modelDir, defFile))
	else:
		trainExp.init_netdef_from_external(defData)
	trainExp.write_solver()
	trainExp.write_netdef()
	trainExp.write_run_train()
	
	#Test Phase	
	if testIterations is not None:
		assert modelIterations is not None	 
		testExp      = ExperimentFiles(modelDir=repDir, defFile=repDef, solverFile=repSol,
									 logFileTest=repLogTest, runFileTest=repRunTest, deviceId=deviceId, repNum=repNum)
		testExp.write_run_test(modelIterations, testIterations)

	return trainExp, testExp
