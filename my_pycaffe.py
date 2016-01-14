## @package my_pycaffe
#  Major Wrappers.
#

import numpy as np
import h5py
import caffe
import pdb
import matplotlib.pyplot as plt
import os
from six import string_types
import copy
from easydict import EasyDict as edict 
import my_pycaffe_utils as mpu
import pickle
import collections as co

class layerSz:
	def __init__(self, stride, filterSz):
		self.imSzPrev = [] #We will assume square images for now
		self.stride   = stride #Stride with which filters are applied
		self.filterSz = filterSz #Size of filters. 
		self.stridePixPrev = [] #Stride in image pixels of the filters in the previous layers.
		self.pixelSzPrev   = [] #Size of the filters in the previous layers in the image space
		#To be computed
		self.pixelSz   = [] #the receptive field size of the filter in the original image.
		self.stridePix = [] #Stride of units in the image pixel domain.

	def prev_prms(self, prevLayer):
		self.set_prev_prms(prevLayer.stridePix, prevLayer.pixelSz)

	def set_prev_prms(self, stridePixPrev, pixelSzPrev):
		self.stridePixPrev = stridePixPrev
		self.pixelSzPrev   = pixelSzPrev

	def compute(self):
		self.pixelSz   = self.pixelSzPrev + (self.filterSz-1)*self.stridePixPrev	  
		self.stridePix = self.stride * self.stridePixPrev


## 
# Calculate the receptive field size and the stride of the Alex-Net
# Something
def calculate_size():
	conv1 = layerSz(4,11)
	conv1.set_prev_prms(1,1)
	conv1.compute()
	pool1 = layerSz(2,3)
	pool1.prev_prms(conv1)
	pool1.compute()

	conv2 = layerSz(1,5)
	conv2.prev_prms(pool1)
	conv2.compute()
	pool2 = layerSz(2,3)
	pool2.prev_prms(conv2)
	pool2.compute()

	conv3 = layerSz(1,3)
	conv3.prev_prms(pool2)
	conv3.compute()

	conv4 = layerSz(1,3)
	conv4.prev_prms(conv3)
	conv4.compute()

	conv5 = layerSz(1,3)
	conv5.prev_prms(conv4)
	conv5.compute()
	pool5 = layerSz(2,3)
	pool5.prev_prms(conv5)
	pool5.compute()

	print 'Pool1: Receptive: %d, Stride: %d ' % (pool1.pixelSz, pool1.stridePix)	
	print 'Pool2: Receptive: %d, Stride: %d ' % (pool2.pixelSz, pool2.stridePix)	
	print 'Conv3: Receptive: %d, Stride: %d ' % (conv3.pixelSz, conv3.stridePix)	
	print 'Conv4: Receptive: %d, Stride: %d ' % (conv4.pixelSz, conv4.stridePix)	
	print 'Pool5: Receptive: %d, Stride: %d ' % (pool5.pixelSz, pool5.stridePix)	
	
##
# Find the Layer Type
def find_layer(lines):
	layerName = []
	for l in lines:
		if 'type' in l:
			_,layerName = l.split()
			return layerName

##
# Find the Layer Name
def find_layer_name(lines):
	layerName = None
	topName   = None
	flagCount = 0
	firstL    = lines[0]
	assert firstL.split()[1] is '{', 'Something is wrong'
	brackCount = 1
	for l in lines[1:]:
		if '{' in l:
			brackCount  += 1
		if '}' in l:
			brackCount -= 1
		if brackCount ==0:
			break
		if 'name' in l and brackCount==1:
			flagCount += 1
			_,layerName = l.split()
			layerName   = layerName[1:-1]
		if 'top' in l and brackCount==1:
			flagCount += 1
			_,topName  = l.split()
			topName    = topName[1:-1]
	
	assert layerName is not None, 'no name of a layer found'		
	return layerName, topName


##
# Converts definition file of a network into siamese network. 
def netdef2siamese(defFile, outFile):
	outFid = open(outFile,'w')
	stream1, stream2 = [],[]
	with open(defFile,'r') as fid:
		lines     = fid.readlines()
		layerFlag = 0
		for (i,l) in enumerate(lines):
			#Indicates if the line has been added or not
			addFlag = False
			if 'layers' in l:
				layerName = find_layer(lines[i:])
				print layerName
				#Go in the state that this a useful layer to model. 
				if layerName in ['CONVOLUTION', 'INNER_PRODUCT']:
					layerFlag = 1
			
			#Manage the top, bottom and name for the two streams in case of a layer with params. 
			if 'bottom' in l or 'top' in l or 'name' in l:
				stream1.append(l)
				pre, suf = l.split()
				suf  = suf[0:-1] + '_p"'
				newL = pre + ' ' + suf + '\n'
				stream2.append(newL)
				addFlag = True

			#Store the name of the parameters	
			if layerFlag > 0 and 'name' in l:
				_,paramName = l.split()
				paramName   = paramName[1:-1]
			
			#Dont want to overcount '{' multiple times for the line 'layers {'	
			if (layerFlag > 0) and ('{' in l) and ('layers' not in l):
				layerFlag += 1	
			
			if '}' in l:
				print layerFlag
				layerFlag = layerFlag - 1
				#Before the ending the layer definition inlucde the param
				if layerFlag == 0:
					stream1.append('\t param: "%s" \n' % (paramName + '_w'))
					stream1.append('\t param: "%s" \n' % (paramName + '_b'))
					stream2.append('\t param: "%s" \n' % (paramName + '_w'))
					stream2.append('\t param: "%s" \n' % (paramName + '_b'))
						
			if not addFlag:
				stream1.append(l)
				stream2.append(l)

	#Write the first stream of the siamese net. 
	for l in stream1:
		outFid.write('%s' % l)

	#Write the second stream of the siamese net. 
	skipFlag  = False
	layerFlag = 0
	for (i,l) in enumerate(stream2):
		if 'layers' in l:
			layerName = find_layer(stream2[i:])
			#Skip writing the data layers in stream 2
			if layerName in ['DATA']:
				skipFlag  = True
				layerFlag = 1
		
		#Dont want to overcount '{' multiple times for the line 'layers {'	
		if layerFlag > 1 and '{' in l:
			layerFlag += 1	

		#Write to the out File	
		if not skipFlag:
			outFid.write('%s' % l)	
	
		if '}' in l:
			layerFlag = layerFlag - 1
			if layerFlag == 0:
				skipFlag = False
	outFid.close()


##
# Get Model and Mean file for a mdoel.  
#
#		the model file - the .caffemodel with the weights
#		the mean file of the imagenet data
def get_model_mean_file(netName='vgg'):
	modelDir = '/data1/pulkitag/caffe_models/'
	bvlcDir  = modelDir + 'bvlc_reference/'
	if netName  in ['alex', 'alex_deploy']:
		modelFile  = modelDir + 'caffe_imagenet_train_iter_310000'
		imMeanFile = modelDir + 'ilsvrc2012_mean.binaryproto'
	elif netName == 'bvlcAlexNet':
		modelFile  = bvlcDir + 'bvlc_reference_caffenet.caffemodel'
		imMeanFile = bvlcDir + 'imagenet_mean.binaryproto'  
	elif netName == 'vgg':
		modelFile    = '/data1/pulkitag/caffe_models/VGG_ILSVRC_19_layers.caffemodel'
		imMeanFile = '/data1/pulkitag/caffe_models/ilsvrc2012_mean.binaryproto'
	elif netName  == 'lenet':
		modelFile  = '/data1/pulkitag/mnist/snapshots/lenet_iter_20000.caffemodel'
		imMeanFile = None
	else:
		print 'netName not recognized'
		return

	return modelFile, imMeanFile
	

##
# Get the architecture definition file. 
def get_layer_def_files(netName='vgg', layerName='pool4'):
	'''
		Returns
			the architecture definition file of the network uptil layer layerName
	'''
	modelDir = '/data1/pulkitag/caffe_models/'
	bvlcDir  = modelDir + 'bvlc_reference/'
	if netName=='vgg':
		defFile = modelDir + 'layer_def_files/vgg_19_%s.prototxt' % layerName
	elif netName == 'alex_deploy':
		if layerName is not None:
			defFile = bvlcDir + 'caffenet_deploy_%s.prototxt' % layerName
		else:
			defFile = bvlcDir + 'caffenet_deploy.prototxt'
	else:
		print 'Cannont get files for networks other than VGG'
	return defFile	


##
# Get the shape of input blob from the defFile
def get_input_blob_shape(defFile):
	blobShape = []
	with open(defFile,'r') as f:
		lines  = f.readlines()
		ipMode = False
		for l in lines:
			if 'input:' in l:
				ipMode = True
			if ipMode and 'input_dim:' in l:
				ips = l.split()
				blobShape.append(int(ips[1]))
	return blobShape

				
def read_mean(protoFileName):
	'''
		Reads mean from the protoFile
	'''
	with open(protoFileName,'r') as fid:
		ss = fid.read()
		vec = caffe.io.caffe_pb2.BlobProto()
		vec.ParseFromString(ss)
		mn = caffe.io.blobproto_to_array(vec)
	mn = np.squeeze(mn)
	return mn


class MyNet:
	def __init__(self, defFile, modelFile=None, isGPU=True, testMode=True, deviceId=None):
		self.defFile_   = defFile
		self.modelFile_ = modelFile
		self.testMode_  = testMode
		self.set_mode(isGPU, deviceId=deviceId)
		self.setup_network()
		self.transformer = {}


	def setup_network(self):
		if self.testMode_:
			if not self.modelFile_ is None:
				self.net = caffe.Net(self.defFile_, self.modelFile_, caffe.TEST)
			else:
				self.net = caffe.Net(self.defFile_, caffe.TEST)
		else:
			if not self.modelFile_ is None:
				self.net = caffe.Net(self.defFile_, self.modelFile_, caffe.TRAIN)
			else:
				self.net = caffe.Net(self.defFile_, caffe.TRAIN)
		self.batchSz   = self.get_batchsz()


	def set_mode(self, isGPU=True, deviceId=None):
		if isGPU:
			caffe.set_mode_gpu()
		else:
			caffe.set_mode_cpu()
		if deviceId is not None:
			caffe.set_device(deviceId)

	
	def get_batchsz(self):
		if len(self.net.inputs) > 0:
			return self.net.blobs[self.net.inputs[0]].num
		else:
			return None

	def get_blob_shape(self, blobName):
		assert blobName in self.net.blobs.keys(), 'Blob Name is not present in the net'
		blob = self.net.blobs[blobName]
		return blob.num, blob.channels, blob.height, blob.width

	
	def set_preprocess(self, ipName='data',chSwap=(2,1,0), meanDat=None,
			imageDims=None, isBlobFormat=False, rawScale=None, cropDims=None,
			noTransform=False, numCh=3):
		'''
			isBlobFormat: if the images are already coming in blobFormat or not. 
			ipName    : the blob for which the pre-processing parameters need to be set. 
			meanDat   : the mean which needs to subtracted
			imageDims : the size of the images as H * W * K where K is the number of channels
			cropDims  : the size to which the image needs to be cropped. 
									if None - then it is automatically determined
									this behavior is undesirable for some deploy prototxts 
			noTransform: if no transform needs to be applied
			numCh      : number of channels
		'''
		if chSwap is not None:
			assert len(chSwap) == numCh, 'Number of channels mismatch'
		if noTransform:
			self.transformer[ipName] = None
			return
		self.transformer[ipName] = caffe.io.Transformer({ipName: self.net.blobs[ipName].data.shape})
		#Note blobFormat will be so used that finally the image will need to be flipped. 
		self.transformer[ipName].set_transpose(ipName, (2,0,1))	

		if isBlobFormat:
			assert chSwap is None, 'With Blob format chSwap should be none' 

		if chSwap is not None:
			#Required for eg RGB to BGR conversion.
			print (ipName, chSwap)
			self.transformer[ipName].set_channel_swap(ipName, chSwap)
	
		if rawScale is not None:
			self.transformer[ipName].set_raw_scale(ipName, rawScale)
	
		#Crop Dimensions
		ipDims            = np.array(self.net.blobs[ipName].data.shape)
		if cropDims is not None:
			assert len(cropDims)==2, 'Length of cropDims needs to be corrected'
			self.cropDims   = np.array(cropDims)
		else:
			self.cropDims     = ipDims[2:]
		self.isBlobFormat = isBlobFormat 
		if imageDims is None:
			imageDims = np.array([ipDims[2], ipDims[3], ipDims[1]])
		else:
			assert len(imageDims)==3
			imageDims = np.array([imageDims[0], imageDims[1], imageDims[2]])
		self.imageDims = imageDims
		self.get_crop_dims()		
	
		#Mean Subtraction
		if not meanDat is None:
			isTuple = False
			if isinstance(meanDat, string_types):
				meanDat = read_mean(meanDat)
			elif type(meanDat) ==  tuple:
				meanDat = np.array(meanDat).reshape(numCh,1,1)
				meanDat = meanDat * (np.ones((numCh, self.crop[2] - self.crop[0],\
									 self.crop[3]-self.crop[1])).astype(np.float32))
				isTuple = True
			_,h,w = meanDat.shape
			assert self.imageDims[0]<=h and self.imageDims[1]<=w,\
				 'imageDims must match mean Image size, (h,w), (imH, imW): (%d, %d), (%d,%d)'\
				 % (h,w,self.imageDims[0],self.imageDims[1])
			if not isTuple:
				meanDat  = meanDat[:, self.crop[0]:self.crop[2], self.crop[1]:self.crop[3]] 
			self.transformer[ipName].set_mean(ipName, meanDat)
	
	
	def get_crop_dims(self):
		# Take center crop.
		center = np.array(self.imageDims[0:2]) / 2.0
		crop = np.tile(center, (1, 2))[0] + np.concatenate([
				-self.cropDims / 2.0,
				self.cropDims / 2.0
		])
		self.crop = crop


	def resize_batch(self, ims):
		if ims.shape[0] > self.batchSz:
			assert False, "More input images than the batch sz"
		if ims.shape[0] == self.batchSz:
			return ims
		print "Adding Zero Images to fix the batch, Size: %d" % ims.shape[0]
		N,ch,h,w = ims.shape
		imZ      = np.zeros((self.batchSz - N, ch, h, w))
		ims      = np.concatenate((ims, imZ))
		return ims 
		

	def preprocess_batch(self, ims, ipName='data'):	
		'''
			ims: iterator over H * W * K sized images (K - number of channels) or K * H * W format. 
		'''
		#The image necessary needs to be float - otherwise caffe.io.resize fucks up.
		assert ipName in self.transformer.keys()
		ims = ims.astype(np.float32)
		if self.transformer[ipName] is None:
			ims = self.resize_batch(ims)
			return ims

		if np.max(ims)<=1.0:
			print "There maybe issues with image scaling. The maximum pixel value is 1.0 and not 255.0"
	
		im_ = np.zeros((len(ims), 
            self.imageDims[0], self.imageDims[1], self.imageDims[2]),
            dtype=np.float32)
		#Convert to normal image format if required. 
		if self.isBlobFormat:
			ims = np.transpose(ims, (0,2,3,1))
	
		#Resize the images
		h, w = ims.shape[1], ims.shape[2]
		for ix, in_ in enumerate(ims):
			if h==self.imageDims[0] and w==self.imageDims[1]:
				im_[ix] = np.copy(in_)
			else:
				#print (in_.shape, self.imageDims)
				im_[ix] = caffe.io.resize_image(in_, self.imageDims[0:2])

		#Required cropping
		im_ = im_[:,self.crop[0]:self.crop[2], self.crop[1]:self.crop[3],:]	
		#Applying the preprocessing
		caffe_in = np.zeros(np.array(im_.shape)[[0,3,1,2]], dtype=np.float32)
		for ix, in_ in enumerate(im_):
			caffe_in[ix] = self.transformer[ipName].preprocess(ipName, in_)

		#Resize the batch appropriately
		caffe_in = self.resize_batch(caffe_in)
		return caffe_in

	
	def deprocess_batch(self, caffeIn, ipName='data'):	
		'''
			ims: iterator over H * W * K sized images (K - number of channels)
		'''
		#Applying the deprocessing
		im_ = np.zeros(np.array(caffeIn.shape)[[0,2,3,1]], dtype=np.float32)
		for ix, in_ in enumerate(caffeIn):
			im_[ix] = self.transformer[ipName].deprocess(ipName, in_)

		ims = np.zeros((len(im_), 
            self.imageDims[0], self.imageDims[1], im_[0].shape[2]),
            dtype=np.float32)
		#Resize the images
		for ix, in_ in enumerate(im_):
			ims[ix] = caffe.io.resize_image(in_, self.imageDims)

		return ims


	##
	#Core function for running forward and backward passes. 
	def _run_forward_backward_all(self, runType, blobs=None, noInputs=False, diffs=None, **kwargs):
		'''
			runType : 'forward_all'
								'forward_backward_all'
			blobs   : The blobs to extract in the forward_all pass
			noInputs: Set to true when there are no input blobs. 
			diffs   : the blobs for which the gradient needs to be extracted. 
			kwargs  : A dictionary where each input blob has associated data
		'''
		if not noInputs:
			if kwargs:
				if (set(kwargs.keys()) != set(self.transformer.keys())):
					raise Exception('Data Transformer has not been set for all input blobs')
				#Just pass all the inputs
				procData = {}
				N        = self.batchSz
				for in_, data in kwargs.iteritems():
					N             = data.shape[0] #The first dimension must be equivalent of batchSz
					procData[in_] = self.preprocess_batch(data, ipName=in_)
				
				if runType == 'forward_all':
					ops = self.net.forward_all(blobs=blobs, **procData)
				elif runType == 'forward':
					ops = self.net.forward(blobs=blobs, **procData)
				elif runType == 'backward':
					ops = self.net.backward(diff=diff, **procData)
				elif runType == 'forward_backward_all':
					ops, opDiff = self.net.forward_backward_all(blobs=blobs, diffs=diffs, **procData)
					#Resize diffs in the right size
					for opd_, data in ops.iteritems():
						opDiff[opd_] = data[0:N]
				else:
					raise Exception('runType %s not recognized' % runType)
				#Resize data in the right size
				for op_, data in ops.iteritems():
					if data.ndim==0:
						continue
					#print (op_, data.shape)
					ops[op_] = data[0:N]
			else:
				raise Exception('No Input data specified.')
		else:
			if runType in ['forward_all', 'forward']:
				ops = self.net.forward(blobs=blobs)
			elif runType in ['backward']:	
				ops = self.net.backward(diffs=diffs)
			elif runType in ['forward_backward_all']:
				ops, opDiff = self.net.forward_backward_all(blobs=blobs, diffs=diffs)
			else:
				raise Exception('runType %s not recognized' % runType)

		if runType in ['forward', 'forward_all', 'backward']:
			return copy.deepcopy(ops) 
		else:
			return copy.deepcopy(ops), copy.deepcopy(opDiff)
	

	def forward_all(self, blobs=None, noInputs=False, **kwargs):
		'''
			See _run_forward_backward_all
		'''
		return self._run_forward_backward_all(runType='forward_all', blobs=blobs,
											 noInputs=noInputs, **kwargs)
	
		
	def forward_backward_all(self, blobs=None, noInputs=False, **kwargs):
		return self._run_forward_backward_all(runType='forward_backward_all', blobs=blobs,
											 noInputs=noInputs, **kwargs)

	
	def forward(self, blobs=None, noInputs=False, **kwargs):
		return self._run_forward_backward_all(runType='forward', blobs=blobs,
											 noInputs=noInputs, **kwargs)


	def backward(self, diffs=None, noInputs=False, **kwargs):
		return self._run_forward_backward_all(runType='backward', diffs=diffs,
											 noInputs=noInputs, **kwargs)


	def vis_weights(self, blobName, blobNum=0, ax=None, titleName=None, isFc=False,
						 h=None, w=None, returnData=False, chSt=0, chEn=3): 
		assert blobName in self.net.params, 'BlobName not found'
		dat  = copy.deepcopy(self.net.params[blobName][blobNum].data)
		if isFc:
			dat = dat.transpose((2,3,0,1))
			print dat.shape
			assert dat.shape[2]==1 and dat.shape[3]==1
			ch = dat.shape[1]
			assert np.sqrt(ch)*np.sqrt(ch)==ch, 'Cannot transform to filter'
			h,w = int(np.sqrt(ch)), int(np.sqrt(ch))
			dat = np.reshape(dat,(dat.shape[0],h,w,1))
			print dat.shape
			weights = vis_square(dat, ax=ax, titleName=titleName, returnData=returnData)	
		else:
			if h is None and w is None:
				weights = vis_square(dat.transpose(0,2,3,1), ax=ax, titleName=titleName,
									 returnData=returnData, chSt=chSt, chEn=chEn)	
			else:
				weights = vis_rect(dat.transpose(0,2,3,1), h, w, ax=ax, titleName=titleName, returnData=returnData)	

		if returnData:
			return weights


class SolverDebugStore(object):
	pass	

class MySolver(object):
	def __init__(self):
		self.solver_ = None
		self.phase_  = ['train', 'test']

	def __del__(self):
		del self.solver_
		del self.net_

	@classmethod
	def from_file(cls, solFile, recIter=20):
		'''
			solFile: solver prototxt from which to load the net
			recIter: the frequency of recording
		'''
		self = cls()
		self.solFile_    = solFile
		self.recIter_    = recIter
		self.setup_solver()
		self.plotSetup_  = False
		return self	

	##
	#setup the solver
	def setup_solver(self):
		self.solDef_  = mpu.SolverDef.from_file(self.solFile_)
		self.maxIter_ = self.solDef_.get_property('max_iter')
		if self.solDef_.has_property('solver_mode'):
			solverMode    = self.solDef_.get_property('solver_mode')
		else:
			solverMode    = 'GPU'
		if solverMode == 'GPU':
			if self.solDef_.has_property('device_id'):
				device = int(self.solDef_.get_property('device_id'))
			else:
				device = 0
			print ('GPU Mode, setting device %d' % device)
			caffe.set_device(device)
			caffe.set_mode_gpu()
		else:
			print ('CPU Mode')
			caffe.set_mode_cpu()
	
		self.solver_       = caffe.SGDSolver(self.solFile_)
		self.net_          = co.OrderedDict()
		self.net_[self.phase_[0]] = self.solver_.net 	
		self.net_        = self.solver_.net
		self.testNet_    = self.solver_.test_nets[0]
		if len(self.solver_.test_nets) > 1:
			print (' ##### WARNING - THERE ARE MORE THAN ONE TEST-NETS, FEATURE VALS
							FOR TEST NETS > 1 WILL NOT BE RECORDED #################')
			ip = raw_input('ARE YOU SURE YOU WANT TO CONTINUE(y/n)?')
			if ip == 'n':
				raise Exception('Quitting')
		self.layerNames_ = [l for l in self.net_._layer_names]
		self.paramNames_ = self.net_.params.keys()
		self.blobNames_  = self.net_.blobs.keys()
		
		#Storing the data
		self.featVals    = edict()
		self.paramVals   = [edict(), edict()]	
		self.paramUpdate = [edict(), edict()]	
		#Blobs
		for i,b in enumerate(self.blobNames_):
			self.featVals[b] = []
		#Params
		for p in self.paramNames_:
			self.paramVals[0][p] = []
			self.paramVals[1][p] = []
			self.paramUpdate[0][p] = []
			self.paramUpdate[1][p] = []

	##
	# Solve	
	def solve(self, numSteps=None):
		if numSteps is None:
			numSteps = self.maxIter_
		for i in range(numSteps):
			self.solver_.step(1)
			if np.mod(self.solver_.iter, self.recIter_)==1:
				self.record_feats_params()	

	##
	#Record the data
	def record_feats_params(self):
		for b in self.blobNames_:
			self.featVals[b].append(np.mean(np.abs(self.net_.blobs[b].data)))
		for p in self.paramNames_:
			for i in range(2):
				self.paramVals[i][p].append(np.mean(np.abs(self.net_.params[p][i].data)))	
				self.paramUpdate[i][p].append(np.mean(np.abs(self.net_.params[p][i].diff)))

	##
	#Dump the data to the file
	def dump_to_file(self, fName):
		data = co.OrderedDict()
		data['blobs']  = co.OrderedDict()
		for b in self.blobNames_:
			data['blobs'][b] = self.featVals[b]
		data['params']       = co.OrderedDict()
		data['paramsUpdate'] = co.OrderedDict()
		for p in self.paramNames_:
			data['params'][p]       = []
			data['paramsUpdate'][p] = []
			for i in range(2):
				data['params'][p].append(self.paramVals[i][p])	
				data['paramsUpdate'][p].append(self.paramVals[i][p])
		data['recIter'] = self.recIter_	
		pickle.dump(data, open(fName, 'w'))

	##
	#Read the logging data from file
	def read_from_file(self, fName):
		data = pickle.load(open(fName, 'r'))
		for k, b in enumerate(data['blobs'].keys()):
			self.featVals[b] = data['blobs'][b]
			assert b == self.blobNames_[k]
		for k, p in enumerate(data['params'].keys()):
			for i in range(2):
				self.paramVals[i][p]   = data['params'][p][i]
				self.paramUpdate[i][p] = data['paramsUpdate'][p][i] 
				assert p == self.paramNames_[k]

	##
	# Return pointer to layer
	def get_layer_pointer(self, layerName):
		assert layerName in self.layerNames_, 'layer not found'
		index = self.layerNames_.index(layerName)
		return self.net_.layers[index]

	##
	#Internal funciton for defining axes
	def _get_axes(self, titleNames, figTitle):
		numSub = np.ceil(np.sqrt(self.maxPerFigure_))
		N      = len(titleNames)
		allAx  = []
		count  = 0
		for fn in range(int(np.ceil(float(N)/self.maxPerFigure_))):
			#Given a figure
			fig = plt.figure()
			fig.suptitle(figTitle)
			ax = []
			en = min(N, count + self.maxPerFigure_)
			for i,tit in enumerate(titleNames[count:en]):
				ax.append((fig, fig.add_subplot(numSub, numSub, i+1)))
				ax[i][1].set_title(tit)
			count += self.maxPerFigure_
			allAx = allAx + ax
		return allAx

	##
	#Setup for plotting
	def setup_plots(self):
		plt.close('all')
		plt.ion()
		self.maxPerFigure_   = 16
		self.axBlobs_        = self._get_axes(self.blobNames_,  'Feature Values')
		self.axParamValW_    = self._get_axes(self.paramNames_, 'Parameter Values') 
		self.axParamDeltaW_  = self._get_axes(self.paramNames_, 'Parameter Updates') 
		self.plotSetup_      = True
	
	##
	#Plot the log
	def plot(self):
		if not self.plotSetup_:
			self.setup_plots()
		plt.ion()
		for i,bn in enumerate(self.blobNames_):
			fig, ax = self.axBlobs_[i]
			plt.figure(fig.number)
			ax.plot(range(len(self.featVals[bn])), self.featVals[bn])			
			plt.draw()
			plt.show()
		for i,pn in enumerate(self.paramNames_):
			#The parameters
			fig, ax = self.axParamValW_[i]
			plt.figure(fig.number)
			ax.plot(range(len(self.paramVals[0][pn])), self.paramVals[0][pn])			
			#The delta in parameters
			fig, ax = self.axParamDeltaW_[i]
			plt.figure(fig.number)
			ax.plot(range(len(self.paramUpdate[0][pn])), self.paramUpdate[0][pn])			
			plt.draw()
			plt.show()
	

##
# Visualize filters
def vis_square(data, padsize=1, padval=0, ax=None, titleName=None, returnData=False,
							chSt=0, chEn=3):
	'''
		data is numFitlers * height * width or numFilters * height * width * channels
	'''
	if data.ndim == 4:
		data = data[:,:,:,chSt:chEn]

	data -= data.min()
	data /= data.max()

	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
	data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

	# tile the filters into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

	if titleName is None:
		titleName = ''

	data = data.squeeze()
	if ax is not None:
		ax.imshow(data, interpolation='none')
		ax.set_title(titleName)
	else:
		plt.imshow(data, interpolation='none')
		plt.title(titleName)

	if returnData:
		return data


#Make rectangular filters
def vis_rect(data, h, w, padsize=1, padval=0, ax=None, titleName=None, returnData=False):
	'''
		data is numFitlers * height * width or numFilters * height * width * channels
	'''
	data -= data.min()
	data /= data.max()

	padding = ((0, h * w - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
	data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

	# tile the filters into an image
	data = data.reshape((h, w) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((h * data.shape[1], w * data.shape[3]) + data.shape[4:])

	if titleName is None:
		titleName = ''
	data = data.squeeze()
	if ax is not None:
		ax.imshow(data, interpolation='none')
		ax.set_title(titleName)
	else:
		plt.imshow(data)
		plt.title(titleName, interpolation='none')

	if returnData:
		return data



def setup_prototypical_network(netName='vgg', layerName='pool4'):
	'''
		Sets up a network in a configuration in which I commonly use it. 
	'''
	modelFile, meanFile = get_model_mean_file(netName)
	defFile             = get_layer_def_files(netName, layerName=layerName)
	meanDat             = read_mean(meanFile)
	net                 = MyNet(defFile, modelFile)
	net.set_preprocess(ipName='data', meanDat=meanDat, imageDims=(256,256,3))
	return net	


'''
def get_features(net, im, layerName=None, ipLayerName='data'):
	dataBlob = net.blobs['data']
	batchSz  = dataBlob.num
	assert im.ndim == 4
	N,nc,h,w = im.shape
	assert h == dataBlob.height and w==dataBlob.width

	if not layerName==None:
		assert layerName in net.blobs.keys()
		layerName = [layerName]
		outName   = layerName[0]	
	else:
		outName   = net.outputs[0]
		layerName = []
	
	print layerName
	imBatch = np.zeros((batchSz,nc,h,w))
	outFeats = {}
	outBlob  = net.blobs[outName]
	outFeats = np.zeros((N, outBlob.channels, outBlob.height, outBlob.width))

	for i in range(0,N,batchSz):
		st = i
		en = min(N, st + batchSz)
		l  = en - st
		imBatch[0:l,:,:,:] = np.copy(im[st:en])
		dataLayer = {ipLayerName:imBatch}
		feats = net.forward(blobs=layerName, start=None, end=None, **dataLayer)		 
		outFeats[st:en] = feats[outName][0:l]

	return outFeats

'''


def compute_error(gtLabels, prLabels, errType='classify'):
	N, lblSz = gtLabels.shape
	res = []
	assert prLabels.shape[0] == N and prLabels.shape[1] == lblSz
	if errType == 'classify':
		assert lblSz == 1
		cls     = np.unique(gtLabels)
		cls     = np.sort(cls)
		nCl     = cls.shape[0]
		confMat = np.zeros((nCl, nCl)) 
		for i in range(nCl):
			for j in range(nCl):
				confMat[i,j] = float(np.sum(np.bitwise_and((gtLabels == cls[i]),(prLabels == cls[j]))))/(np.sum(gtLabels == cls[i]))
		res = confMat
	else:
		print "Error type not recognized"
		raise
	return res	


def feats_2_labels(feats, lblType, maskLastLabel=False):
	#feats are assumed to be numEx * featDims
	labels = []
	if lblType in ['uniform20', 'kmedoids30_20']:
		r,c = feats.shape
		if maskLastLabel:
			feats = feats[0:r,0:c-1]
		labels = np.argmax(feats, axis=1)
		labels = labels.reshape((r,1))
	else:
		print "UNrecognized lblType"
		raise
	return labels


def save_images(ims, gtLb, pdLb, svFileStr, stCount=0, isSiamese=False):
	'''
		Saves the images
		ims: N * nCh * H * W 
		gtLb: Ground Truth Label
		pdLb: Predicted Label
		svFileStr: Path should contain (%s, %d) - which will be filled in by correct/incorrect and count
	'''
	N = ims.shape[0]
	ims = ims.transpose((0,2,3,1))
	fig = plt.figure()
	for i in range(N):
		im  = ims[i]
		plt.title('Gt-Label: %d, Predicted-Label: %d' %(gtLb[i], pdLb[i]))
		gl, pl = gtLb[i], pdLb[i]
		if gl==pl:
			fStr = 'correct'
		else:
			fStr = 'mistake'
		if isSiamese:
			im1  = im[:,:,0:3]
			im2  = im[:,:,3:]
			im1  = im1[:,:,[2,1,0]]
			im2  = im2[:,:,[2,1,0]]
			plt.subplot(1,2,1)
			plt.imshow(im1)
			plt.subplot(1,2,2)
			plt.imshow(im2)
			fName = svFileStr % (fStr, i + stCount)
			if not os.path.exists(os.path.dirname(fName)):
				os.makedirs(os.path.dirname(fName))
			print fName
			plt.savefig(fName)
		

def test_network_siamese_h5(imH5File=[], lbH5File=[], netFile=[], defFile=[], imSz=128, cropSz=112, nCh=3, outLblSz=1, meanFile=[], ipLayerName='data', lblType='uniform20',outFeatSz=20, maskLastLabel=False, db=None, svImg=False, svImFileStr=None, deviceId=None):
	'''
		defFile: Architecture prototxt
		netFile : The model weights
		maskLastLabel: In some cases it is we may need to compute the error bt ignoring the last label
									 for example in det - where the last class might be the backgroud class
		db: instead of h5File, provide a dbReader 
	'''
	isBlobFormat = True
	if db is None:
		isBlobFormat = False
		print imH5File, lbH5File
		imFid = h5py.File(imH5File,'r')
		lbFid = h5py.File(lbH5File,'r')
		ims1 = imFid['images1/']
		ims2 = imFid['images2/']
		lbls = lbFid['labels/']
		
		#Get Sizes
		imSzSq = imSz * imSz
		assert(ims1.shape[0] % imSzSq == 0 and ims2.shape[0] % imSzSq ==0)
		N     = ims1.shape[0]/(imSzSq * nCh)
		assert(lbls.shape[0] % N == 0)
		lblSz = outLblSz

	#Get the mean
	imMean = []
	if not meanFile == []:
		imMean = read_mean(meanFile)	

	#Initialize network
	net  = MyNet(defFile, netFile, deviceId=deviceId)
	net.set_preprocess(chSwap=None, meanDat=imMean,imageDims=(imSz, imSz, 2*nCh), isBlobFormat=isBlobFormat, ipName='data')
	
	#Initialize variables
	batchSz  = net.get_batchsz()
	ims      = np.zeros((batchSz, 2 * nCh, imSz, imSz))
	count    = 0
	imCount  = 0

	if db is None:	
		labels   = np.zeros((N, lblSz))
		gtLabels = np.zeros((N, lblSz)) 
		#Loop through the images
		for i in np.arange(0,N,batchSz):
			st = i * nCh * imSzSq 
			en = min(N, i + batchSz) * nCh * imSzSq
			numIm = min(N, i + batchSz) - i
			ims[0:batchSz] = 0
			ims[0:numIm,0:nCh,:,:] = ims1[st:en].reshape((numIm,nCh,imSz,imSz))
			ims[0:numIm,nCh:2*nCh,:,:] = ims2[st:en].reshape((numIm,nCh,imSz,imSz))
			imsPrep   = prepare_image(ims, cropSz, imMean)  
			predFeat  = get_features(net, imsPrep, ipLayerName=ipLayerName)
			predFeat  = predFeat[0:numIm]
			print numIm
			try:
				labels[i : i + numIm, :]    = feats_2_labels(predFeat.reshape((numIm,outFeatSz)), lblType, maskLastLabel=maskLastLabel)[0:numIm]
				gtLabels[i : i + numIm, : ] = (lbls[i * lblSz : (i+numIm) * lblSz]).reshape(numIm, lblSz) 
			except ValueError:
				print "Value Error found"
				pdb.set_trace()
	else:
		labels, gtLabels = [], []
		runFlag = True
		while runFlag:
			count = count + 1
			print "Processing Batch: ", count
			dat, lbl = db.read_batch(batchSz)
			N        = dat.shape[0]
			print N
			if N < batchSz:
				runFlag = False
			batchDat  = net.preprocess_batch(dat, ipName='data')
			dataLayer = {}
			dataLayer[ipLayerName] = batchDat
			feats     = net.net.forward(**dataLayer)
			feats     = feats[feats.keys()[0]][0:N]	
			gtLabels.append(lbl)	
			predLabels = feats_2_labels(feats.reshape((N,outFeatSz)), lblType)
			labels.append(predLabels)
			if svImg:
				save_images(dat, lbl, predLabels, svImFileStr, stCount=imCount, isSiamese=True)
			imCount = imCount + N
		labels   = np.concatenate(labels)
		gtLabels = np.concatenate(gtLabels) 
		
	confMat = compute_error(gtLabels, labels, 'classify')
	return confMat, labels, gtLabels	


def read_mean_txt(fileName):
	with open(fileName,'r') as f:
		l = f.readlines()
		mn = [float(i) for i in l]
		mn = np.array(mn)
	return mn
