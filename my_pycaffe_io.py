## @package my_pycaffe_io 
#  IO operations. 
#

import h5py as h5
import numpy as np
import my_pycaffe as mp
import caffe
import pdb
import os
import lmdb
import shutil
import scipy.misc as scm
import scipy.io as sio
import copy
from pycaffe_config import cfg

if not cfg.IS_EC2:
	#import matlab.engine as men
	MATLAB_PATH = '/work4/pulkitag-code/pkgs/caffe-v2-2/matlab/caffe'
else:
	MATLAB_PATH = ''


## 
# Write array as a proto
def  write_proto(arr, outFile):
	'''
		Writes the array as a protofile
	'''
	blobProto = caffe.io.array_to_blobproto(arr)
	ss        = blobProto.SerializeToString()
	fid       = open(outFile,'w')
	fid.write(ss)
	fid.close()


##
# Convert the mean to be useful for siamese network. 
def mean2siamese_mean(inFile, outFile, isGray=False):
	mn = mp.read_mean(inFile)
	if isGray:
		mn = mn.reshape((1,mn.shape[0],mn.shape[1]))
	mn    = np.concatenate((mn, mn))
	dType = mn.dtype 
	mn = mn.reshape((1, mn.shape[0], mn.shape[1], mn.shape[2]))
	print "New mean shape: ", mn.shape, dType
	write_proto(mn, outFile)

##
# Convert the siamese mean to be the mean
def siamese_mean2mean(inFile, outFile):
	assert not os.path.exists(outFile), '%s already exists' % outFile
	mn = mp.read_mean(inFile)
	ch = mn.shape[0]
	assert np.mod(ch,2)==0
	ch = ch / 2
	print "New number of channels: %d" % ch
	newMn = mn[0:ch].reshape(1,ch,mn.shape[1],mn.shape[2])
	write_proto(newMn.astype(mn.dtype), outFile)

##
# Convert to grayscale, mimics the matlab function
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

##
# Convert the mean grayscale mean
def mean2graymean(inFile, outFile):
	assert not os.path.exists(outFile), '%s already exists' % outFile
	mn    = mp.read_mean(inFile)
	dType = mn.dtype
	ch = mn.shape[0]
	assert ch==3
	mn = rgb2gray(mn.transpose((1,2,0))).reshape((1,1,mn.shape[1],mn.shape[2]))
	print "New mean shape: ", mn.shape, dType
	write_proto(mn.astype(dType), outFile)

		
##
# Resize the mean to a different size
def resize_mean(inFile, outFile, imSz):
	mn = mp.read_mean(inFile)
	dType = mn.dtype
	ch, rows, cols = mn.shape
	mn = mn.transpose((1,2,0))
	mn = scm.imresize(mn, (imSz, imSz)).transpose((2,0,1)).reshape((1,ch,imSz,imSz))
	write_proto(mn.astype(dType), outFile)


def ims2hdf5(im, labels, batchSz, batchPath, isColor=True, batchStNum=1, isUInt8=True, scale=None, newLabels=False):
	'''
		Converts an image dataset into hdf5
	'''
	h5SrcFile = os.path.join(batchPath, 'h5source.txt')
	strFid    = open(h5SrcFile, 'w')

	dType = im.dtype
	if isUInt8:
		assert im.dtype==np.uint8, 'Images should be in uint8'
		h5DType = 'u1'
	else:
		assert im.dtype==np.float32, 'Images can either be uint8 or float32'		
		h5DType = 'f'

	if scale is not None:
		im = im * scale

	if isColor:
		assert im.ndim ==4 
		N,ch,h,w = im.shape
		assert ch==3, 'Color images must have 3 channels'
	else:
		assert im.ndim ==3
		N,h,w    = im.shape
		im       = np.reshape(im,(N,1,h,w))
		ch       = 1

	count = batchStNum
	for i in range(0,N,batchSz):
		st      = i
		en      = min(N, st + batchSz)
		if st + batchSz > N:
			break
		h5File    = os.path.join(batchPath, 'batch%d.h5' % count)
		h5Fid     = h5.File(h5File, 'w')
		imBatch = np.zeros((N, ch, h, w), dType) 
		imH5      = h5Fid.create_dataset('/data',(batchSz, ch, h, w), dtype=h5DType)
		imH5[0:batchSz] = im[st:en]
		if newLabels:
			lbH5 = h5Fid.create_dataset('/label', (batchSz,), dtype='f')
			lbH5[0:batchSz] = labels[st:en].reshape((batchSz,))
		else: 
			lbH5 = h5Fid.create_dataset('/label', (batchSz,1,1,1), dtype='f')
			lbH5[0:batchSz] = labels[st:en].reshape((batchSz,1,1,1))
		h5Fid.close()
		strFid.write('%s \n' % h5File)
		count += 1	
	strFid.close()


class DbSaver:
	def __init__(self, dbName, isLMDB=True):
		if os.path.exists(dbName):
			print "%s already existed, but not anymore ..removing.." % dbName
			shutil.rmtree(dbName)
		self.db    = lmdb.open(dbName, map_size=int(1e12))
		self.count = 0

	def __del__(self):
		self.db.close()

	def add_batch(self, ims, labels=None, imAsFloat=False, svIdx=None):
		'''
			Assumes ims are numEx * ch * h * w
			svIdx: Allows one to store the images randomly. 
		'''
		self.txn   = self.db.begin(write=True)
		if labels is not None:
			assert labels.dtype == np.int or labels.dtype==np.long
		else:
			N      = ims.shape[0]
			labels = np.zeros((N,)).astype(np.int)

		if svIdx is not None:
			itrtr = zip(svIdx, ims, labels)
		else:
			itrtr = zip(range(self.count, self.count + ims.shape[0]), ims, labels)

		#print svIdx.shape, ims.shape, labels.shape
		for idx, im, lb in itrtr:
			if not imAsFloat:
				im    = im.astype(np.uint8)
			imDat = caffe.io.array_to_datum(im, label=lb)
			aa    = imDat.SerializeToString()
			self.txn.put('{:0>10d}'.format(idx), imDat.SerializeToString())
		self.txn.commit()
		self.count = self.count + ims.shape[0]

	def close(self):
		self.db.close()



class DoubleDbSaver:
	'''
		Useful for example when storing images and labels in two different dbs
	'''
	def __init__(self, dbName1, dbName2, isLMDB=True):
		self.dbs_ = []
		self.dbs_.append(DbSaver(dbName1, isLMDB=isLMDB))
		self.dbs_.append(DbSaver(dbName2, isLMDB=isLMDB))

	def __del__(self):
		for db in self.dbs_:
			db.__del__()

	def close(self):
		for db in self.dbs_:
			db.close()

	def add_batch(self, ims, labels=(None,None), imAsFloat=(False,False), svIdx=(None,None)):
		for (i,db) in enumerate(self.dbs_):
			im = ims[i]
			db.add_batch(ims[i], labels[i], imAsFloat=imAsFloat[i], svIdx=svIdx[i])	


class DbReader:
	def __init__(self, dbName, isLMDB=True, readahead=True):
		#For large LMDB set readahead to be False
		self.db_     = lmdb.open(dbName, readonly=True, readahead=readahead)
		self.txn_    = self.db_.begin(write=False) 
		self.cursor_ = self.txn_.cursor()		
		self.nextValid_ = True
		self.cursor_.first()

	def __del__(self):
		self.txn_.commit()
		self.db_.close()
		
	def read_next(self):
		if not self.nextValid_:
			return None, None
		else:
			key, dat = self.cursor_.item()
			datum  = caffe.io.caffe_pb2.Datum()
			datStr = datum.FromString(dat)
			data   = caffe.io.datum_to_array(datStr)
			label  = datStr.label
		self.nextValid_ = self.cursor_.next()
		return data, label

	def read_batch(self, batchSz):
		data, label = [], []
		count = 0
		for b in range(batchSz):
			dat, lb = self.read_next()
			if dat is None:
				break
			else:
				count += 1
				ch, h, w = dat.shape
				dat = np.reshape(dat,(1,ch,h,w))
				data.append(dat)
				label.append(lb)
		if count > 0:
			data  = np.concatenate(data[:])
			label = np.array(label)
			label = label.reshape((len(label),1))
		else:
			data, label = None, None
		return data, label 
		 
	def get_label_stats(self, maxLabels):
		countArr  = np.zeros((maxLabels,))
		countFlag = True
		while countFlag:
			_,lb     = self.read_next()	
			if lb is not None:
				countArr[lb] += 1
			else:
				countFlag = False
		return countArr				

	def get_count(self):
		return self.db_.stat()['entries']
	
	def close(self):
		self.txn_.commit()
		self.db_.close()


class SiameseDbReader(DbReader):
	def get_next_pair(self, flipColor=True):
		imDat,label  = self.read_next()
		ch,h,w = imDat.shape
		assert np.mod(ch,2)==0
		ch = ch / 2
		imDat  = np.transpose(imDat,(1,2,0))
		im1    = imDat[:,:,0:ch]
		im2    = imDat[:,:,ch:2*ch]
		if flipColor:
			im1 = im1[:,:,[2,1,0]]
			im2 = im2[:,:,[2,1,0]]
		return im1, im2, label
			 
##
# Read two LMDBs simultaneosuly
class DoubleDbReader:
	def __init__(self, dbNames, isLMDB=True, readahead=True):
		#For large LMDB set readahead to be False
		self.dbs_ = []
		for d in dbNames:
			self.dbs_.append(DbReader(d, isLMDB=isLMDB, readahead=readahead))	

	def __del__(self):
		for db in self.dbs_:
			db.__del__()
		
	def read_next(self):
		data = []
		for db in self.dbs_:
			dat,_ = db.read_next()
			data.append(dat)
		return data[0], data[1]

	def read_batch(self, batchSz):
		data = []
		for db in self.dbs_:
			dat,_ = db.read_batch(batchSz)
			data.append(dat)
		return data[0], data[1]

	def read_batch_data_label(self, batchSz):
		data, label = [], []
		for db in self.dbs_:
			dat,lb = db.read_batch(batchSz)
			data.append(dat)
			label.append(lb)
		return data[0], data[1], label[0], label[1]

	def close(self):
		for db in self.dbs_:
			db.close()

##
# For reading generic window reader. 
class GenericWindowReader:
	def __init__(self, fileName):
		self.fid_ = open(fileName,'r')
		line      = self.fid_.readline()
		assert(line.split()[1] == 'GenericDataLayer')
		self.num_   = int(self.fid_.readline())
		self.numIm_ = int(self.fid_.readline())
		self.lblSz_ = int(self.fid_.readline()) 
		self.count_ = 0
		self.open_  = True

	def read_next(self):
		if self.count_ == self.num_:
			print "All lines already read"
			return None, None
		count = int(self.fid_.readline().split()[1])
		assert count == self.count_
		self.count_ += 1
		imDat = []
		for n in range(self.numIm_): 
			imDat.append(self.fid_.readline())
		lbls = self.fid_.readline().split()
		lbls = np.array([float(l) for l in lbls]).reshape(1,self.lblSz_)
		return imDat, lbls
				
	def get_all_labels(self):
		readFlag = True
		lbls     = []
		while readFlag:
			_, lbl = self.read_next()
			if lbl is None:
				readFlag = False
				continue
			else:
				lbls.append(lbl)
		lbls = np.concatenate(lbls)
		return lbls
		
	def is_open(self):
		return self.open_

	def is_eof(self):
		return self.count_ >= self.num_

	def close(self):
		self.fid_.close()
		self.open_ = False

##
# For writing generic window file layers. 
class GenericWindowWriter:
	def __init__(self, fileName, numEx, numImgPerEx, lblSz):
		'''
			fileName   : the file to write to.
			numEx      : the number of examples
			numImgPerEx: the number of images per example
			lblSz      : the size of the labels 
		'''
		self.file_  = fileName
		self.num_   = numEx
		self.numIm_ = numImgPerEx
		self.lblSz_ = lblSz 
		self.count_ = 0 #The number of examples written. 

		dirName = os.path.dirname(fileName)
		if len(dirName) >0 and not os.path.exists(dirName):
			os.makedirs(dirName)

		self.fid_ = open(self.file_, 'w')	
		self.fid_.write('# GenericDataLayer\n')
		self.fid_.write('%d\n' % self.num_) #Num Examples. 
		self.fid_.write('%d\n' % self.numIm_) #Num Images per Example. 
		self.fid_.write('%d\n' % self.lblSz_) #Num	Labels

	##
	# Private Helper function for writing the images for the WindowFile
	def write_image_line_(self, imgName, imgSz, bbox):
		'''
			imgSz: channels * height * width
			bbox : x1, y1, x2, y2
		'''
		ch, h, w = imgSz
		x1,y1,x2,y2 = bbox
		x1  = max(0, x1)
		y1  = max(0, y1)
		x2  = min(x2, w-1)
		y2  = min(y2, h-1)
		self.fid_.write('%s %d %d %d %d %d %d %d\n' % (imgName, 
							ch, h, w, x1, y1, x2, y2))

	##
	def write(self, lbl, *args):
		assert len(args)==self.numIm_, 'Wrong input arguments: (%d v/s %d)' % (len(args),self.numIm_)
		self.fid_.write('# %d\n' % self.count_)
		#Write the images
		for arg in args:
			if type(arg)==str:
				#Assuming arg is the imageline read from another window-file
				#and the last character in the str is \n
				self.fid_.write(arg)
			else:
				imName, imSz, bbox = arg
				self.write_image_line_(imName, imSz, bbox)	
		
		#Write the label
		lbStr = ['%f '] * self.lblSz_
		lbStr = ''.join(lbS % lb for (lb, lbS) in zip(lbl, lbStr))
		lbStr = lbStr[:-1] + '\n'
		self.fid_.write(lbStr)
		self.count_ += 1

		if self.count_ == self.num_:
			self.close()	
	##
	def close(self):
		self.fid_.close()


##
# For writing sqbox window file layers. 
class SqBoxWindowWriter:
	def __init__(self, fileName, numEx):
		'''
			fileName   : the file to write to.
			numEx      : the number of examples
			The format
			# ExNum
			IMG_NAME IMG_SZ
			NUM_OBJ
			OBJ1_X OBJ1_Y BBOX1_X BBOX1_Y BBOX1_SZ
			..
			.
			# ExNum
			..
			.
			- x1, y1 for object position
			- xc, yc for the center of desired bbox
			- sqSz the length of the desired bbox
			express sqSz as the ratio of the largest imgSz / sqSz 
		'''
		self.file_  = fileName
		self.num_   = numEx
		self.count_ = 0 #The number of examples written. 

		dirName = os.path.dirname(fileName)
		if not os.path.exists(dirName):
			os.makedirs(dirName)

		self.fid_ = open(self.file_, 'w')	
		self.fid_.write('# SqBoxWindowDataLayer\n')
		self.fid_.write('%d\n' % self.num_) #Num Examples. 

	##
	# Private Helper function for writing the images for the WindowFile
	def write_image_line_(self, imgName, imgSz, numObj):
		'''
			imgSz : channels * height * width
			numObj: number of objects in the image
		'''
		ch, h, w = imgSz
		self.fid_.write('%d\n' % numObj)
		self.fid_.write('%s %d %d %d\n' % (imgName, 
							ch, h, w))

	##
	def write(self, *args):
		self.fid_.write('# %d\n' % self.count_)
		#Write the images
		imName, imSz, objPos, bboxPos, bboxSz = args
		numObj = len(objPos)
		self.write_image_line_(imName, imSz, numObj)
		for i in range(numObj):
			xObjPos, yObjPos = objPos[i]
			xBbxPos, yBbxPos = bboxPos[i]
			self.fid_.write('%d %d %d %d %d\n' % (xObjPos, yObjPos, xBbxPos, yBbxPos, bboxSz[i]))		
		self.count_ += 1
		if self.count_ == self.num_:
			self.close()	

	##
	def close(self):
		self.fid_.close()

##
# For reading generic window reader. 
class SqBoxWindowReader:
	def __init__(self, fileName):
		self.fid_ = open(fileName,'r')
		line      = self.fid_.readline()
		assert(line.split()[1] == 'SqBoxWindowDataLayer')
		self.num_   = int(self.fid_.readline())
		self.count_ = 0

	def read_next(self):
		if self.count_ == self.num_:
			print "All lines already read"
			return None, None
		count = int(self.fid_.readline().split()[1])
		assert count == self.count_
		self.count_ += 1
		#The number of boxes in the image
		numBox = int(self.fid_.readline())
		imName.append(self.fid_.readline())
		#Read all the boxes
		objPos, bbxPos, bbxSz = [], [], []
		for n in range(numBox):
			lbls = self.fid_.readline().split()
			lbls = [int(l) for l in lbls]
			ox, oy, bx, by, bs = lbls
			objPos.append([ox, oy])
			bbxPos.append([bx, by])
			bbxSz.append(bs)
		return imName, objPos, bbxPos, bbxSz
				
	def get_all_labels(self):
		readFlag = True
		lbls     = []
		while readFlag:
			_, lbl = self.read_next()
			if lbl is None:
				readFlag = False
				continue
			else:
				lbls.append(lbl)
		lbls = np.concatenate(lbls)
		return lbls
		
	def close(self):
		self.fid_.close()



	
def save_lmdb_images(ims, dbFileName, labels=None, asFloat=False):
	'''
		Assumes ims are numEx * ch * h * w
	'''
	N,_,_,_ = ims.shape
	if labels is not None:
		assert labels.dtype == np.int or labels.dtype==np.long
	else:
		labels = np.zeros((N,)).astype(np.int)

	db = lmdb.open(dbFileName, map_size=int(1e12))
	with db.begin(write=True) as txn:
		for (idx, im) in enumerate(ims):
			if not asFloat:
				im    = im.astype(np.uint8)
			imDat = caffe.io.array_to_datum(im, label=labels[idx])
			txn.put('{:0>10d}'.format(idx), imDat.SerializeToString())
	db.close()

##
# Save the weights in a form that will be used by matlab function
# swap_weights to generate files useful for matconvnet. 
def save_weights_for_matconvnet(net, outName, matlabRefFile=None):
	'''
		net    : Instance of my_pycaffe.MyNet
		outName: The matlab file which needs to store parameters. 
	'''
	params = {}
	for (count,key) in enumerate(net.net.params.keys()):
		blob = net.net.params[key]
		wKey = key + '_w'
		bKey = key + '_b'
		params[wKey] = copy.deepcopy(blob[0].data)
		params[bKey]    = copy.deepcopy(blob[1].data)
		N,ch,h,w     = params[wKey].shape
		print params[wKey].shape, params[bKey].shape, N
		num = N * ch * h * w
		if count==0:
			print 'Converting BGR filters to RGB filters'
			assert ch==3, 'The code is hacked as MatConvNet works with RGB format instead of BGR'
			params[wKey] = params[wKey][:,[2,1,0],:,:]
		params[wKey]  	= params[wKey].transpose((2,3,1,0)).reshape(1,num,order='F')
		if N==1 and ch==1:
			#Hacky way of finding a FC layer
			N = h
		params[bKey]    = params[bKey].reshape((1,N))
	if matlabRefFile is not None:
		params['refFile'] = matlabRefFile
	else:
		params['refFile'] = ''
	sio.savemat(outName, params)
	_mat_to_matconvnet(matlabRefFile, outName, outName)


##
# Convert matconvnet network into a caffemodel
def matconvnet_to_caffemodel(inFile, outFile):
	'''
		Relies on a matlab helper function which converts a matconvnet model 
		into an approrpriate format. 
	'''
	#Right now the code is hacked to work with the BVLC reference model definition. 
	#defFile = '/data1/pulkitag/caffe_models/bvlc_reference/caffenet_deploy.prototxt'
	#defFile = '/work4/pulkitag-code/code/ief/models/vgg_16_base.prototxt'
	defFile = '/work4/pulkitag-code/code/ief/models/vgg_s.prototxt'
	net     = caffe.Net(defFile, caffe.TEST)

	#Load the weights
	dat = sio.loadmat(inFile, squeeze_me=True)
	w     = dat['weights']
	b     = dat['biases']
	names = dat['names']
	
	#Hack the names
	#names[5] = 'fc6'
	#names[6] = 'fc7'
	#names[7] = 'fc8'

	count = 0
	for n,weight,bias in zip(names, w, b):
		print n
		if 'conv' in n:
			weight = weight.transpose((3,2,0,1))
		elif 'fc' in n:
			print weight.shape
			if weight.ndim==4:
				weight = weight.transpose((3,2,0,1))
				print weight.shape
				num,ch,h,w = weight.shape
				weight = weight.reshape((1,1,num,ch*h*w))
			elif weight.ndim==1:
				#This can happen because adding a singleton dimension in matlab in the end
				#is not possible
				print type(bias)
				assert type(bias)==type(1.0)
				weight = weight.reshape((1,len(weight)))
				print weight.shape
			else:
				weight = weight.transpose((1,0))		
		
		if type(bias)==type(1.0):
			#i.e. 1-D bias
			bias   = bias * np.ones((1,1,1,1))
		else:	
			bias   = bias.reshape((1,1,1,len(bias)))
		if count == 0:
			#RGB to BGR flip for the first layer channels
			weight[:,0:3,:,:] = weight[:,[2,1,0],:,:]	
		net.params[n][0].data[...] = weight
		net.params[n][1].data[...] = bias
		count+=1		
	#Save the network
	print outFile
	net.save(outFile)


#Converts the weights stored in a .mat file into format
#for matconvnet. 
def _mat_to_matconvnet(srcFile, targetFile, outFile):
	'''
		srcFile   : Provided the matconvnet format
		targetFile: Provides the n/w weights in .mat format
								obtained from first part of save_weights_for_matconvnet()
		outFile: Where should the weights be saved. 
	'''
	#meng = men.start_matlab()
	_ = meng.addpath(MATLAB_PATH, nargout=1)
	meng.swap_weights_matconvnet(srcFile, targetFile, outFile, nargout=0)
	meng.exit()


##
# Test the conversion of matconvnet into caffemodel
def test_convert():
	inFile = '/data1/pulkitag/others/tmp.mat'
	outFile = '/data1/pulkitag/others/tmp.caffemodel'
	#inFile   = '/data1/pulkitag/others/alex-matconvnet.mat'
	#outFile  = '/data1/pulkitag/others/alex-matconvnet.caffemodel'
	matconvnet_to_caffemodel(inFile, outFile)

def vis_convert():
	defFile = '/data1/pulkitag/caffe_models/bvlc_reference/caffenet_deploy.prototxt'
	modelFile  = '/data1/pulkitag/others/alex-matconvnet.caffemodel'
	net = mp.MyNet(defFile, modelFile)
	net.vis_weights('conv1')


def test_convert_features():	
	
	defFile = '/data1/pulkitag/caffe_models/bvlc_reference/caffenet_deploy.prototxt'
	netFile = '/data1/pulkitag/others/tmp.caffemodel'
	#netFile  = '/data1/pulkitag/others/alex-matconvnet.caffemodel'
	net = mp.MyNet(defFile, netFile)

	net.set_preprocess(isBlobFormat=True, chSwap=None)	
	imData = sio.loadmat('/data1/pulkitag/others/ref_imdata.mat',squeeze_me=True)
	imData = imData['im']
	imData = imData.transpose((3,2,0,1))
	imData = imData[:,[2,1,0],:,:]
	imData = imData[0:10]

	op = net.forward_all(blobs=['fc8','conv1','data','conv2','conv5','fc6','fc7'],**{'data': imData})
	pdb.set_trace()
	
