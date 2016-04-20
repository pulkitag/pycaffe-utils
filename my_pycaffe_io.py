## @package my_pycaffe_io 
#  IO operations. 
#

try:
	import h5py as h5
except:
	print ('WARNING: h5py not found, some functions may not work')
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
from os import path as osp
import other_utils as ou

if not cfg.IS_EC2:
	#import matlab.engine as men
	MATLAB_PATH = '/work4/pulkitag-code/pkgs/caffe-v2-2/matlab/caffe'
else:
	MATLAB_PATH = ''


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

'''
def ims2hdf5(im, labels, batchSz, batchPath, isColor=True, batchStNum=1, isUInt8=True, scale=None, newLabels=False):
	#Converts an image dataset into hdf5
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
'''

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
	def __init__(self, dbName, isLMDB=True, readahead=True, wrapAround=False):
		'''
				wrapAround: False - return None, None if end of file is reached
										True  - move to the first element
		'''
		#For large LMDB set readahead to be False
		self.db_     = lmdb.open(dbName, readonly=True, readahead=readahead)
		self.txn_    = self.db_.begin(write=False) 
		self.cursor_ = self.txn_.cursor()		
		self.nextValid_ = True
		self.wrap_      = wrapAround
		self.cursor_.first()

	def __del__(self):
		#self.txn_.commit()
		self.db_.close()
	
	#Maintain the appropriate variables
	def _maintain(self):
		if self.wrap_:
			if not self.nextValid_:
				print ('Going to first element of lmdb')
				self.cursor_.first()
				self.nextValid_ = True

	#Get the current key
	def get_key(self):
		if not self.nextValid_:
			return self.cursor_.key() 
		else:
			return None	

	#Get all keys
	def get_key_all(self):
		keys = []
		self.cursor_.first()
		isNext = True
		while isNext:
			key    = self.cursor_.key()
			isNext = self.cursor_.next()
			keys.append(key)
		self.cursor_.first()
		return keys 
		
	def read_key(self, key):
		dat    = self.cursor_.get(key)
		datum  = caffe.io.caffe_pb2.Datum()
		datStr = datum.FromString(dat)
		data   = caffe.io.datum_to_array(datStr)
		label  = datStr.label
		return data, label
	
 
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
		self._maintain()
		return data, label

	#Read a batch of elements
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

	#Get number of elements
	def get_count(self):
		return int(self.db_.stat()['entries'])

	#Skip one element
	def skip(self):
		isNext = self.cursor_.next()
		if not isNext:
			self.cursor_.first()
		self._maintain()
	
	#Skip in reverse
	def skip_reverse(self):
		isPrev = self.cursor_.prev()
		#Prev skip will not be possible if we are the first element
		if not isPrev:
			self.cursor_.last()
		self._maintain()

	#Compute the mean of the data
	def compute_mean(self):
		self.cursor_.first()
		im, _ = self.read_next()
		mu    = np.zeros(im.shape)
		mu[...] = im[...]
		wrap    = self.wrap_
		self.wrap_ = False
		N       = 1
		while True:
			im, _ = self.read_next()
			if im is None:
				break
			mu += im
			N  += 1
			if np.mod(N,1000)==1:
				print ('Processed %d images' % N)
		mu = mu / float(N)
		self.wrap_ = wrap
		return mu	
	
	#close
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
class DoubleDbReader(object):
	def __init__(self, dbNames, isLMDB=True, readahead=True, 
							 wrapAround=False, isMulti=False):
		'''
				wrapAround: False - return None, None if end of file is reached
										True  - move to the first element
				isMulti   : False - read only two dbs v(flag for backward compatibility)
									  True  - read from arbitrary number of dbs
		'''
		#For large LMDB set readahead to be False
		self.dbs_     = []
		self.isMulti_ = isMulti
		for d in dbNames:
			self.dbs_.append(DbReader(d, isLMDB=isLMDB, readahead=readahead,
												wrapAround=wrapAround))	

	def __del__(self):
		for db in self.dbs_:
			db.__del__()
	
	def read_key(self, keys):
		data = []
		for db, key in zip(self.dbs_, keys):
			dat, _ = db.read_key(key)
			data.append(dat)
		return data
	
	def read_next(self):
		data = []
		for db in self.dbs_:
			dat,_ = db.read_next()
			data.append(dat)
		if self.isMulti_:
			return data
		else:
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
		if self.isMulti_:
			return data, label
		else:
			return data[0], data[1], label[0], label[1]

	def close(self):
		for db in self.dbs_:
			db.close()

##
# Read multiple LMDBs simultaneosuly
class MultiDbReader(DoubleDbReader):
	def __init__(self, dbNames, isLMDB=True, readahead=True, 
							 wrapAround=False):
		DoubleDbReader.__init__(self, dbNames, isLMDB=isLMDB,
			readahead=readahead, wrapAround=wrapAround, isMulti=True)

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
		
	#Get the processed images and labels
	def read_next_processed(self, rootFolder, returnName=False):
		imDat, lbls = self.read_next()
		ims     = []
		imNames, outNames = [], []
		for l in imDat:
			imName, ch, h, w, x1, y1, x2, y2 = l.strip().split()
			imName = osp.join(rootFolder, imName)
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
			im = scm.imread(imName)
			ims.append(im[y1:y2, x1:x2,:])
			imNames.append(imName)
			#Generate an outprefix that maybe used to save the images
			_, fName  = osp.split(imName)
			ext       = fName[-4:]	
			outNames.append(fName[:-4] + '-%d-%d-%d-%d%s' % (x1,y1,x2,y2,ext))
		if returnName:
			return ims, lbls[0], imNames, outNames
		else:
			return ims, lbls[0]	
	
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

	#Save image crops
	def save_crops(self, rootFolder, tgtDir, numIm=None):
		'''
			rootFolder: the root folder for the window file
			tgtDir    : the directory where the images should be saved
		'''
		count    = 0
		readFlag = True
		ou.mkdir(tgtDir)	
		while readFlag:	
			ims, _, imNames, oNames = self.read_next_processed(rootFolder,
																 returnName=True)
			for im, name, oName in zip(ims, imNames, oNames):
				svName   = osp.join(tgtDir, oName)
				scm.imsave(svName, im)
			if self.is_eof():
				readFlag = False
			count += 1
			if numIm is not None and count >= numIm:
				readFlag = False

	

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
		self.initWrite_ = False
		
		#If image and labels are being stacked
		self.imStack_ = []
		self.lbStack_ = []

	#Start writing
	def init_write(self):
		if self.initWrite_:
			return
		self.fid_ = open(self.file_, 'w')	
		self.fid_.write('# GenericDataLayer\n')
		self.fid_.write('%d\n' % self.num_) #Num Examples. 
		self.fid_.write('%d\n' % self.numIm_) #Num Images per Example. 
		self.fid_.write('%d\n' % self.lblSz_) #Num	Labels
		self.initWrite_ = True

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
		assert len(args)==self.numIm_,\
			 'Wrong input arguments: (%d v/s %d)' % (len(args),self.numIm_)
		#Make sure the writing has been intialized
		if not self.initWrite_:
			self.init_write()
		#Start writing the current stuff
		self.fid_.write('# %d\n' % self.count_)
		#Write the images
		for arg in args:
			if type(arg)==str:
				#Assuming arg is the imageline read from another window-file
				#and the last character in the str is \n
				self.fid_.write(arg)
			else:
				#print (len(arg), arg)
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
	#Instead of writing, just stack
	def push_to_stack(self, lbl, *args):
		assert len(args)==self.numIm_,\
			 'Wrong input arguments: (%d v/s %d)' % (len(args),self.numIm_)
		self.imStack_.append(args)
		self.lbStack_.append(lbl)

	##
	#Write the stack
	def write_stack(self, rndState=None, rndSeed=None):
		if rndSeed is not None:
			rndState = np.random.RandomState(rndSeed)
		N = len(self.imStack_)
		assert N == len(self.lbStack_)

		if rndState is None:
			perm = range(N)
		else:
			perm = rndState.permutation(N)
		ims    = [self.imStack_[p] for p in perm]
		lbs    = [self.lbStack_[p] for p in perm]
		self.num_ = N
		for n in range(N):
			self.write(lbs[n], *(ims[n][0]))
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

def red_col_sel(col):
  if col[0] < 0.6:
    return False
  else:
    return True

#READ PCD Files
class PCDReader(object):
  def __init__(self, fName=None, keepNaN=False, subsample=None, colsel=None, fromFile=True):
    self.fName = fName
    self.ax_   = None
    if fromFile:
      self.read(keepNaN=keepNaN,subsample=subsample, colsel=colsel)

  @classmethod
  def from_pts(cls, pts, subsample=0.2):
    self = cls(None, fromFile=False)
    N       = pts.shape[0]
    pts     = pts.copy()
    if subsample is not None:
      perm = np.random.permutation(N)
      perm = perm[0:int(subsample*N)]
      pts  = pts[perm]
    self.x_ = pts[:,0]
    self.y_ = pts[:,1]
    self.z_ = pts[:,2]
    self.c_ = pts[:,3:6]
    return self 
 
  @classmethod
  def from_db(cls, dbPath):
    self     = cls(None, fromFile=False)
    self.db_ = DbReader(dbPath)
    return self

  def read_next(self, subSample=None):
    dat,_ = self.db_.read_next()
    if dat is None:
      return None
    dat       = dat.transpose((1,0,2))
    nr, nc, _ = dat.shape
    if subSample is None:
      subSample=1
    rows      = range(0, nr, subSample)
    cols      = range(0, nc, subSample)
    xIdx, yIdx = np.meshgrid(cols, rows)
    dat       = dat[yIdx, xIdx]
    self.x_ = dat[:,:,0]
    self.y_ = dat[:,:,1]
    self.z_ = dat[:,:,2]
    self.c_ = dat[:,:,3:6]
    return True
  
  def get_mask(self): 
    nanMask = np.isnan(self.z_) 
    d       = self.z_.copy()
    d[nanMask] = -10
    d   = d + 0.255
    self.mask_     = d > 0
 
  def to_rgbd(self):
    self.get_mask()
    im  = (self.c_.copy() * 255).astype(np.uint8)
    d   = self.z_.copy()
    d[~self.mask_] = 0.0
    assert np.max(d) < 0.1, np.max(d)
    d[self.mask_] = 255 * (d[self.mask_]/0.1) 
    d        = d.astype(np.uint8)
    print (np.min(d), np.max(d))
    return im, d

  def get_masked_pts(self):
    self.get_mask()
    N   = np.sum(self.mask_)
    pts = np.zeros((N,6), np.float32)
    pts[:,0] = 10*self.x_[self.mask_].reshape(N,)
    pts[:,1] = 10*self.y_[self.mask_].reshape(N,)
    pts[:,2] = 10*self.z_[self.mask_].reshape(N,)
    pts[:,3:6] = self.c_[self.mask_,0:3].reshape(N,3)    
    return pts

  def save_rgbd(self, dirName=''):
    count = 0
    imName = osp.join(dirName, 'im%06d.png')
    dpName = osp.join(dirName, 'dp%06d.png')
    while True:
      isExist = self.read_next()
      if isExist is None:
        break
      im, d = self.to_rgbd()
      print im.shape, d.shape
      ou.mkdir(osp.dirname(imName)) 
      scm.imsave(imName % (count+1), im)
      scm.imsave(dpName % (count+1), d)
      count += 1
       
  def plot_next_rgbd(self):
    import matplotlib.pyplot as plt
    self.read_next()
    im, d = self.to_rgbd()
    if self.ax_ is None:
      self.ax_ = []
      plt.ion()
      fig = plt.figure()
      self.ax_.append(fig.add_subplot(121))
      self.ax_.append(fig.add_subplot(122))
    self.ax_[0].imshow(im)
    self.ax_[1].imshow(d)  
    plt.draw()
    plt.show()

  def read(self, keepNaN=False, subsample=None, colsel=None):
    with open(self.fName, 'r') as fid:
      lines = fid.readlines()
    for i, l in enumerate(lines):
      if i<=8:
        continue
      if i==9:
        N = int(l.strip().split()[1])
        break
    lines = lines[11:]
    assert len(lines)==N
    self.x_ = np.zeros((N,), np.float32) 
    self.y_ = np.zeros((N,), np.float32) 
    self.z_ = np.zeros((N,), np.float32)
    self.c_ = np.zeros((N,3), np.float32)
    count   = 0 
    for i,l in enumerate(lines):
      if subsample is not None:
        if not np.mod(i, subsample) == 0:
          continue
      x, y, z, rgb = l.strip().split()
      nanVal = False
      if x=='nan' or y=='nan' or z=='nan':
        nanVal = True
      if not keepNaN and nanVal:
        continue
      #col = np.array([np.float32(rgb)])
      col =  np.array([np.float64(rgb)], np.float32)
      col = (col.view(np.uint8)[0:3])/256.0
      #to rgb
      col = np.array((col[2], col[1], col[0]))
      if nanVal:
        col = np.array((0,0,1.0))
      if colsel is not None:
        isValid = colsel(col)
        if not isValid:
          continue
      self.x_[count] = float(x)
      self.y_[count] = float(y)
      self.z_[count] = float(z)
      self.c_[count] = col
      count += 1
    self.x_ = self.x_[0:count]
    self.y_ = self.y_[0:count]
    self.z_ = self.z_[0:count]
    self.c_ = self.c_[0:count]
    self.N_ = count

  def get_rgb_im(self):
    im = np.zeros((480, 640, 3)).astype(np.uint8)
    count = 0
    for r in range(480): 
      for c in range(640):
        im[r, c] = (255 * self.c_[count].reshape((1,1,3))).astype(np.uint8)
        count += 1
    return im

  def matplot(self, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
      from mpl_toolkits.mplot3d import Axes3D
      plt.ion()
      fig = plt.figure()
      ax  = fig.add_subplot(111, projection='3d')
    #perm = np.random.permutation(self.x_.shape[0])
    #perm = perm[0:5000]
    perm  = np.array(range(self.x_.shape[0]))
    for i in range(1):
      #ax.scatter(self.x_, self.y_, self.z_, 
      #   c=tuple(np.random.rand(self.N_,3)))
      ax.scatter(self.x_[perm], self.y_[perm], self.z_[perm], 
         c=tuple(self.c_[perm]))
    plt.draw()
    plt.show()
    
	
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
	
