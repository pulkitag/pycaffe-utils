## @package other_utils
#  Miscellaneous Util Functions
#
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt
import copy
import os
from os import path as osp
import collections as co
import pdb
from easydict import EasyDict as edict

##
# Get the defaults
def get_defaults(setArgs, defArgs, defOnly=True):
	for key in setArgs.keys():
		if defOnly:
			assert defArgs.has_key(key), 'Key not found: %s' % key
		if key in defArgs.keys():
			defArgs[key] = copy.deepcopy(setArgs[key])
	return defArgs



##
# Verify if all the keys are present recursively in the dict
def verify_recursive_key(data, keyNames, verifyOnly=False):
	'''
		data    : dict like data['a']['b']['c']...['l']
		keyNames: list of keys 
		verifyOnly: if TRUE then dont raise exceptions - just return the truth value
	'''
	assert isinstance(keyNames, list), 'keyNames is required to be a list'
	#print data, keyNames
	if verifyOnly:
		if not data.has_key(keyNames[0]):
			return False
	else:
		assert data.has_key(keyNames[0]), '%s not present' % keyNames[0]
	for i in range(1,len(keyNames)):
		dat = reduce(lambda dat, key: dat[key], keyNames[0:i], data)
		assert isinstance(dat, dict), 'Wrong Keys'
		if verifyOnly:
			if not dat.has_key(keyNames[i]):
				return False
		else:
			assert dat.has_key(keyNames[i]), '%s key not present' % keyNames[i]
	return True

##
# Set the value of a recursive key. 
def set_recursive_key(data, keyNames, val):
	if verify_recursive_key(data, keyNames):
		dat = reduce(lambda dat, key: dat[key], keyNames[:-1], data)
		dat[keyNames[-1]] = val
	else:
		raise Exception('Keys not present')


def add_recursive_key(data, keyNames, val):
	#isKey = verify_recursive_key(data, keyNames)
	#assert not isKey, 'key is already present'
	dat = reduce(lambda dat, key: dat[key], keyNames[:-1], data)
	dat[keyNames[-1]] = val


##
# Delete the recursive key
def del_recursive_key(data, keyNames):
	if verify_recursive_key(data, keyNames):
		dat = reduce(lambda dat, key: dat[key], keyNames[:-1], data)
		del dat[keyNames[-1]]
	else:
		raise Exception('Keys not present')



##
# Get the item from a recursive key
def get_item_recursive_key(data, keyNames, verifyOnly=False):
	if verify_recursive_key(data, keyNames, verifyOnly=verifyOnly):
		dat = reduce(lambda dat, key: dat[key], keyNames[:-1], data)
		return dat[keyNames[-1]]
	else:
		print "Not found:", keyNames
		return None

##
# Find the path to the key in a recursive dictionary. 
def find_path_key(data, keyName):
	'''
		Returns path to the first key of name keyName that is found.
		if keyName is a list - [k1, k2 ..kp] then find in data[k1][k2]...[kp-1] the key kp
	'''
	path    = []
	prevKey = []
	if not isinstance(data, dict):
		return path
	#Find if all keys except the last one exist or not. 
	if isinstance(keyName, list):
		data = copy.deepcopy(data)
		for key in keyName[0:-1]:
			if key not in data:
				return []
			else:
				data = data[key]
		prevKey = keyName[0:-1]
		keyName = keyName[-1]

	if data.has_key(keyName):
		return [keyName]
	else:
		for key in data.keys():
			pathFound = find_path_key(data[key], keyName)
			if len(pathFound) > 0:
				return prevKey + [key] + pathFound
	return path

##
# Find an item in dict
# keyName should be a string or an list of a single name. 
def get_item_dict(data, keyName):
	keyPath = find_path_key(data, keyName)
	#print keyPath
	if len(keyPath)==0:
		return None
	else:
		return get_item_recursive_key(data, keyPath)

##
#Find the key to the item in a dict
def find_keyofitem(data, item):
	keyName = None
	for k in data.keys():
		tp = type(data[k])
		if tp == dict or tp ==edict:
			keyPath = find_keyofitem(data[k], item) 
			if keyPath is None:
				continue
			else:
				keyName = [k] + keyPath
		else:
			if data[k] == item:
				keyName = [k]
	return keyName

##
# Read the image
def read_image(imName, color=True, isBGR=False, imSz=None):
	'''
		color: True - if a gray scale image is encountered convert into color
	'''
	im = plt.imread(imName)
	if color:
		if im.ndim==2:
			print "Converting grayscale image into color image"
			im = np.tile(im.reshape(im.shape[0], im.shape[1],1),(1,1,3))
		if isBGR:
			im = im[:,:,[2,1,0]]
	#Resize if needed
	if imSz is not None:
		assert isinstance(imSz,int)
		im = scm.imresize(im, (imSz,imSz))
	return im			


##
# Crop the image
def crop_im(im, bbox, **kwargs):
	'''
		The bounding box is assumed to be in the form (xmin, ymin, xmax, ymax)
		kwargs:
			imSz: Size of the image required
	'''
	cropType = kwargs['cropType']
	imSz  = kwargs['imSz']
	x1,y1,x2,y2 = bbox
	x1 = max(0, x1)
	y1 = max(0, y1)
	x2 = min(im.shape[1], x2)
	y2 = min(im.shape[0], y2)
	if cropType=='resize':
		imBox = im[y1:y2, x1:x2]
		imBox = scm.imresize(imBox, (imSz, imSz))
	if cropType=='contPad':
		contPad = kwargs['contPad']
		x1 = max(0, x1 - contPad)
		y1 = max(0, y1 - contPad)
		x2 = min(im.shape[1], x2 + contPad)
		y2 = min(im.shape[0], y2 + contPad)	
		imBox = im[y1:y2, x1:x2]
		imBox = scm.imresize(imBox, (imSz, imSz))
	else:
		raise Exception('Unrecognized crop type')
	return imBox		

##
# Read and crop the image. 
def read_crop_im(imName, bbox, **kwargs):
	if kwargs.has_key('color'):
		im = read_image(imName, color=kwargs['color'])
	else:
		im = read_image(imName)
	return crop_im(im, bbox, **kwargs)	


##
# Makes a table from dict
def make_table(keyOrder=None, colWidth=15, sep=None, **kwargs):
	'''
		kwargs should contains keys and lists as the values.
		Each dictionaty will be plotted as a column.
	'''
	if keyOrder is None:
		keyOrder = kwargs.keys()
	if sep is None:
		sepStr = ''
	elif sep == 'csv':
		sepStr = ','
	elif sep == 'tab':
		sepStr == '\t'

	for i,key in enumerate(keyOrder):
		if i==0:
			L = len(kwargs[key])
		else:
			assert L == len(kwargs[key]), 'Wrong length for %s' % key

	N = len(keyOrder)
	formatStr = ("{:<%d} " % colWidth) + sepStr
	lines = []
	lines.append(''.join(formatStr.format(k) for k in keyOrder) + '\n')
	if sepStr is None:
		lines.append('-' * 15 * N + '\n')

	for i in range(L):
		line = ''
		for key in keyOrder:
			if isinstance(kwargs[key][i], int):
				fStr = '%d'   + sepStr
			elif type(kwargs[key][i]) in [float, np.float32, np.float64]:
				fStr = '%.1f' + sepStr
			elif isinstance(kwargs[key][i], str):
				fStr = '%s' + sepStr
			else:
				fStr = '%s' + sepStr
			line = line + formatStr.format(fStr % kwargs[key][i])
		line = line + '\n'
		lines.append(line)	
	
	for l in lines:
		print l


#I will make the rows. 
def make_table_rows(**kwargs):
	#Find the maximum length of the key. 
	maxKeyLen = 0
	for key,val in kwargs.iteritems():
		maxKeyLen = max(maxKeyLen, len(key))
	keyLen = maxKeyLen + 15
	keyStr = "{:<%d} " % keyLen
	formatStr = "{:<15} "
	#Lets start printing
	lines = []
	count = 0	
	for key,val in kwargs.iteritems():
		line = ''
		line = line + keyStr.format('%s' % key)
		for v in val:
			if isinstance(v, int):
				fStr = '%d'
			elif isinstance(v, np.float32) or isinstance(v, np.float64):
				fStr = '%.3f'
			elif isinstance(v, str):
				fStr = '%s'
			else:
				fStr = '%s'
			line = line + formatStr.format(fStr % v)
		line = line + '\n'
		lines.append(line)
		if count == 0:
			lines.append('-' * 100 + '\n')
			count += 1
			
	for l in lines:
		print l

##
# In a recursive dictionary - subselect a few fields
# while vary others. For eg d['vr1']['a1']['b1'], d['vr2']['a1']['b2'], d['vr3']['a2']['b3']
# Now I might be interested only in values such that the second field is fixed to 'a1'
# So that I get the output as d['vr1']['b1'], d['vr2']['b2']
def conditional_select(data, fields, reduceFn=None):
	'''
		data       : dict
		fields     : fields (a list)
			           [None,'a1',None] means that keep the second field fixed to 'a1',
			           but consider all values of other fields.
		reduceFn   : Typically the dict would store an array
								 reductionFn can be any function to reduce this array to
								 a quantity of interset like mean etc. 
	'''
	newData = co.OrderedDict()
	for key in data.keys():
		if fields[0] is None:
			#Chose all the keys
			newData[key] = conditional_select(data[key], fields[1:], reduceFn=reduceFn) 
		else:
			if key == fields[0]:	
				if len(fields) > 1:
					newData = conditional_select(data[key], fields[1:], reduceFn=reduceFn)
				else:
					if reduceFn is None:
						newData = copy.deepcopy(data[key])
					else:
						newData = copy.deepcopy(reduceFn(data[key]))
					return newData
	return newData

##
# Count the things.
def count_unique(arr, maxVal=None):
	if maxVal is None:
		elms = np.unique(arr)
	else:
		elms = range(maxVal+1)
	count = np.zeros((len(elms),))
	for i,e in enumerate(elms):
		count[i] = np.sum(arr==e)

	return count
	 
##
# Create dir
def create_dir(dirName):
	if not os.path.exists(dirName):
		os.makedirs(dirName)

##
#Private function for chunking a path
def _chunk_path(fName, N):
	assert '/' not in fName
	L = len(fName)
	if L <= N:
		return fName
	else:
		slices=[]
		for i in range(0,L,N):
			slices.append(fName[i:min(L, i+N)])
	newName = ''.join('%s/' % s for s in slices)
	newName = newName[0:-1]
	return newName
		
##
# chunk filenames
def chunk_filename(fName, maxLen=255):
	'''
		if any of the names is larger than 256 then
		the file cannot be stored so some chunking needs
		to be done
	'''
	splitNames = fName.split('/')
	newSplits  = []
	for s in splitNames:
		if len(s)>=maxLen:
			newSplits.append(_chunk_path(s, maxLen-1))
		else:
			newSplits.append(s)
	newName = ''.join('%s/' % s for s in newSplits)
	newName = newName[0:-1]
	dirName = os.path.dirname(newName)
	create_dir(dirName)
	return newName	

##
# Hash a dictonary into string
def hash_dict_str(d, ignoreKeys=[]):
	d     = copy.deepcopy(d)
	oKeys = []
	for k in ignoreKeys:
		if k in d.keys():
			del d[k]
	for k,v in d.iteritems():
		if type(v) in [bool, int, float, str, type(None)]:
			continue
		else:
			assert type(v) in [dict, edict, co.OrderedDict],\
				 'Type not recognized %s, for this type different results for different runs of\
					hashing can be obtained, therefore the exception' % v	
			oKeys.append(k)
	hStr = []
	for k in oKeys:
		hStr.append('-%s' % hash_dict_str({k: d[k]}))
		del d[k]
	hStr = ''.join('%s' % s for s in hStr)
	return '%d%s' % (hash(frozenset(d.items())), hStr)

##
#
def mkdir(fName):
	if not osp.exists(fName):
		os.makedirs(fName)

##
#Make parameter string for python layers
def make_python_param_str(params, ignoreKeys=['expStr']):
	paramStr = ''
	for k,v in params.iteritems():
		if k in ignoreKeys:
			continue
		if type(v) == bool:
			if v:
				paramStr = paramStr + ' --%s' % k
			else:
				paramStr = paramStr + ' --no-%s' % k
		else:
			paramStr = paramStr + ' --%s %s' % (k,v)
	return paramStr

##
#Convert a list of ints into a string
def ints_to_str(ints):
	ch = ''.join(str(unichr(i)) for i in ints)
	return ch
