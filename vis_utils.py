## @package vis_utils
#  Miscellaneous Functions for visualizations
#
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt
import my_pycaffe_utils as mpu
import copy
import os
import caffe
import pdb
import my_pycaffe as mp

TMP_DATA_DIR = '/data1/pulkitag/others/caffe_tmp_data/'

##
# Plots pairs of images. 
def plot_pairs(im1, im2, fig=None, titleStr='', figTitle=''):
	if fig is None:
		fig = plt.figure()
	plt.figure(fig.number)
	ax1 = plt.subplot(1,2,1)
	ax2 = plt.subplot(1,2,2)
	ax1.imshow(im1.astype(np.uint8))
	ax1.axis('off')
	ax2.imshow(im2.astype(np.uint8))
	ax2.axis('off')
	plt.title(titleStr)
	fig.suptitle(figTitle)
	plt.show()

##
# Visualize GenericWindowDataLayer file
def vis_generic_window_data(protoDef, numLabels, layerName='window_data', phase='TEST',
												maxVis=100):
	'''
		layerName: The name of the generic_window_data layer
		numLabels: The number of labels. 
	'''
	#Just write the data part of the file. 
	if not isinstance(protoDef, mpu.ProtoDef):
		protoDef = mpu.ProtoDef(protoDef)
	protoDef.del_all_layers_above(layerName)
	randInt = np.random.randint(1e+10)
	outProto = os.path.join(TMP_DATA_DIR, 'gn_window_%d.prototxt' % randInt)
	protoDef.write(outProto)		
	#Extract the name of the data and the label blobs. 
	dataName  = protoDef.get_layer_property(layerName, 'top', propNum=0)[1:-1]
	labelName = protoDef.get_layer_property(layerName, 'top', propNum=1)[1:-1]
	crpSize   = int(protoDef.get_layer_property(layerName, ['crop_size']))
	mnFile    = protoDef.get_layer_property(layerName, ['mean_file'])[1:-1]
	mnDat     = mp.read_mean(mnFile)
	ch,nr,nc  = mnDat.shape
	xMn       = int((nr - crpSize)/2)
	mnDat     = mnDat[:,xMn:xMn+crpSize,xMn:xMn+crpSize]
	print mnDat.shape
	
	#Create a network	
	if phase=='TRAIN':	
		net     = caffe.Net(outProto, caffe.TRAIN)
	else:
		net     = caffe.Net(outProto, caffe.TEST)

	lblStr = ''.join('lb-%d: %s, ' % (i,'%.2f') for i in range(numLabels))
	figDt = plt.figure()
	plt.ion()
	for i in range(maxVis):
		allDat  = net.forward([dataName,labelName])
		imData  = allDat[dataName] + mnDat
		lblDat  = allDat[labelName]
		batchSz = imData.shape[0]
		for b in range(batchSz):
			#Plot network data. 
			im1 = imData[b,0:3].transpose((1,2,0))
			im2 = imData[b,3:6].transpose((1,2,0))
			im1 = im1[:,:,[2,1,0]]
			im2 = im2[:,:,[2,1,0]]
			lb  = lblDat[b].squeeze()
			lbStr = lblStr % tuple(lb)	
			plot_pairs(im1, im2, figDt, lbStr) 
			raw_input()

