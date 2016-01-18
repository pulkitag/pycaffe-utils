## @package my_pycaffe_tests
#  Unit Testing functions. 
#

import my_pycaffe as mp
import my_pycaffe_utils as mpu
import numpy as np
import pdb
import os
try:
	import h5py
except:
	print ('WARNING: h5py not found, some functions may not work')

##
# Test code for Zeiler-Fergus Saliency. 
def test_zf_saliency(dataSet='mnist', stride=2, patchSz=5):

	if dataSet=='mnist':
		defFile   = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/hdf5_test/lenet.prototxt'
		modelFile,_ = mp.get_model_mean_file('lenet')
		net       = mp.MyNet(defFile, modelFile, isGPU=False)
		N         = net.get_batchsz()
		net.set_preprocess(chSwap=None, imageDims=(28,28,1), isBlobFormat=True)	

		h5File    = '/data1/pulkitag/mnist/h5store/test/batch1.h5' 
		fid       = h5py.File(h5File,'r')
		data      = fid['data']
		data      = data[0:N]

		#Do the saliency
		imSal, score  = mpu.zf_saliency(net, data, 10, 'ip2', patchSz=patchSz, stride=stride)	
		gtLabels      = fid['label']
	else:
		netName = 'bvlcAlexNet'
		opLayer = 'fc8'
		defFile = mp.get_layer_def_files(netName, layerName=opLayer)
		modelFile, meanFile = mp.get_model_mean_file(netName)
		net  = mp.MyNet(defFile, modelFile)
		net.set_preprocess(imageDims=(256,256,3), meanDat=meanFile, rawScale=255, isBlobFormat=True)

		ilDat = mpu.ILSVRC12Reader()
		ilDat.set_count(2)
		data,gtLabels,syn,words = ilDat.read()
		data = data.reshape((1,data.shape[0],data.shape[1],data.shape[2]))
		data = data.transpose((0,3,1,2))
		print data.shape
		imSal, score = mpu.zf_saliency(net, data, 1000, 'fc8', patchSz=patchSz, stride=stride) 

	pdLabels      = np.argmax(score.squeeze(), axis=1)
	return data, imSal, pdLabels, gtLabels


##
# Test Reading the protoFile
def test_get_proto_param():
	paths    = mpu.get_caffe_paths()
	testFile = os.path.join(paths['pythonTest'], 'test_conv_param.txt')
	fid = open(testFile, 'r')
	lines = fid.readlines()
	fid.close()
	params = mpu.get_proto_param(lines)
	return params  		
