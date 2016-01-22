## @package my_pycaffe
#  Some quick and dirty functions
#

import my_pycaffe as mp
import my_pycaffe_utils as mpu
from os import path as osp
import caffe

##
#Save alexnet weights stored uptil various levels
def save_alexnet_levels():
	maxLayer = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6']
	modelDir = '/data1/pulkitag/caffe_models/bvlc_reference'
	defFile    = osp.join(modelDir, 'caffenet_deploy.prototxt')
	oDefFile   = osp.join(modelDir, 'alexnet_levels', 'caffenet_deploy_%s.prototxt')
	modelFile  = osp.join(modelDir, 'bvlc_reference_caffenet.caffemodel')
	oModelFile = osp.join(modelDir, 'alexnet_levels', 
											'bvlc_reference_caffenet_%s.caffemodel')
	for l in maxLayer:
		print (l)
		dFile = mpu.ProtoDef(defFile=defFile)
		dFile.del_all_layers_above(l)
		dFile.write(oDefFile % l)
		net = caffe.Net((oDefFile % l), modelFile, caffe.TEST)
		net.save(oModelFile % l)
