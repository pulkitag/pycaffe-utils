import h5py as h5
import numpy as np
import other_utils as ou
import my_pycaffe_utils as mpu
import caffe
import collections as co

class MatConvNetModel:
	def __init__(self, inFile):
		self.dat_ = h5.File(inFile, 'r')
	
	def ref_to_str(self, ref):
		secRefs = self.dat_['#refs#'][ref][:]
		ch = []
		#print secRefs
		for sec in secRefs:
			if not sec == 0:
				#If the reference is not empty
				ch.append(ou.ints_to_str(self.dat_['#refs#'][sec[0]]))
		return ch

	def make_caffe_layer(self, lNum):
		#Get the name
		nameRef = self.dat_['net']['layers']['name'][lNum][0]
		name    = ou.ints_to_str(self.dat_['#refs#'][nameRef][:])
		#Get the inputs
		ipRef   = self.dat_['net']['layers']['inputs'][lNum][0]
		ipNames = self.ref_to_str(ipRef)
		#Get the parameter names
		pmRef   = self.dat_['net']['layers']['params'][lNum][0]
		pmNames = self.ref_to_str(pmRef) 
		#Get the output names
		opRef   = self.dat_['net']['layers']['outputs'][lNum][0]
		opNames = self.ref_to_str(opRef) 
		#Get the layer type
		tpRef   = self.dat_['net']['layers']['type'][lNum][0]
		lType   = ou.ints_to_str(self.dat_['#refs#'][tpRef][:])
		#Get the layer params
		lpRef   = self.dat_['net']['layers']['block'][lNum][0]
		lParam  = self.dat_['#refs#'][lpRef]
		assert (lType[0:5] == 'dagnn')
		lType   = lType[6:]
	
		if lType == 'Conv':
			paramW  = {'name': pmNames[0]}
			paramB  = {'name': pmNames[1]}
			pDupKey = mpu.make_key('param', ['param'])
			lDef = mpu.get_layerdef_for_proto('Convolution', name, ipNames[0], 
							**{'num_output': int(lParam['size'][3][0]),
							  'param': paramW, pDupKey: paramB,
								'kernel_size': int(lParam['size'][0][0]), 
								'stride': int(lParam['stride'][0][0]),
								'pad': int(lParam['pad'][0][0])})				
	
		elif lType == 'ReLU':
			lDef = mpu.get_layerdef_for_proto(lType, name, ipNames[0],
							**{'top': opNames[0]})
	
		elif lType == 'Pooling':
			lDef = mpu.get_layerdef_for_proto(lType, name, ipNames[0], 
							**{'top': opNames[0], 'kernel_size': int(lParam['poolSize'][0][0]),
								 'stride': int(lParam['stride'][0][0]), 'pad': int(lParam['pad'][0][0])})
	
		elif lType == 'LRN':
			N, kappa, alpha, beta = lParam['param'][0][0], lParam['param'][1][0],\
															lParam['param'][2][0], lParam['param'][3][0]
			lDef = mpu.get_layerdef_for_proto(lType, name, ipNames[0], 
							**{'top': opNames[0],
								 'local_size': int(N), 
								 'alpha': N * alpha,
								 'beta' : beta,
								 'k'    : kappa})
	
		elif lType == 'Concat':
			lDef = mpu.get_layerdef_for_proto(lType, name, ipNames[0],
							**{'bottom2': ipNames[1:],
								 'concat_dim': 1, 
								 'top': opNames[0]}) 
	
		elif lType == 'Loss':
			lossType = ou.ints_to_str(lParam['loss'])
			if lossType == 'pdist':
				p = lParam['p'][0][0]
				if p == 2:
					lossName = 'EuclideanLoss'
				else:
					raise Exception('Loss type %s not recognized' % lossType)		
			else:
				raise Exception('Loss type %s not recognized' % lossType)		
			lDef = mpu.get_layerdef_for_proto(lossName, name, ipNames[0],
							**{'bottom2': ipNames[1]})
	
		elif lType == 'gaussRender':
			lDef = mpu.get_layerdef_for_proto(lType, name, ipNames[0], 
				**{'top': opNames[0], 
					 'K': lParam['K'][0][0], 'T': lParam['T'][0][0], 
					 'sigma': lParam['sigma'][0][0], 'imgSz': int(lParam['img_size'][0][0])})
						
		else:
			raise Exception('Layer Type %s not recognized, %d' % (lType, lNum))
		return lDef

	#Convert the model to Caffe
	def to_caffe(self, ipLayers=[], layerOrder=[]):
		'''
			Caffe doesnot support DAGs but MatConvNet does. layerOrder allows some matconvnet 
			nets to expressed as caffe nets by moving the order of layers so as to allow caffe
			to read the generated prototxt file. 
		'''
		pDef = mpu.ProtoDef()
		caffeLayers = co.OrderedDict()
		for lNum in range(len(self.dat_['net']['layers']['name'])):
			cl = self.make_caffe_layer(lNum)
			caffeLayers[cl['name'][1:-1]] = cl
		#Add input layers if needed
		for ipl in ipLayers:
			pDef.add_layer(ipl['name'][1:-1], ipl)
		#Add the ordered layers first
		for l in layerOrder:
			pDef.add_layer(l, caffeLayers[l])
			del caffeLayers[l]
		for key, cl in caffeLayers.iteritems():
			pDef.add_layer(key, cl)
		return pDef		

##
# Convert matconvnet network into a caffemodel
def matconvnet_dag_to_caffemodel(inFile, outFile):
	dat = h5.File(inFile, 'r')


## test the conversion
def test_convert():
	fName   = '/work4/pulkitag-code/code/ief/IEF/models/new_models/models/new-model.mat' 	
	outName = 'try.prototxt'
	model = MatConvNetModel(fName)
	imgLayer = mpu.get_layerdef_for_proto('DeployData', 'image', None, 
								**{'ipDims': [1, 3, 224, 224]})
	kpLayer  = mpu.get_layerdef_for_proto('DeployData', 'kp_pos', None, 
								**{'ipDims': [1, 17, 2, 1]}) 
	lbLayer  = mpu.get_layerdef_for_proto('DeployData', 'label', None, 
								**{'ipDims': [1, 16, 2, 1]}) 
	pdef  = model.to_caffe(ipLayers=[imgLayer, kpLayer, lbLayer], layerOrder=['render1', 'concat1'])
	pdef.write(outName)
	net   = caffe.Net(outName, caffe.TEST)
	
