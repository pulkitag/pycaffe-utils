import os
import os.path as osp
import numpy as np
import my_pycaffe as mp
import my_pycaffe_utils as mpu
from easydict import EasyDict as edict
import copy
import other_utils as ou
import pickle
import my_sqlite as msq

REAL_PATH = os.path.dirname(os.path.realpath(__file__))
DEF_DB    = osp.join(REAL_PATH, 'test_data/default-exp-db.sqlite')

def get_sql_id(dbFile, dArgs, ignoreKeys=[]):
	sql    = msq.SqDb(dbFile)
	try: 
		sql.fetch(dArgs, ignoreKeys=ignoreKeys)
		idName = sql.get_id(dArgs, ignoreKeys=ignoreKeys)
	except:
		sql.close()
		raise Exception('Error in fetching a name from database')
	sql.close()
	return idName	 


def get_default_net_prms(dbFile=DEF_DB, **kwargs):
	dArgs = edict()
	#Name of the net which will be constructed
	dArgs.netName = 'alexnet'
	#For layers below lrAbove, learning rate is set to 0
	dArgs.lrAbove     = None
	#If weights from a pretrained net are to be used
	dArgs.preTrainNet = None
	#The base proto from which net will be constructed
	dArgs.baseNetDefProto = 'deploy.prototxt'
	#Batch size
	dArgs.batchsize   = None
	#runNum
	dArgs.runNum      = 0
	dArgs = mpu.get_defaults(kwargs, dArgs, False)
	dArgs.expStr      = get_sql_id(dbFile, dArgs)
	return dArgs


def get_siamese_net_prms(dbFile=DEF_DB, **kwargs):
	dArgs = get_default_net_prms()
	del dArgs['expStr']
	#Layers at which the nets are to be concatenated
	dArgs.concatLayer = 'fc6'
	#If dropouts should be used in the concatenation layer
	dArgs.concatDrop  = False
	#Number of filters in concatenation layer
	dArgs.concatSz    = None
	#If an extra FC layer needs to be added
	dArgs.extraFc     = None
	dArgs = mpu.get_defaults(kwargs, dArgs, False)
	dArgs.expStr      = get_sql_id(dbFile, dArgs)
	return dArgs


def get_siamese_window_net_prms(dbFile=DEF_DB, **kwargs):
	dArgs = get_siamese_net_prms()
	del dArgs['expStr']
	#Size of input image
	dArgs.imSz = 227
	#If random cropping is to be used	
	dArgs.randCrop = False
	#If gray scale images need to be used
	dArgs.isGray   = False
	dArgs = mpu.get_defaults(kwargs, dArgs, False)
	dArgs.expStr   = get_sql_id(dbFile, dArgs)
	return dArgs

'''
Defining get_custom_net_prms()
def get_custom_net_prms(**kwargs):
	dArgs = get_your_favorite_prms()
	del dArgs['expStr']
	##DEFINE NEW PROPERTIES##
	dArgs.myNew = value
	################
	dArgs = mpu.get_defaults(kwargs, dArgs, False)
	dArgs.expStr      = get_sql_id(dbFile, dArgs)
	return dArgs
'''

##
# Parameters that specify the learning
def get_default_solver_prms(dbFile=DEF_DB, **kwargs):
	'''
		Refer to caffe.proto for a description of the
		variables. 
	'''	
	dArgs = edict()
	dArgs.baseSolDefFile = None
	dArgs.iter_size   = 1
	dArgs.max_iter    = 250000
	dArgs.base_lr   = 0.001
	dArgs.lr_policy   = 'step' 
	dArgs.stepsize    = 20000	
	dArgs.gamma     = 0.5
	dArgs.weight_decay = 0.0005
	dArgs.clip_gradients = -1
	#Momentum
	dArgs.momentum  = 0.9
	#Other
	dArgs.regularization_type = 'L2'
	dArgs.random_seed = -1
	#Testing info
	dArgs.test_iter     = 100
	dArgs.test_interval = 1000
	dArgs.snapshot      = 2000	
	dArgs.display       = 20
	#Update parameters
	dArgs        = mpu.get_defaults(kwargs, dArgs, False)
	dArgs.expStr = 'solprms' + get_sql_id(dbFile, dArgs,
									ignoreKeys=['test_iter',  'test_interval',
								 'snapshot', 'display'])
	return dArgs 


def get_solver_caffe_prms(nwFn=None, nwPrms={}, solFn=None,
						solPrms={}, resumeIter=None, baseDefDir=''):
	if nwFn is None:
		nwFn = get_default_net_prms
	if solFn is None:
		solFn = get_default_solver_prms
	nwPrms  = nwFn(**nwPrms)
	solPrms = solFn(**solPrms) 
	cPrms   = edict()
	cPrms.baseDefDir = baseDefDir
	cPrms.nwPrms     = copy.deepcopy(nwPrms)
	cPrms.lrPrms     = copy.deepcopy(solPrms)	
	cPrms.resumeIter = resumeIter
	expStr = osp.join(cPrms.nwPrms.expStr, cPrms.lrPrms.expStr) + '/'
	del solPrms['expStr']
	del solPrms['baseSolDefFile']
	cPrms.solver = mpu.make_solver(**solPrms)	
	cPrms.expStr = expStr
	return cPrms	

##
# Programatically make a Caffe Experiment. 
class CaffeSolverExperiment:
	def __init__(self, prms, cPrms):
		'''
			prms:          dict containing key 'expName' and 'paths'
										 contains dataset specific parameters
			cPrms:         dict contraining 'expStr', 'resumeIter', 'nwPrms'
									 	 contains net and sovler spefici parameters
		'''
		dataExpName        = prms['expName']
		caffeExpName       = cPrms['expStr']
		expDirPrefix       = prms.paths.exp.dr
		snapDirPrefix      = prms.paths.exp.snapshot.dr
		#Relevant directories. 
		self.dirs_  = {}
		self.dirs_['exp']  = osp.join(expDirPrefix,  dataExpName)
		self.dirs_['snap'] = osp.join(snapDirPrefix, dataExpName)  
		self.resumeIter_  = cPrms.resumeIter
		self.runNum_      = cPrms.nwPrms.runNum 
		self.preTrainNet_ = cPrms.nwPrms.preTrainNet

		solverFile    = caffeExpName + '_solver.prototxt'
		defFile       = caffeExpName + '_netdef.prototxt'
		defDeployFile = caffeExpName + '_netdef_deploy.prototxt'
		defRecFile    = caffeExpName + '_netdef_reconstruct.prototxt'
		logFile       = caffeExpName + '_log.pkl'
		snapPrefix    = caffeExpName + '_caffenet_run%d' % self.runNum_ 

		self.files_   = {}
		self.files_['solver'] = osp.join(self.dirs_['exp'], solverFile) 
		self.files_['netdef'] = osp.join(self.dirs_['exp'], defFile)
		self.files_['netdefDeploy'] = osp.join(self.dirs_['exp'], defDeployFile) 
		self.files_['netdefRec']    = osp.join(self.dirs_['exp'], defRecFile) 
		self.files_['log'] = osp.join(self.dirs_['exp'], logFile)
		#snapshot
		self.files_['snap'] = osp.join(snapDirPrefix, dataExpName,
													snapPrefix + '_iter_%d.caffemodel')  
		self.snapPrefix_    = '"%s"' % osp.join(snapDirPrefix, dataExpName, snapPrefix)		
		self.snapPrefix_    = ou.chunk_filename(self.snapPrefix_, maxLen=242)
		#Chunk all the filnames if needed
		for key in self.files_.keys():
			self.files_[key] = ou.chunk_filename(self.files_[key])

		#Store the solver and the net definition files
		self.expFile_   = edict()
		self.expFile_.solDef_ = copy.deepcopy(cPrms.solver)
		self.setup_solver()
		self.expFile_.netDef_ = mpu.ProtoDef(osp.join(cPrms.baseDefDir,
														cPrms.nwPrms.baseNetDefProto)) 
		#Other class parameters
		self.solver_    = None
		self.expMake_   = False

	def setup_solver(self):
		self.expFile_.solDef_.add_property('device_id', 0)
		self.expFile_.solDef_.set_property('net', '"%s"' % self.files_['netdef'])
		self.expFile_.solDef_.set_property('snapshot_prefix', self.snapPrefix_)	
	
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
	def get_snapshot_name(self, numIter=10000, getSolverFile=False):
		'''
			Find the name with which models are being stored. 
		'''
		snapshot   = self.solDef_.get_property('snapshot_prefix')
		#_iter_%d.caffemodel is added by caffe while snapshotting. 
		snapshotName = snapshot[1:-1] + '_iter_%d.caffemodel'
		snapshotName = ou.chunk_filename(snapshotName)
		#solver file
		solverName   = snapshot[1:-1] + '_iter_%d.solverstate'
		solverName   = ou.chunk_filename(snapshotName)	
		if getSolverFile:
			return solverName
		else:	
			return snapshot


	## Only finetune the layers that are above ( and including) layerName
	def finetune_above(self, layerName):
		self.expFile_.netDef_.set_no_learning_until(layerName)	

	## All layernames
	def get_all_layernames(self, phase='TRAIN'):
		return self.expFile_.netDef_.get_all_layernames(phase=phase)

	## Get the top name of the last layer
	def get_last_top_name(self):
		return self.expFile_.netDef_.get_last_top_name()

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
	def make(self, deviceId=0, dumpLogFreq=1000):
		'''
			deviceId: the gpu on which to run
		'''
		self.expFile_.solDef_.set_property('device_id', deviceId)
		#Write the solver and the netdef file
		if not osp.exists(self.dirs_['exp']):
			os.makedirs(self.dirs_['exp'])
		if not osp.exists(self.dirs_['snap']):
			os.makedirs(self.dirs_['snap'])
		self.expFile_.netDef_.write(self.files_['netdef'])
		self.expFile_.solDef_.write(self.files_['solver'])
		#Create the solver	
		self.solver_ = mp.MySolver.from_file(self.files_['solver'],
									 dumpLogFreq=dumpLogFreq, logFile=self.files_['log'])
		if self.preTrainNet_ is not None:
			assert (self.resumeIter_ is None)
			self.solver_.copy_weights(self.preTrainNet_)
		if self.resumeIter_ is not None:
			solverStateFile = self.get_snapshot_name(snapName, getSolverFile=True)
			self.solver_.restore(solverStateFile)
		self.expMake_ = True
	
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
	def run(self, recFreq=20):
		if not self.expMake_:
			print ('Make the experiment using exp.make(), before running, returning')
			return
		self.solver_.solve()
	
	def get_test_accuracy(self):
		print ('NOT IMPLEMENTED YET')


	

def get_caffe_prms_old(nwPrms, lrPrms, finePrms=None, 
									 isScratch=True, deviceId=1,
									 runNum=0, resumeIter=0): 
	caffePrms = edict()
	caffePrms.deviceId  = deviceId
	caffePrms.isScratch = isScratch
	caffePrms.nwPrms    = copy.deepcopy(nwPrms)
	caffePrms.lrPrms    = copy.deepcopy(lrPrms)
	caffePrms.finePrms  = copy.deepcopy(finePrms)
	caffePrms.resumeIter = resumeIter

	expStr = nwPrms.expStr + '/' + lrPrms.expStr
	if finePrms is not None:
		expStr = expStr + '/' + finePrms.expStr
	if runNum > 0:
		expStr = expStr + '_run%d' % runNum
	caffePrms['expStr'] = expStr
	caffePrms['solver'] = lrPrms.solver
	return caffePrms



##
# Parameters required to specify the n/w architecture
def get_nw_prms(isHashStr=False, **kwargs):
	dArgs = edict()
	dArgs.netName     = 'alexnet'
	dArgs.concatLayer = 'fc6'
	dArgs.concatDrop  = False
	dArgs.contextPad  = 0
	dArgs.imSz        = 227
	dArgs.imgntMean   = True
	dArgs.maxJitter   = 0
	dArgs.randCrop    = False
	dArgs.lossWeight  = 1.0
	dArgs.multiLossProto   = None
	dArgs.ptchStreamNum    = 256
	dArgs.poseStreamNum    = 256
	dArgs.isGray           = False
	dArgs.isPythonLayer    = False
	dArgs.extraFc          = None
	dArgs.numFc5           = None
	dArgs.numConv4         = None
	dArgs.numCommonFc      = None
	dArgs.lrAbove          = None
	dArgs = mpu.get_defaults(kwargs, dArgs)
	if dArgs.numFc5 is not None:
		assert(dArgs.concatLayer=='fc5')
	expStr = 'net-%s_cnct-%s_cnctDrp%d_contPad%d_imSz%d_imgntMean%d_jit%d'\
						%(dArgs.netName, dArgs.concatLayer, dArgs.concatDrop, 
							dArgs.contextPad,
							dArgs.imSz, dArgs.imgntMean, dArgs.maxJitter)
	if dArgs.numFc5 is not None:
		expStr = '%s_numFc5-%d' % (expStr, dArgs.numFc5)
	if dArgs.numConv4 is not None:
		expStr = '%s_numConv4-%d' % (expStr, dArgs.numConv4)
	if dArgs.numCommonFc is not None:
		expStr = '%s_numCommonFc-%d' % (expStr, dArgs.numCommonFc)
	if dArgs.randCrop:
		expStr = '%s_randCrp%d' % (expStr, dArgs.randCrop)
	if not(dArgs.lossWeight==1.0):
		if type(dArgs.lossWeight)== list:
			lStr = ''
			for i,l in enumerate(dArgs.lossWeight):
				lStr = lStr + 'lw%d-%.1f_' % (i,l)
			lStr = lStr[0:-1]
			print lStr
			expStr = '%s_%s' % (expStr, lStr)
		else:
			assert isinstance(dArgs.lossWeight, (int, long, float))
			expStr = '%s_lw%.1f' % (expStr, dArgs.lossWeight)
	if dArgs.multiLossProto is not None:
		expStr = '%s_mlpr%s-posn%d-ptsn%d' % (expStr,
							dArgs.multiLossProto, dArgs.poseStreamNum, dArgs.ptchStreamNum)
	if dArgs.isGray:
		expStr = '%s_grayIm' % expStr
	if dArgs.isPythonLayer:
		expStr = '%s_pylayers' % expStr
	if dArgs.extraFc is not None:
		expStr = '%s_extraFc%d' % (expStr, dArgs.extraFc)
	if dArgs.lrAbove is not None:
		expStr = '%s_lrAbove-%s' % (expStr, dArgs.lrAbove)
	if not isHashStr:
		dArgs.expStr = expStr 
	else:
		dArgs.expStr = 'nwPrms-%s' % ou.hash_dict_str(dArgs)
	return dArgs 

##
# Parameters that specify the learning
def get_lr_prms(isHashStr=False, **kwargs):	
	dArgs = edict()
	dArgs.batchsize = 128
	dArgs.stepsize  = 20000	
	dArgs.base_lr   = 0.001
	dArgs.max_iter  = 250000
	dArgs.gamma     = 0.5
	dArgs.weight_decay = 0.0005
	dArgs.clip_gradients = -1
	dArgs.debug_info = False
	dArgs  = mpu.get_defaults(kwargs, dArgs)
	#Make the solver 
	debugStr = '%s' % dArgs.debug_info
	debugStr = debugStr.lower()
	del dArgs['debug_info']
	solArgs = edict({'test_iter': 100, 'test_interval': 1000,
						 'snapshot': 2000, 
							'debug_info': debugStr})
	#print dArgs.keys()
	expStr = 'batchSz%d_stepSz%.0e_blr%.5f_mxItr%.1e_gamma%.2f_wdecay%.6f'\
					 % (dArgs.batchsize, dArgs.stepsize, dArgs.base_lr,
							dArgs.max_iter, dArgs.gamma, dArgs.weight_decay)
	if not(dArgs.clip_gradients==-1):
		expStr = '%s_gradClip%.1f' % (expStr, dArgs.clip_gradients)
	if not isHashStr:
		dArgs.expStr = expStr 
	else:
		dArgs.expStr = 'lrPrms-%s' % ou.hash_dict_str(dArgs)
	for k in dArgs.keys():
		if k in ['batchsize', 'expStr']:
			continue
		solArgs[k] = copy.deepcopy(dArgs[k])

	dArgs.solver = mpu.make_solver(**solArgs)	
	return dArgs 

##
# Parameters for fine-tuning
def get_finetune_prms(isHashStr=False, **kwargs):
	'''
		sourceModelIter: The number of model iterations of the source model to consider
		fine_max_iter  : The maximum iterations to which the target model should be trained.
		lrAbove        : If learning is to be performed some layer. 
		fine_base_lr   : The base learning rate for finetuning. 
 		fineRunNum     : The run num for the finetuning.
		fineNumData    : The amount of data to be used for the finetuning. 
		fineMaxLayer   : The maximum layer of the source n/w that should be considered.  
	''' 
	dArgs = edict()
	dArgs.base_lr = 0.001
	dArgs.runNum  = 1
	dArgs.maxLayer = None
	dArgs.lrAbove  = None
	dArgs.dataset  = 'sun'
	dArgs.maxIter  = 40000
	dArgs.extraFc     = False
	dArgs.extraFcDrop = False
	dArgs.sourceModelIter = 60000 
	dArgs = mpu.get_defaults(kwargs, dArgs)
 	return dArgs 

def get_caffe_prms(nwPrms, lrPrms, finePrms=None, 
									 isScratch=True, deviceId=1,
									 runNum=0, resumeIter=0): 
	caffePrms = edict()
	caffePrms.deviceId  = deviceId
	caffePrms.isScratch = isScratch
	caffePrms.nwPrms    = copy.deepcopy(nwPrms)
	caffePrms.lrPrms    = copy.deepcopy(lrPrms)
	caffePrms.finePrms  = copy.deepcopy(finePrms)
	caffePrms.resumeIter = resumeIter

	expStr = nwPrms.expStr + '/' + lrPrms.expStr
	if finePrms is not None:
		expStr = expStr + '/' + finePrms.expStr
	if runNum > 0:
		expStr = expStr + '_run%d' % runNum
	caffePrms['expStr'] = expStr
	caffePrms['solver'] = lrPrms.solver
	return caffePrms

def get_default_caffe_prms(deviceId=1):
	nwPrms = get_nw_prms()
	lrPrms = get_lr_prms()
	cPrms  = get_caffe_prms(nwPrms, lrPrms, deviceId=deviceId)
	return cPrms

def get_experiment_object(prms, cPrms):
	#Legacy support
	if prms['paths'].has_key('exp'):
		expDir = prms.paths.exp.dr
	else:
		expDir = prms['paths']['expDir']
	caffeExp = mpu.CaffeExperiment(prms['expName'], cPrms['expStr'], 
							expDir, prms.paths.exp.snapshot.dr,
						  deviceId=cPrms['deviceId'])
	return caffeExp



