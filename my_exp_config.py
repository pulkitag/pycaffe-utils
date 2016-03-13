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
	#try:
	#print ('Ignore KEYS: ', ignoreKeys) 
	sql.fetch(dArgs, ignoreKeys=ignoreKeys)
	idName = sql.get_id(dArgs, ignoreKeys=ignoreKeys)
	#except:
	#	sql.close()
	#	raise Exception('Error in fetching a name from database')
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
	dArgs.baseNetDefProto = None
	#Batch size
	dArgs.batchSize   = None
	#runNum
	dArgs.runNum      = 0
	dArgs = mpu.get_defaults(kwargs, dArgs, False)
	dArgs.expStr      = get_sql_id(dbFile, dArgs)
	return dArgs


def get_siamese_net_prms(dbFile=DEF_DB, **kwargs):
	dArgs = get_default_net_prms(dbFile)
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
	dArgs = get_siamese_net_prms(dbFile)
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
								 'snapshot', 'display', 'resumeIter'])
	return dArgs 

##
#Make Solver
def get_solver_def(solPrms):
	solPrms = copy.deepcopy(solPrms)
	del solPrms['expStr']
	if solPrms['baseSolDefFile'] is not None:
		solDef = mpu.Solver.from_file(solPrms)	
	else:
		del solPrms['baseSolDefFile']
		solDef = mpu.make_solver(**solPrms)	
	return solDef

##
#Make Solver
def get_net_def(dPrms, nwPrms):
	'''
		dPrms : data parameters
		nwPrms: parameters that define the net 
	'''
	if nwPrms.baseNetDefProto is None:
		return None
	else:
		netDef = mpu.ProtoDef(nwPrms.baseNetDefProto) 
	return netDef


def get_caffe_prms(nwFn=None, nwPrms={}, solFn=None,
						solPrms={}, resumeIter=None):
	'''
		nwFn, nwPrms  : nwPrms are passed as input to nwFn to generate the 
										complete set of network parameters. 
		solFn, solPrms: solPrms are passes as input ro solFn to generate the
										complete set of solver parameters.
		resumeIter    : if the experiment needs to be resumed from a previously
										stored number of iterations.  
	'''
	if nwFn is None:
		nwFn = get_default_net_prms
	if solFn is None:
		solFn = get_default_solver_prms
	nwPrms  = nwFn(**nwPrms)
	solPrms = solFn(**solPrms) 
	cPrms   = edict()
	cPrms.nwPrms     = copy.deepcopy(nwPrms)
	cPrms.lrPrms     = copy.deepcopy(solPrms)	
	cPrms.resumeIter = resumeIter
	expStr = osp.join(cPrms.nwPrms.expStr, cPrms.lrPrms.expStr) + '/'
	cPrms.expStr = expStr
	return cPrms	

##
# Programatically make a Caffe Experiment. 
class CaffeSolverExperiment:
	def __init__(self, dPrms, cPrms, 
           netDefFn=get_net_def, solverDefFn=get_solver_def,
           isLog=True,  addFiles=None):
		'''
			dPrms:         dict containing key 'expStr' and 'paths'
										 contains dataset specific parameters
			cPrms:         dict contraining 'expStr', 'resumeIter', 'nwPrms'
									 	 contains net and sovler specific parameters
			isLog:         if logging data should be recorded
			addFiles:      if additional files need to be stored - not implemented yet
		'''
		self.isLog_        = isLog
		dataExpName        = dPrms['expStr']
		caffeExpName       = cPrms['expStr']
		expDirPrefix       = dPrms.paths.exp.dr
		snapDirPrefix      = dPrms.paths.exp.snapshot.dr
		#Relevant directories.
		self.dPrms_ = copy.deepcopy(dPrms)
		self.cPrms_ = copy.deepcopy(cPrms) 
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
		self.expFile_.solDef_ = solverDefFn(cPrms.lrPrms)
		self.setup_solver()
		self.expFile_.netDef_ = netDefFn(dPrms, cPrms.nwPrms) 
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
		assert self.expFile_.solDef_ is not None, 'Solver has not been formed'
		snapshot   = self.expFile_.solDef_.get_property('snapshot_prefix')
		#_iter_%d.caffemodel is added by caffe while snapshotting. 
		snapshotName = snapshot[1:-1] + '_iter_%d.caffemodel' % numIter
		snapshotName = ou.chunk_filename(snapshotName) 
		#solver file
		solverName   = snapshot[1:-1] + '_iter_%d.solverstate' % numIter
		solverName   = ou.chunk_filename(solverName)	
		if getSolverFile:
			return solverName
		else:	
			return snapshotName


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
		assert self.expFile_.solDef_ is not None, 'SolverDef has not been formed'
		assert self.expFile_.netDef_ is not None, 'NetDef has not been formed'
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
									 dumpLogFreq=dumpLogFreq, logFile=self.files_['log'],
                   isLog=self.isLog_)

		if self.resumeIter_ is not None:
			solverStateFile = self.get_snapshot_name(self.resumeIter_, getSolverFile=True)
			assert osp.exists(solverStateFile), '%s not present' % solverStateFile
			self.solver_.restore(solverStateFile)
		elif self.preTrainNet_ is not None:
			assert (self.resumeIter_ is None)
			self.solver_.copy_weights(self.preTrainNet_)
		self.expMake_ = True
	
	## Make the deploy file. 
	def make_deploy(self, dataLayerNames, imSz, **kwargs):
		self.deployProto_ = mpu.ProtoDef.deploy_from_proto(self.expFile_.netDef_,
									 dataLayerNames=dataLayerNames, imSz=imSz, **kwargs)
		self.deployProto_.write(self.files_['netdefDeploy'])
	
	## Get the deploy proto
	def get_deploy_proto(self):
		if not(osp.exists(self.files_['netdefDeploy'])):
			self.make_deploy,	
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

	##
	#Visualize the log
	def vis_log(self):
		self.solver_.read_log_from_file()
		self.solver_.plot()
	
	def get_test_accuracy(self):
		print ('NOT IMPLEMENTED YET')


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



