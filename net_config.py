import os.path as osp
import numpy as np
import my_pycaffe_utils as mpu
from easydict import EasyDict as edict
import copy

##
# Parameters required to specify the n/w architecture
def get_nw_prms(**kwargs):
	dArgs = edict()
	dArgs.netName     = 'alexnet'
	dArgs.concatLayer = 'fc6'
	dArgs.concatDrop  = False
	dArgs.contextPad  = 0
	dArgs.imSz        = 227
	dArgs.imgntMean   = True
	dArgs.maxJitter   = 11
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
	dArgs.expStr = expStr 
	return dArgs 

##
# Parameters that specify the learning
def get_lr_prms(**kwargs):	
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
	print dArgs.keys()
	for k in dArgs.keys():
		if k in ['batchsize']:
			continue
		solArgs[k] = copy.deepcopy(dArgs[k])
	dArgs.solver = mpu.make_solver(**solArgs)	
	expStr = 'batchSz%d_stepSz%.0e_blr%.5f_mxItr%.1e_gamma%.2f_wdecay%.6f'\
					 % (dArgs.batchsize, dArgs.stepsize, dArgs.base_lr,
							dArgs.max_iter, dArgs.gamma, dArgs.weight_decay)
	if not(dArgs.clip_gradients==-1):
		expStr = '%s_gradClip%.1f' % (expStr, dArgs.clip_gradients)
	dArgs.expStr = expStr
	return dArgs 

##
# Parameters for fine-tuning
def get_finetune_prms(**kwargs):
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


