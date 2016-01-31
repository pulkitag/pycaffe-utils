import my_exp_config as mec
from easydict import EasyDict as edict

####### EXAMPLE 1 - CONFIGURING AN MNIST EXPERIMENT #########
##
#Define the experiment, snapshot and other required paths
def get_mnist_paths():
	paths = edict()
	#Path to store experiment details
	paths.exp    = edict()
	paths.exp.dr = './test_data/mnist/exp'
	#Paths to store snapshot details
	paths.exp.snapshot = edict()
	paths.exp.snapshot.dr = './test_data/mnist/snapshots' 
	return paths

##
#Define any parameters that may influence the experiment details
def get_mnist_prms():
	prms = edict()
	prms['expStr'] = 'mnist'
	prms.paths = get_mnist_paths() 
	return prms

##
#Setup the experiment
def setup_experiment():
	prms    = get_mnist_prms()
	nwPrms  = {'netName': 'MyNet', 
						 'baseNetDefProto': 'trainval.prototxt'}
	cPrms   = mec.get_solver_caffe_prms(mec.get_default_net_prms, nwPrms, 
						mec.get_default_solver_prms,
						baseDefDir='./test_data/mnist')	
	exp     = mec.CaffeSolverExperiment(prms, cPrms)
	exp.make()
	return exp

####### END OF EXAMPLE 1 ###################
