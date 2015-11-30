## @package pycaffe_config 
#  Specify the configurations 
#

import socket
from easydict import EasyDict as edict
from os import path as osp

cfg = edict()
cfg.HOSTNAME = socket.gethostname()
if cfg.HOSTNAME in ['anakin', 'vader', 'spock', 'poseidon']:
	cfg.IS_EC2 = False	
	cfg.CAFFE_PATH = '/work4/pulkitag-code/pkgs/caffe-v2-3'
	cfg.STREETVIEW_CODE_PATH = '/work4/pulkitag-code/code/projStreetView'
	cfg.STREETVIEW_DATA_MAIN = '/data0'
	cfg.STREETVIEW_DATA_READ_IM = cfg.STREETVIEW_DATA_MAIN
	#Billiards Path
	cfg.BILLIARDS_DATA_MAIN = '/data1'
	#Caffe Model Path
	cfg.CAFFE_MODEL_PATH = '/data1/pulkitag/caffe_models/'
else:
	cfg.IS_EC2 = True
	if osp.exists('/home-2/pagrawal'):
		cfg.STREETVIEW_CODE_PATH = '/home-2/pagrawal/code/streetview'
		cfg.CAFFE_PATH = '/home-2/pagrawal/pkgs/caffe-v2-3'
	else:
		cfg.STREETVIEW_CODE_PATH = '/home/ubuntu/code/streetview'
		cfg.CAFFE_PATH = '/home/ubuntu/caffe-v2-3'

	if osp.exists('/data0'):
		cfg.STREETVIEW_DATA_MAIN    = '/data0'
		cfg.STREETVIEW_DATA_READ_IM = cfg.STREETVIEW_DATA_MAIN
		#BILLIARDS PATH
		cfg.BILLIARDS_DATA_MAIN = '/data1'
	else:
		cfg.STREETVIEW_DATA_MAIN    = '/puresan/shared'
		cfg.STREETVIEW_DATA_READ_IM = '/dev/shm'
		#BILLIARDA PATH
		cfg.BILLIARDS_DATA_MAIN = '/puresan/shared'
