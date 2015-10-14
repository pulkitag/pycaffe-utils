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
else:
	cfg.IS_EC2 = True
	cfg.CAFFE_PATH = '/home/ubuntu/caffe-v2-3'
	cfg.STREETVIEW_CODE_PATH = '/home/ubuntu/code/streetview'
	if osp.exists('/data0'):
		cfg.STREETVIEW_DATA_MAIN = '/data0'
	else:
		cfg.STREETVIEW_DATA_MAIN = '/puresan/shared'
