## @package pycaffe_config 
#  Specify the configurations 
#

import socket
from easydict import EasyDict as edict

cfg = edict()
cfg.hostname = socket.gethostname()
if cfg.HOSTNAME in ['anakin', 'vader', 'spock', 'poseidon']:
	cfg.IS_EC2 = False	
	cfg.CAFFE_PATH = '/work4/pulkitag-code/pkgs/caffe-v2-3'
else:
	cfg.IS_EC2 = True
	cfg.CAFFE_PATH = '/home/ubuntu/caffe-v2-3'
	
