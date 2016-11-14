## @package vis_utils
#  Miscellaneous Functions for visualizations
#
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt
import copy
import os
import pdb
try:
  import caffe
  import my_pycaffe_utils as mpu
  import my_pycaffe as mp
  import my_pycaffe_io as mpio
except:
  print ('CAFFE PACKAGES CANNOT BE LOADED')
import scipy
from matplotlib import gridspec

TMP_DATA_DIR = '/data1/pulkitag/others/caffe_tmp_data/'

##
#Plot n images
def plot_n_ims(ims, fig=None, titleStr='', figTitle='',
				 axTitles = None, subPlotShape=None,
				 isBlobFormat=False, chSwap=None, trOrder=None,
         showType=None):
  '''
    ims: list of images
    isBlobFormat: Caffe stores images as ch x h x w
                  True - convert the images into h x w x ch format
    trOrder     : If certain transpose order of channels is to be used
                  overrides isBlobFormat
    showType    : imshow or matshow (by default imshow)
  '''
  ims = copy.deepcopy(ims)
  if trOrder is not None:
    for i, im in enumerate(ims):
      ims[i] = im.transpose(trOrder)
  if trOrder is None and isBlobFormat:
    for i, im in enumerate(ims):
      ims[i] = im.transpose((1,2,0))
  if chSwap is not None:
    for i, im in enumerate(ims):
      ims[i] = im[:,:,chSwap]
  plt.ion()
  if fig is None:
    fig = plt.figure()
  plt.figure(fig.number)
  plt.clf()
  if subPlotShape is None:
    N = np.ceil(np.sqrt(len(ims)))
    subPlotShape = (N,N)
    #gs = gridspec.GridSpec(N, N)
  ax = []
  for i in range(len(ims)):
    shp = subPlotShape + (i+1,)
    aa  = fig.add_subplot(*shp)
    aa.autoscale(False)
    ax.append(aa)
    #ax.append(plt.subplot(gs[i]))

  if showType is None:
    showType = ['imshow'] * len(ims)
  else:
    assert len(showType) == len(ims)

  for i, im in enumerate(ims):
    ax[i].set_ylim(im.shape[0], 0)
    ax[i].set_xlim(0, im.shape[1])
    if showType[i] == 'imshow':
      ax[i].imshow(im.astype(np.uint8))
    elif showType[i] == 'matshow':
      res = ax[i].matshow(im)
      plt.colorbar(res, ax=ax[i])
    ax[i].axis('off')
    if axTitles is not None:
      ax[i].set_title(axTitles[i])
  if len(figTitle) > 0:
    fig.suptitle(figTitle)
  plt.show()
  return ax


def plot_pairs(im1, im2, **kwargs):
	ims = []
	ims.append(im1)
	ims.append(im2)
	return plot_n_ims(ims, subPlotShape=(1,2), **kwargs)
	

##
#Plot pairs of images from an iterator_fun
def plot_pairs_iterfun(ifun, **kwargs):
	'''
		ifun  : iteration function
		kwargs: look at input arguments for plot_pairs
	'''
	plt.ion()
	fig = plt.figure()
	pltFlag = True
	while pltFlag:
		im1, im2 = ifun()
		plot_pairs(im1, im2, fig=fig, **kwargs)
		ip = raw_input('Press Enter for next pair')
		if ip == 'q':
			pltFlag = False	


class MyAnimation(object):
  def __init__(self, vis_func, frames=100, fps=20, height=200, width=200, fargs=[]):
    self.frames = frames
    self.vis_func = vis_func
    self.vis_func_args = fargs
    self.fps = fps
    self.fig, self.ax = plt.subplots(1,1)
    plt.show(block=False)
    self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
    im = np.zeros((height, width, 3)).astype(np.uint8)
    self.image_obj = self.ax.imshow(im)
    self.fig.canvas.draw()

  def __del__(self):
    plt.close(self.fig)

  def run(self, fargs=[]):
    if len(fargs)==0:
      func_args = self.vis_func_args
    else:
      func_args = fargs
    time_diff = float(1.0)/self.fps
    for i in range(self.frames):
      op = self.vis_func(i, *func_args)
      if type(op) == tuple:
        im, is_stop = op
      else:
        im = op
        is_stop = False
      self._display(im)
      time.sleep(time_diff)
      if is_stop:
        break
      
  def _display(self, pixels):
    self.image_obj.set_data(pixels)
    self.fig.canvas.restore_region(self.bg)
    self.ax.draw_artist(self.image_obj)
    self.fig.canvas.blit(self.ax.bbox)


def draw_square_on_im(im, sq, width=4, col='w'):
  x1, y1, x2, y2 = sq
  h = im.shape[0]
  w = im.shape[1]
  if col == 'w':
    col = (255 * np.ones((1,1,3))).astype(np.uint8)
  elif col == 'r':
    col = np.zeros((1,1,3))
    col[0,0,0] = 255
    col =  col.astype(np.uint8)
  #Top Line
  im[max(0,int(y1-width/2)):min(h, y1+int(width/2)),x1:x2,:] = col
  #Bottom line
  im[max(0,int(y2-width/2)):min(h, y2+int(width/2)),x1:x2,:] = col
  #Left line
  im[y1:y2, max(0,int(x1-width/2)):min(h, x1+int(width/2))] = col
  #Right line
  im[y1:y2, max(0,int(x2-width/2)):min(h, x2+int(width/2))] = col
  return im


##
# Visualize GenericWindowDataLayer file
def vis_generic_window_data(protoDef, numLabels, layerName='window_data', phase='TEST',
												maxVis=100):
	'''
		layerName: The name of the generic_window_data layer
		numLabels: The number of labels. 
	'''
	#Just write the data part of the file. 
	if not isinstance(protoDef, mpu.ProtoDef):
		protoDef = mpu.ProtoDef(protoDef)
	protoDef.del_all_layers_above(layerName)
	randInt = np.random.randint(1e+10)
	outProto = os.path.join(TMP_DATA_DIR, 'gn_window_%d.prototxt' % randInt)
	protoDef.write(outProto)		
	#Extract the name of the data and the label blobs. 
	dataName  = protoDef.get_layer_property(layerName, 'top', propNum=0)[1:-1]
	labelName = protoDef.get_layer_property(layerName, 'top', propNum=1)[1:-1]
	crpSize   = int(protoDef.get_layer_property(layerName, ['crop_size']))
	mnFile    = protoDef.get_layer_property(layerName, ['mean_file'])[1:-1]
	mnDat     = mpio.read_mean(mnFile)
	ch,nr,nc  = mnDat.shape
	xMn       = int((nr - crpSize)/2)
	mnDat     = mnDat[:,xMn:xMn+crpSize,xMn:xMn+crpSize]
	print mnDat.shape
	
	#Create a network	
	if phase=='TRAIN':	
		net     = caffe.Net(outProto, caffe.TRAIN)
	else:
		net     = caffe.Net(outProto, caffe.TEST)

	lblStr = ''.join('lb-%d: %s, ' % (i,'%.2f') for i in range(numLabels))
	figDt = plt.figure()
	plt.ion()
	for i in range(maxVis):
		allDat  = net.forward([dataName,labelName])
		imData  = allDat[dataName] + mnDat
		lblDat  = allDat[labelName]
		batchSz = imData.shape[0]
		for b in range(batchSz):
			#Plot network data. 
			im1 = imData[b,0:3].transpose((1,2,0))
			im2 = imData[b,3:6].transpose((1,2,0))
			im1 = im1[:,:,[2,1,0]]
			im2 = im2[:,:,[2,1,0]]
			lb  = lblDat[b].squeeze()
			lbStr = lblStr % tuple(lb)	
			plot_pairs(im1, im2, figDt, lbStr) 
			raw_input()


def rec_fun_grad(x, myNet, blobDat, blobLbl, shp, lamda):
	'''
		Consider one batch a time. 
	'''
	#print x.shape, blobDat.shape, blobLbl.shape, lamda
	#Put the data
	myNet.net_.net.set_input_arrays(blobDat, blobLbl)
	print shp
	#Get the Error
	feats, diffs = myNet.net_.forward_backward_all(blobs=['loss'],
									 diffs=['data'],data=x.reshape(shp))
	grad         = diffs['data'] + lamda * x.reshape(shp)
	batchLoss    = feats['loss'][0] + 0.5 * lamda * np.dot(x,x)

	grad = grad.flatten()
	return batchLoss, grad


def reconstruct_optimal_input(exp, modelIter, im, recLayer='conv1', 
								imH=101, imW=101, cropH=101, cropW=101, channels=3,
								meanFile=None, lamda=1e-8, batchSz=1, **kwargs):
	exp = copy.deepcopy(exp)	
	kwargs['delAbove'] = recLayer

	#Setup the original network	
	origNet = mpu.CaffeTest.from_caffe_exp(exp)
	origNet.setup_network(opNames=recLayer, imH=imH, imW=imW, cropH=cropH, cropW=cropW,
								modelIterations=modelIter, batchSz=batchSz,
								isAccuracyTest=False, meanFile=meanFile, **kwargs)

	#Get the size of the features in the layer that needs to be reconstructed
	#Shape of the layer to be reconstructed
	blob = origNet.net_.net.blobs[recLayer]
	initBlobDat = np.zeros((blob.num, blob.channels, blob.height, blob.width)).astype('float32')
	blobLbl  = np.zeros((blob.num, 1, 1, 1)).astype('float32')
	recShape = (blob.num, blob.channels, blob.height, blob.width) 

	#Get the initial layer features
	print "Extracting Initial Features"
	blobDat = np.zeros((blob.num, blob.channels, blob.height, blob.width)).astype('float32')
	if im.ndim == 3:
		imIp    = im.reshape((batchSz,) + im.shape)
	else:
		imIp = im	
	feats   = origNet.net_.forward_all(blobs=[recLayer], data=imIp)
	blobDat = feats[recLayer]
	#imDat   = np.asarray(imDat)

	#Get the net for reconstruvtions
	#print (exp.expFile_.netDef_.get_all_layernames())
	recProto = mpu.ProtoDef.recproto_from_proto(exp.expFile_.netDef_, featDims=recShape,
								imSz=[[channels, cropH, cropW]], batchSz=batchSz, **kwargs)
	recProto.write(exp.files_['netdefRec'])
	recModel = exp.get_snapshot_name(modelIter)
	recNet   = edict()
	recNet.net_ = mp.MyNet(exp.files_['netdefRec'], recModel, caffe.TRAIN)
	#recNet   = mpu.CaffeTest.from_model(exp.files_['netdefRec'], recModel)		
	#kwargs['dataLayerNames'] = ['data']
	#kwargs['newDataLayerNames'] = None
	#recNet.setup_network(opNames=[recLayer], imH=imH, imW=imW, cropH=cropH, cropW=cropW,
	#				 modelIterations=modelIter, isAccuracyTest=False, meanFile=meanFile,
	#				 testMode=False, **kwargs)
	recNet.net_.net.set_force_backward(recLayer)
	#Start the reconstruction
	ch,h,w = imIp.shape[3], imIp.shape[1], imIp.shape[2]
	imRec  = 255*np.random.random((batchSz,ch,h,w)).astype('float32')
	print imRec.shape, blobDat.shape, blobLbl.shape
	sol = scipy.optimize.fmin_l_bfgs_b(rec_fun_grad, imRec.flatten(),args=[recNet, blobDat, blobLbl, imRec.shape, lamda], maxfun=1000, factr=1e+7,pgtol=1e-07, iprint=0, disp=1)

	imRec = np.reshape(sol[0],((batchSz,ch,h,w)))
	#imRec = im2visim(np.copy(imRec))
	#imGt  = im2visim(np.copy(imDat))
	return imRec, imGt	
