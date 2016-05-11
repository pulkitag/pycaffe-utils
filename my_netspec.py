## @package my_netspec
#  This was developed indepedently of caffe's netspec in 2014. 
#  This maybe phased out in favor of caffe's netspec in the near
#  future
#

import my_pycaffe_utils as mpu
import os


def make_def_proto(nw, isSiamese=True,
          baseFileStr='split_im.prototxt', getStreamTopNames=False):
  '''
    If is siamese then wait for the Concat layers - and make all layers until then siamese.
  '''
  baseFile = os.path.join(baseFileStr)
  protoDef = mpu.ProtoDef(baseFile)

  #if baseFileStr in ['split_im.prototxt', 'normal.prototxt']:
  lastTop  = 'data'

  siameseFlag = isSiamese
  stream1, stream2 = [], []
  mainStream = []

  nameGen     = mpu.LayerNameGenerator()
  for l in nw:
    lType, lParam = l
    lName         = nameGen.next_name(lType)
    #To account for layers that should not copied while finetuning
    # Such layers need to named differently.
    if lParam.has_key('nameDiff'):
      lName = lName + '-%s' % lParam['nameDiff']
    if lType == 'Concat':
      siameseFlag = False
      if not lParam.has_key('bottom2'):
        lParam['bottom2'] = lastTop + '_p'

    if siameseFlag:
      lDef, lsDef = mpu.get_siamese_layerdef_for_proto(lType, lName, lastTop, **lParam)
      stream1.append(lDef)
      stream2.append(lsDef)
    else:
      lDef = mpu.get_layerdef_for_proto(lType, lName, lastTop, **lParam)
      mainStream.append(lDef)

    if lParam.has_key('shareBottomWithNext'):
      assert lParam['shareBottomWithNext']
      pass
    else:
      lastTop = lName

  #Add layers
  mainStream = stream1 + stream2 + mainStream
  for l in mainStream:
    protoDef.add_layer(l['name'][1:-1], l)

  if getStreamTopNames:
    if isSiamese:
      top1Name = stream1[-1]['name'][1:-1]
      top2Name = stream2[-1]['name'][1:-1]
    else:
      top1Name, top2Name = None, None
    return protoDef, top1Name, top2Name
  else:
    return protoDef


##
# Generates a string to represent the n/w name
def nw2name(nw, getLayerNames=False):
  nameGen     = mpu.LayerNameGenerator()
  nwName   = []
  allNames = []
  for l in nw:
    lType, lParam = l
    lName = nameGen.next_name(lType)
    if lParam.has_key('nameDiff'):
      allNames.append(lName + '-%s' % lParam['nameDiff'])
    else:
      allNames.append(lName)
    if lType in ['InnerProduct', 'Convolution']:
      lName = lName + '-%d' % lParam['num_output']
      if lType == 'Convolution':
        lName = lName + 'sz%d-st%d' % (lParam['kernel_size'], lParam['stride'])
      nwName.append(lName)
    elif lType in ['Pooling']:
      lName = lName + '-sz%d-st%d' % (lParam['kernel_size'],
                    lParam['stride'])
      nwName.append(lName)
    elif lType in ['Concat', 'Dropout', 'Sigmoid']:
      nwName.append(lName)
    elif lType in ['RandomNoise']:
      if lParam.has_key('adaptive_sigma'):
        lName = lName + '-asig%.2f' % lParam['adaptive_factor']
      else:
        lName = lName + '-sig%.2f' % lParam['sigma']
      nwName.append(lName)
    else:
      pass
  nwName = ''.join(s + '_' for s in nwName)
  nwName = nwName[:-1]
  if getLayerNames:
    return nwName, allNames
  else:
    return nwName

##
# This is highly hand engineered to suit my current needs for ICCV submission. 
def nw2name_small(nw, isLatexName=False):
  nameGen   = mpu.LayerNameGenerator()
  nwName    = []
  latexName = []
  for l in nw:
    lType, lParam = l
    lName = ''
    latName = ''
    if lType in ['Convolution']:
      lName   = 'C%d_k%d' % (lParam['num_output'], lParam['kernel_size'])
      latName = 'C%d' % (lParam['num_output'])
      nwName.append(lName)
      latexName.append(latName)
    elif lType in ['InnerProduct']:
      lName = 'F%d' % lParam['num_output']
      nwName.append(lName)
      latexName.append(latName)
    elif lType in ['Pooling']:
      lName = lName + 'P'
      nwName.append(lName)
      latexName.append(lName)
    elif lType in ['Sigmoid']:
      lName = lName + 'S'
      nwName.append(lName)
      latexName.append(lName)
    elif lType in ['Concat']:
      break
    else:
      pass
  nwName = ''.join(s + '-' for s in nwName)
  nwName = nwName[:-1]

  latexName = ''.join(s + '-' for s in latexName)
  latexName = latexName[:-1]

  if isLatexName:
    return nwName, latexName
  else:
    return nwName


