This repository provides utilities for conveniently defining and running deep learning experiments using Caffe. Functions in this repository are especially useful for performing parameter sweeps, visualizing and recording results or debugging the training process of nets.

Note: This README is being constantly updated and currently covers only a few functions provided as part of pycaffe-utils.

Dependencies
-------------
This is not an exhaustive list. 
```
sudo apt-get install liblmdb-dev
sudo pip install lmdb
```


Setting up a Caffe Experiment
----------------------------------

There are three main classes of parameters needed to define an experiment:
- What data is to be used (i.e. images/labels) (called <i>dPrms</i>  or data parameters)
- What should be the structure of the network  (called <i>nPrms</i> or network parameters)
- How should the learning proceed (called <i>sPrms</i> or solver parameters)

Details of different experiments (specified by different parameters) are stored in SQL database. The SQL database stores an automatically generated hash string for each parameter setting and that is used to automatically generate and name files that are used to run
the experiment.

#### Specifying <i>dPrms</i>
type: EasyDict

The minimal definition of <i>dPrms</i> is below:
<pre><code>
from easydict import EasyDict as edict
dPrms     =   edict()
dPrms['expStr'] = 'demo-experiment' #The name of the experiment
dPrms.paths     = edict() #The paths that will be used
dPrms.paths.exp    = edict() #Paths for storing experiment files
dPrms.paths.exp.dr = '/directory/for/storing/experiment/files'
dPrms.paths.snapshot    = edict()
dPrms.paths.snapshot.dr = '/directory/for/storing/snapshots'
</code></pre>

#### Specifying <i>nPrms</i>
type: EasyDict

The minimal definition is defined in module <i>my_exp_config</i> in function <i> get_default_net_prms</i>.

Custom <i>nPrms</i> should be defined as following:
(To be updated soon).

#### Specifying <i>sPrms</i>
type: EasyDict

To be updated soon.






Debugging a Caffe Experiment
-------------------------------------------------------------------------

If a deep network is not training, it is instructive to look at how the parameters, gradients and feature values of different layers change with iterations. It is easy to log,
- The parameter values
- The parameter update values (i.e. gradients)
- The feature values

of all the blobs in the net using the following code snippet

<pre><code>
import my_pycaffe as mp
#Define the solver using caffe style solver prototxt
sol      = mp.MySolver.from_file(solver_prototxt)
#Number of iterations after which parameters should be saved to log file
saveIter = 1000  
maxIter  = 200000
#Name of the log file
logFile  = 'log.pkl'
for r in range(0, maxIter, saveIter):
    #Train for saveIter iterations
    sol.solve(saveIter)
    #Save the log file
    sol.dump_to_file(logFile)
</code></pre>

The logged values can be easily plotted using, `sol.plot()`


Creating a Siamese prototxt file for Caffe
-------------------------
<pre><code>
import my_pycaffe_utils as mpu
fName = 'deploy.prototxt'
pDef  = mpu.ProtoDef(fName)
#Make a siamese protodef by duplicating layers between 'conv1' and 'conv5', leave
#other layers as such.
siameseDef = pDef.get_siamese('conv1', 'conv5')
#Save the siamese file
siameseDef.write('siamese.prototxt')
</code></pre>
