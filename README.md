This repository provides utilities for conveniently defining and running deep learning experiments using Caffe. Functions in this repository are especially useful for performing parameter sweeps, visualizing and recording results or debugging the training process of nets.

Note: This README is being constantly updated and currently covers only a few functions provided as part of pycaffe-utils.

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
