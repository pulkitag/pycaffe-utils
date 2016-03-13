import my_pycaffe as mp
import my_exp_config as mec

REAL_PATH = cfg.REAL_PATH
DEF_DB    = cfg.DEF_DB % ('default', '%s')

#See if the solver is able to load a pretrained net automatically
def test_solver_load_pretrain():
	#Get net prms
	nPrms = mec.get_default_net_prms(dbFile, **kwargs)
	del nPrms['expStr']
	nPrms.baseNetDefProto = 'doublefc-v1_window_fc6'
	nPrms  = mpu.get_defaults(kwargs, nPrms, False)
	nPrms['expStr'] = mec.get_sql_id(dbFile, dArgs, ignoreKeys=['ncpu'])
	
	dPrms   = get_data_prms()
	nwFn    = process_net_prms
	ncpu = 0
	nwArgs  = {'ncpu': ncpu, 'lrAbove': None, 'preTrainNet':None}
	solFn   = mec.get_default_solver_prms
	solArgs = {'dbFile': DEF_DB % 'sol', 'clip_gradients': 10}
	cPrms   = mec.get_caffe_prms(nwFn=nwFn, nwPrms=nwArgs,
									 solFn=solFn, solPrms=solArgs)
	exp     = mec.CaffeSolverExperiment(dPrms, cPrms,
					  netDefFn=make_net_def, isLog=True)
	if isRun:
		exp.make()
		exp.run() 
	return exp 	 				

	
