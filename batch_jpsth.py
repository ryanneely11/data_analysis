##this is a script to run the JPSTH algorithm on a batch of recording
##sessions by spawning separate processes to calculate the values for each pair.

import numpy as np
import jpsth_2 as jp
import DataSet3 as ds
import multiprocessing as mp
import RatUnits4 as ru
import GuiStuff as gs
import h5py

def run_batch(f_in, f_out, pre_win, post_win, target, bin = 1):
	##open the input data file
	f = h5py.File(f_in, 'r')
	##ask user what animal to get data from
	animal = gs.onechoice(f.keys(), title = "Select Animal to Analyze")
	##get a list of sessions to use for this animal
	session_list = gs.multichoice(f[animal].keys(), title = "Sessions for " + animal)
	##create a list of all the args to send to the processes
	argList = []
	for session in session_list:
		argList.append([f_in, animal, session, pre_win, post_win, target, bin])
	##create a worker pool to process all of the jpsth's
	pool = mp.Pool(processes = mp.cpu_count())
	async_result = pool.map_async(get_jpsth, argList)
	pool.close()
	pool.join()
	print("All processes have returned; parsing results")
	results_list = async_result.get()
	##parse all of the data and save to the specified output file
	parse_results(results_list, f_out)
	f.close()
	print("Complete!")
		

def get_jpsth(args):
	#parse the argument list
	f_in = args[0]
	animal = args[1]
	session = args[2]
	pre_win = args[3]
	post_win = args[4]
	target = args[5]
	bin = args[6]
	f = h5py.File(f_in, 'r')
	##get the e1 and e2 data from the file
	e1_arrays, e2_arrays = ds.get_ensemble_arrays(f_in, animal=animal, session=session)
	##get the corresponding target times
	events = np.asarray(f[animal][session]['event_arrays'][target])
	e1_1 = ds.get_data_window(events, pre_win, post_win, e1_arrays[0][:,0]).T
	e1_2 = ds.get_data_window(events, pre_win, post_win, e1_arrays[0][:,1]).T
	e2_1 = ds.get_data_window(events, pre_win, post_win, e2_arrays[0][:,0]).T
	e2_2 = ds.get_data_window(events, pre_win, post_win, e2_arrays[0][:,1]).T
	#calculate jpsth's for e1 and e2 pairs
	result_e1 = jp.jpsth(e1_1, e1_2, bin = bin)
	print("finished with jpsth for e1 from session " + session)
	result_e2 = jp.jpsth(e2_1, e2_2, bin = bin)
	print("finished with jpsth for e2 from session " + session)
	return (result_e1, result_e2)

def parse_results(results_list, f_out):
	##open the output file
	g = h5py.File(f_out, 'w-')
	##get the number of sessions
	num_sessions = len(results_list)
	##create master arrays for all of the data
	e1_psth_1 = np.zeros(np.append(num_sessions,results_list[0][0]['psth_1'].shape))
	e1_psth_2 = np.zeros(np.append(num_sessions,results_list[0][0]['psth_1'].shape))
	e1_normalizedJPSTH = np.zeros(np.append(num_sessions,results_list[0][0]['normalizedJPSTH'].shape))
	e1_xcorrHist = np.zeros(np.append(num_sessions,results_list[0][0]['xcorrHist'].shape))
	e1_pstch = np.zeros(np.append(num_sessions,results_list[0][0]['pstch'].shape))
	e1_covariogram = np.zeros(np.append(num_sessions,results_list[0][0]['covariogram'].shape))
	e1_sigLow = np.zeros(np.append(num_sessions,results_list[0][0]['sigLow'].shape))
	e1_sigHigh = np.zeros(np.append(num_sessions,results_list[0][0]['sigHigh'].shape))
	#e1_sigPeakEndpoints = np.zeros(np.append(num_sessions,results_list[0][0]['sigPeakEndpoints'].shape))
	#e1_sigTroughEndpoints = np.zeros(np.append(num_sessions,results_list[0][0]['sigTroughEndpoints'].shape))

	e2_psth_1 = np.zeros(np.append(num_sessions,results_list[0][1]['psth_1'].shape))
	e2_psth_2 = np.zeros(np.append(num_sessions,results_list[0][1]['psth_1'].shape))
	e2_normalizedJPSTH = np.zeros(np.append(num_sessions,results_list[0][1]['normalizedJPSTH'].shape))
	e2_xcorrHist = np.zeros(np.append(num_sessions,results_list[0][1]['xcorrHist'].shape))
	e2_pstch = np.zeros(np.append(num_sessions,results_list[0][1]['pstch'].shape))
	e2_covariogram = np.zeros(np.append(num_sessions,results_list[0][1]['covariogram'].shape))
	e2_sigLow = np.zeros(np.append(num_sessions,results_list[0][1]['sigLow'].shape))
	e2_sigHigh = np.zeros(np.append(num_sessions,results_list[0][1]['sigHigh'].shape))
	#e2_sigPeakEndpoints = np.zeros(np.append(num_sessions,results_list[0][1]['sigPeakEndpoints'].shape))
	#e2_sigTroughEndpoints = np.zeros(np.append(num_sessions,results_list[0][1]['sigTroughEndpoints'].shape))
	##run through all of the data dictionaries and add the data to the master arrays
	for i in range(num_sessions):
		e1_psth_1[i,:] = results_list[i][0]['psth_1']
		e1_psth_2[i,:] = results_list[i][0]['psth_1']
		e1_normalizedJPSTH[i,:,:] = results_list[i][0]['normalizedJPSTH']
		e1_xcorrHist [i,:] = results_list[i][0]['xcorrHist']
		e1_pstch[i,:] = results_list[i][0]['pstch']
		e1_covariogram[i,:] = results_list[i][0]['covariogram']
		e1_sigLow[i,:] = results_list[i][0]['sigLow']
		e1_sigHigh[i,:] = results_list[i][0]['sigHigh']
		#e1_sigPeakEndpoints[i,:] = results_list[i][0]['sigPeakEndpoints']
		#e1_sigTroughEndpoints[i,:] = results_list[i][0]['sigTroughEndpoints']

		e2_psth_1[i,:] = results_list[i][1]['psth_1']
		e2_psth_2[i,:] = results_list[i][1]['psth_1']
		e2_normalizedJPSTH[i,:,:] = results_list[i][1]['normalizedJPSTH']
		e2_xcorrHist [i,:] = results_list[i][1]['xcorrHist']
		e2_pstch[i,:] = results_list[i][1]['pstch']
		e2_covariogram[i,:] = results_list[i][1]['covariogram']
		e2_sigLow[i,:] = results_list[i][1]['sigLow']
		e2_sigHigh[i,:] = results_list[i][1]['sigHigh']
		#e2_sigPeakEndpoints[i,:] = results_list[i][1]['sigPeakEndpoints']
		#e2_sigTroughEndpoints[i,:] = results_list[i][1]['sigTroughEndpoints']

	##save data to the file
	e1_data = g.create_group("e1_data")
	e2_data = g.create_group("e2_data")
	##create individual datasets to mirror the output dictionaries
	e1_data.create_dataset("psth_1", data = e1_psth_1)
	e1_data.create_dataset("psth_2", data = e1_psth_2)
	e1_data.create_dataset("normalizedJPSTH", data = e1_normalizedJPSTH)
	e1_data.create_dataset("xcorrHist", data = e1_xcorrHist)
	e1_data.create_dataset("pstch", data = e1_pstch)
	e1_data.create_dataset("covariogram", data = e1_covariogram)
	e1_data.create_dataset("sigLow", data = e1_sigLow)
	e1_data.create_dataset("sigHigh", data = e1_sigHigh)
	#e1_data.create_dataset("sigPeakEndpoints", data = e1_sigPeakEndpoints)
	#e1_data.create_dataset("sigTroughEndpoints", data = e1_sigTroughEndpoints)

	e2_data.create_dataset("psth_1", data = e2_psth_1)
	e2_data.create_dataset("psth_2", data = e2_psth_2)
	e2_data.create_dataset("normalizedJPSTH", data = e2_normalizedJPSTH)
	e2_data.create_dataset("xcorrHist", data = e2_xcorrHist)
	e2_data.create_dataset("pstch", data = e2_pstch)
	e2_data.create_dataset("covariogram", data = e2_covariogram)
	e2_data.create_dataset("sigLow", data = e2_sigLow)
	e2_data.create_dataset("sigHigh", data = e2_sigHigh)
	#e2_data.create_dataset("sigPeakEndpoints", data = e2_sigPeakEndpoints)
	#e2_data.create_dataset("sigTroughEndpoints", data = e2_sigTroughEndpoints)
	##close the file
	g.close()
	print("data saved!")
