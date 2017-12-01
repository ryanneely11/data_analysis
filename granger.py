##granger.py

##functions to look at granger causality

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec

import nitime
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu
from nitime.viz import drawmatrix_channels

import RatUnits4 as ru
import plxread

root_dir = r"D:\Ryan\V1_BMI"

##load a dataset 
def get_dataset(animal,session,sig_type='lfp',target='t1',window=[3000,1000]):
	global root_dir
	##first, we want to figure out what lfp is available in the file
	region_names = ru.animals[animal][1][session][sig_type].keys()
	event_id = ru.animals[animal][1][session]['events'][target][0]
	##load the file
	plxfile = os.path.join(root_dir,animal,session)
	raw_data = plxread.import_file(plxfile,AD_channels=range(1,97),save_wf=False,
		import_unsorted=False,verbose=False)
	##get the event data
	event_ts = np.squeeze(raw_data[event_id]*1000.0)
	##we want to construct a dataset of dims trials x channels x samples
	##we also want to record the region that each channel belongs to
	##start by figuring out how many channels we have
	chan_names = []
	total_chans = 0
	for region in region_names:
		total_chans += len(ru.animals[animal][1][session][sig_type][region])
	##now initialize the data array
	data = np.zeros((event_ts.shape[0],total_chans,window[0]+window[1]))
	##for each channel in each region, add data to the array.
	sig_count = 0 ##keep track of the total signals we have processed across all regions
	for region in region_names:
		signals = ru.animals[animal][1][session][sig_type][region]
		for sig in signals:
			sig_data = load_AD(raw_data,sig)
			trial_data = get_data_windows(sig_data,event_ts,window[0],window[1])
			##now add it to the master array
			data[:,sig_count,:] = trial_data.T #(need to swap the axes here)
			##make up a name for this channel
			chan_names.append(region[0:3]+sig[-2:])
			sig_count += 1
	return data,chan_names

##a function to run and plot all of the basic example analyses.
def analyze(data,chan_names):
	TR = 1000.0 ##sample rate
	f_ub = 100 ##upper freq bound
	f_lb = 0 ##lower freq bound
	##compute the result for each trial
	coh = []
	g1 = []
	g2 = []
	for trial in range(data.shape[0]):
		trialdata = data[trial,:,:]
		time_series = ts.TimeSeries(trialdata, sampling_rate=TR)
		G = nta.GrangerAnalyzer(input=time_series,order=2)
		C1 = nta.CoherenceAnalyzer(time_series)
		C2 = nta.CorrelationAnalyzer(time_series)
		freq_idx_G = np.where((G.frequencies > f_lb) * (G.frequencies < f_ub))[0]
		freq_idx_C = np.where((C1.frequencies > f_lb) * (C1.frequencies < f_ub))[0]
		coh.append(np.nanmean(C1.coherence[:, :, freq_idx_C], -1))  # Averaging on the last dimension
		g1.append(np.nanmean(G.causality_xy[:, :, freq_idx_G], -1))
		g2.append(np.nanmean(G.causality_xy[:, :, freq_idx_G] - G.causality_yx[:, :, freq_idx_G], -1))
	##now average across trials
	coh = np.nanmean(np.asarray(coh),axis=0)
	g1 = np.nanmean(np.asarray(g1),axis=0)
	g2 = np.nanmean(np.asarray(g2),axis=0)
	fig01 = drawmatrix_channels(coh, chan_names, size=[10., 10.], title='Coherence')
	fig02 = drawmatrix_channels(C2.corrcoef, chan_names, size=[10., 10.], title="Correlations")
	fig03 = drawmatrix_channels(g1.T, chan_names, size=[10., 10.],title="Forward causality")
	fig04 = drawmatrix_channels(g2.T, chan_names, size=[10., 10.], title="Reverse causality")

##a helper function to load A/D data and correct for any gaps in 
##the recording
def load_AD(raw_data,AD_name):
	##the voltage signal
	tempdata = raw_data[AD_name]
	##the timestamps of each A/D sample
	sigts = raw_data[AD_name+"_ts"]
	#convert the ad ts to samples, and integers for indexing
	sigts = np.ceil((sigts*1000)).astype(int)
	##construct the resulting array, and put the voltage sigs in their place
	sigdata = np.zeros(sigts.shape[0]+1000)
	sigdata[sigts] = tempdata
	return sigdata


### a helper function to take data in a window around a number of timestamps (centers)
def get_data_windows(lfp, centers, pre_win, post_win):
	verbose = True
	centers = np.squeeze(np.asarray(centers)).astype(np.int64)
	data = np.squeeze(lfp)
	N = data.size
	removed = 0
	for j, center in enumerate(centers):
		if center <= pre_win or center + post_win >= N:
			centers[j] = centers[j-1]
			removed +=1
			if verbose:
				print("Index too close to start or end to take a full window. Deleting event at "+str(center))
	if removed >= centers.size:
		traces = None
	else:
		traces = np.zeros((pre_win+post_win, len(centers)))
		##the actual windowing functionality:
		for n, idx in enumerate(centers):
				traces[:,n] = data[idx-pre_win:idx+post_win]
	return traces