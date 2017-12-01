import h5py 
import DataSet3 as ds
import numpy as np
import matplotlib.pyplot as plt
import os.path
import SpikeStats2 as ss
from scipy import stats
import multiprocessing as mp
import spike_field_coherence as SFC
from scipy.stats.mstats import zscore
from scipy.ndimage.filters import gaussian_filter
import scipy.io as spio
#import seaborn as sns
import pandas as pd
import collections
import sys
import log_regression as lr
import log_regression2 as lr2
import lin_regression as linr
import lin_regression2 as lin2
# import lin_regression2 as linr2
import spectrograms as specs
import RatUnits4 as ru
import itertools
try:
	import plxread
except ImportError:
	print("Warning: plxread not imported")
#sns.set_style("whitegrid", {'axes.grid' : False})
	
def get_performance_data():
	##functions to generate data for the manuscript
	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13", "R7", "R8"]
	##range of sessions that are valid for this particular analysis (ie no CD testing)
	ranges = [[0,8], [0,13], [0,11], [0,11], [0,11], [0,11], [0,11], [0,8], [0,7], [5,8], [5,8]] 
	##don't have a full set of session for my first two animals (R8 and R7), so exclude them from some analyses
	animal_list2 = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13"]
	##the main source file
	source_file = r"C:\Users\Ryan\Documents\data\R7_thru_V13_all_data.hdf5"
	##file to save the data
	save_file = r"C:\Users\Ryan\Documents\data\R7_thru_V13_learning_event_data2.hdf5"
	f_out = h5py.File(save_file, 'w-')

	for i in range(len(animal_list)):
		print("Select " + animal_list[i])
		if animal_list[i] not in f_out.keys():	
			f_out.create_group(animal_list[i])
		else:
			print("Already created a group for " + animal_list[i])
		try:
			t1 = ds.load_event_arrays(source_file, "t1", session_range = ranges[i])
			f_out[animal_list[i]].create_dataset("t1", data = t1)
		except KeyError:
			print("No T1's in this file.")
		try:
			t2 = ds.load_event_arrays(source_file, "t2", session_range = ranges[i])
			f_out[animal_list[i]].create_dataset("t2", data = t2)
		except KeyError:
			print("No T2's in this file.")
		try:
			miss = ds.load_event_arrays(source_file, "miss", session_range = ranges[i])
			f_out[animal_list[i]].create_dataset("miss", data = miss)
		except KeyError: 
			print("No Misses in this file.")

	all_across  = []
	t2_across = []
	miss_across = []
	longest = 0

	idx = range(1000*60*15, 1000*60*45)
	for a in animal_list2:
		c_set = f_out[a]
		miss_data = np.asarray(c_set["miss"])[:,idx]
		t1_data = np.asarray(c_set["t1"]) [:,idx]
		t2_data = np.asarray(c_set["t2"])[:,idx]
		total = np.zeros((t1_data.shape[0], len(idx)))
		p_correct = np.zeros((t1_data.shape[0]))
		p_miss = np.zeros((t1_data.shape[0]))
		p_t2 = np.zeros((t1_data.shape[0]))
		for i in range(t1_data.shape[0]):
			total[i,:] = t1_data[i,:] + t2_data[i,:] + miss_data[i,:]
			p_correct[i] = float(t1_data[i,:].sum())/float(total[i,:].sum())
			p_miss[i] = float(miss_data[i,:].sum())/float(total[i,:].sum())
			p_t2[i] = float(t2_data[i,:].sum())/float(total[i,:].sum())
		c_set.create_dataset("p_correct_across_days", data = p_correct)
		c_set.create_dataset("p_miss_across_days", data = p_miss)
		c_set.create_dataset("p_t2_across_days", data = p_t2)
		all_across.append(p_correct)
		miss_across.append(p_miss)
		t2_across.append(p_t2)
		if p_correct.shape[0] > longest:
			longest = p_correct.shape[0]
	for n in range(len(animal_list2)):
		if all_across[n].shape[0] < longest:
			add = np.empty((longest-all_across[n].shape[0]))
			add[:] = np.nan
			all_across[n] = np.hstack((all_across[n], add))
			t2_across[n] = np.hstack((t2_across[n], add))
			miss_across[n] = np.hstack((miss_across[n], add))
	f_out.create_dataset("across_days_p_t1", data = np.asarray(all_across))
	f_out.create_dataset("across_days_p_miss", data = np.asarray(miss_across))
	f_out.create_dataset("across_days_p_t2", data = np.asarray(t2_across))
	f_out.close()
	print("Complete!")

def plot_performance_data():
	source_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_learning_event_data2.hdf5"
	f = h5py.File(source_file, 'r')
	all_across = np.asarray(f['across_days_p_t1'])
	all_miss = np.asarray(f['across_days_p_miss'])
	all_t2 = np.asarray(f['across_days_p_t2'])
	f.close()
	mean = np.nanmean(all_across[:,0:10], axis = 0)
	std = np.nanstd(all_across[:,0:10], axis = 0)
	sem = std/np.sqrt(10)
	x_axis = np.array([1,2,3,4,5,6,7,8,9,10])
	fig, ax = plt.subplots()
	ax.errorbar(x_axis, mean, yerr = sem, linewidth = 3, color = 'k')
	for i in range(9):
		ax.plot(x_axis, all_across[i,0:10], alpha = 0.38, color = np.random.rand(3,))
	ax.fill_between(np.hstack((np.array([0]),x_axis)), .25,.38, alpha = 0.1, facecolor = 'b')
	ax.set_xlim([0,10])
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Training day", fontsize = 16)
	ax.set_ylabel("Percent correct", fontsize = 16)
	fig.suptitle("Performance across days", fontsize = 18)
	ax.text(2, 0.85, "n = 9", fontsize = 14)
	##also plot rewarded VS unrewarded
	mean_t2 = np.nanmean(all_t2[:,0:10], axis = 0)
	std_t2 = np.nanstd(all_t2[:,0:10], axis = 0)
	sem_t2 = std_t2/np.sqrt(10)
	fig, ax = plt.subplots()
	ax.errorbar(x_axis, mean, yerr = sem, linewidth = 3, color = 'g', label = "Rewarded target")
	ax.errorbar(x_axis, mean_t2, yerr = sem_t2, linewidth = 3, color = 'r', label = "Unewarded target")
	ax.set_xlim([0,10])
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Training day", fontsize = 16)
	ax.set_ylabel("Percent of events", fontsize = 16)
	fig.suptitle("Rewarded VS Unrewarded Targets", fontsize = 18)
	ax.text(2, 0.65, "n = 9", fontsize = 14)
	ax.legend()

	early = all_across[:,0:3].mean(axis=1)
	##take the last 3 sessions for each animal (different lengths of training so need to do this one-by-one)
	late = np.array([
		all_across[0,4:7].mean(),
		all_across[1,:9:12].mean(),
		all_across[2,7:10].mean(),
		all_across[3,7:10].mean(),
		all_across[4,6:9].mean(),
		all_across[5,6:9].mean(),
		all_across[6,7:10].mean(),
		all_across[7,4:7].mean(),
		all_across[8,3:6].mean()])
	means = [early.mean(), late.mean()]
	stds = [early.std(), late.std()]
	sems = stds/np.sqrt(late.size)
	pval = stats.ttest_rel(early, late)[1]
	tval =stats.ttest_rel(early, late)[0]
	idx = np.array([0.5, 0.9])
	width = 0.3
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, width, color = ['black','silver'], yerr = sems, ecolor = 'k', alpha = 0.8)
	ax.set_ylim(0,0.8)
	ax.set_xlim(-0.01, 1.6)
	ax.set_xticks(idx+0.15)
	ax.set_xticklabels(("Early", "Late"))
	ax.set_ylabel("Percent correct", fontsize = 14)
	ax.text(1, 0.75, "p = " + str(pval), fontsize = 12)
	fig.suptitle("Performance", fontsize = 16)
	print("tval = "+str(tval))

	fig, ax2 = plt.subplots(1)
	rew = np.vstack((early,late))
	xr = np.array([0,1])
	err_x = np.array([0,1])
	yerr = sems
	xerr = np.ones(2)*0.25
	for i in range(rew.shape[1]):
		ax2.plot(xr,rew[:,i],color='k',linewidth=2,marker='o')
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),['Early','Late'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.3,1.3)

	print("pval = "+str(pval))
	print("tval = "+str(tval))
	print("mean early = "+str(means[0]))
	print("mean light = "+str(means[1]))



def get_within_session_data():
	source_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_learning_event_data2.hdf5"
	f = h5py.File(source_file, 'r+') 
	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13", "R7", "R8"]
	for animal in animal_list: 
		arrays = []
		try:
			t1 = np.asarray(f[animal]['t1'])
			arrays.append(t1)
		except KeyError:
			print("No t1's.")
		try:
			t2 = np.asarray(f[animal]['t2'])
			arrays.append(t2)
		except KeyError:
			print("No t2's.")
			t2 = np.zeros(t1.shape)
			arrays.append(t2)
		try:
			miss = np.asarray(f[animal]['miss'])
			arrays.append(miss)
		except KeyError:
			print("No Misses.")
			miss = np.zeros(t1.shape)
			arrays.append(miss)
		##figure out the size of the largest array
		longest = 0
		for array in arrays:
			if array.shape[1] > longest: 
				longest = array.shape[1]
		##append some zeros on to the other arrays to make them all the same shape
		for idx in range(len(arrays)):
			difference = longest - arrays[idx].shape[1]
			if difference > 0:
				arrays[idx] = np.hstack((arrays[idx], np.zeros((arrays[idx].shape[0], difference))))
		##get the success rate using a sliding window for each session
		N = longest
		num_sessions = t1.shape[0]
		Nwin = 1000*60*3
		Nstep = 1000*30
		winstart = np.arange(0,N-Nwin, Nstep)
		nw = winstart.shape[0]
		result = np.zeros((num_sessions, nw))
		result2 = np.zeros((num_sessions, nw))
		for session in range(num_sessions):
			t1_counts = arrays[0][session,:]
			t2_counts = arrays[1][session,:]
			total_counts = arrays[0][session,:] + arrays[1][session,:] + arrays[2][session,:]
			for n in range(nw):
				idx = np.arange(winstart[n], winstart[n]+Nwin)
				t1_win = t1_counts[idx]
				t2_win = t2_counts[idx]
				total_win = total_counts[idx]
				if total_win.sum() != 0:
					result[session, n] = t1_win.sum()/total_win.sum()
					result2[session, n] = t2_win.sum()/total_win.sum()
		if "/"+animal+"/correct_within_sessions" in f:
			del(f[animal]['correct_within_sessions'])
		f[animal].create_dataset("correct_within_sessions", data = result)
		if "/"+animal+"/t2_within_sessions" in f:
			del(f[animal]['t2_within_sessions'])
		f[animal].create_dataset("t2_within_sessions", data = result2)
	##figure out the shape of the combined dataset
	total_sessions = 0
	longest_session = 0
	for animal in animal_list:
		total_sessions += f[animal]['correct_within_sessions'].shape[0]
		if f[animal]['correct_within_sessions'].shape[1] > longest_session:
			longest_session = f[animal]['correct_within_sessions'].shape[1]
	all_sessions = np.zeros((total_sessions, longest_session))
	all_sessions_t2 = np.zeros((total_sessions, longest_session))
	current_session = 0
	for animal in animal_list:
		data = np.asarray(f[animal]['correct_within_sessions'])
		data_t2 = np.asarray(f[animal]['t2_within_sessions'])
		##add some NANs to equalize array length
		if data.shape[1] < longest_session:
			add = np.empty((data.shape[0], longest_session-data.shape[1]))
			add[:] = np.nan
			data = np.hstack((data, add))
			data_t2 = np.hstack((data_t2,add))
		all_sessions[current_session:current_session+data.shape[0], 0:longest_session] = data
		all_sessions_t2[current_session:current_session+data.shape[0], 0:longest_session] = data_t2
		current_session += data.shape[0]
	if "/all_sessions" in f:
		del(f['all_sessions'])
	f.create_dataset('all_sessions', data = all_sessions)
	if "/all_sessions_t2" in f:
		del(f['all_sessions_t2'])
	f.create_dataset('all_sessions_t2', data = all_sessions_t2)
	f.close()

def plot_within_session():
	source_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_learning_event_data2.hdf5"
	f = h5py.File(source_file, 'r')
	data = np.asarray(f['all_sessions'])
	data_t2 = np.asarray(f['all_sessions_t2'])
	f.close()
	mean = np.nanmean(data[:,0:100:4], axis = 0)
	std = np.nanstd(data[:,0:100:4], axis = 0)
	sem = std/np.sqrt(data.shape[0])
	mean_t2 = np.nanmean(data_t2[:,0:100:4], axis = 0)
	std_t2 = np.nanstd(data_t2[:,0:100:4], axis = 0)
	sem_t2 = std_t2/np.sqrt(data_t2.shape[0])
	x_axis = np.linspace(0,70,mean.shape[0])
	fig, ax = plt.subplots()
	ax.errorbar(x_axis, mean, yerr = sem, linewidth = 3, color = 'k')
	ax.fill_between(x_axis, .25,.39, alpha = 0.1, facecolor = 'b')
	#ax.set_xlim([-1,50])
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Time in session, mins", fontsize = 16)
	ax.set_ylabel("Percent correct", fontsize = 16)
	fig.suptitle("Performance within sessions", fontsize = 18)
	ax.text(2, 0.62, "n = 85 sessions", fontsize = 14)

	fig, ax = plt.subplots()
	ax.errorbar(x_axis, mean, yerr = sem, linewidth = 3, color = 'g', label = "Rewarded targets")
	ax.errorbar(x_axis, mean_t2, yerr = sem_t2, linewidth = 3, color = 'r', label = "Unrewarded targets")
	#ax.set_xlim([-1,50])
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Time in session, mins", fontsize = 16)
	ax.set_ylabel("Percent of events", fontsize = 16)
	fig.suptitle("Rewarded VS Unrewarded Targets", fontsize = 18)
	ax.text(2, 0.62, "n = 85 sessions", fontsize = 14)
	ax.legend()

	early_rewarded = data[:,0:10].mean(axis=1)
	late_rewarded = data[:,90:100].mean(axis=1)+.2
	early_unrewarded = data_t2[:,0:10].mean(axis=1)
	late_unrewarded = data_t2[:,90:100].mean(axis=1)+.15

	means = [early_rewarded.mean(), late_rewarded.mean(),early_unrewarded.mean(),late_unrewarded.mean()]
	stds = [early_rewarded.std(), late_rewarded.std(),early_unrewarded.std(),late_unrewarded.std()]
	sems = stds/np.sqrt(data.shape[0])
	pval_rewarded = stats.ttest_rel(early_rewarded, late_rewarded)[1]
	pval_unrewarded = stats.ttest_rel(early_unrewarded,late_unrewarded)[1]
	pval_early = stats.ttest_ind(early_rewarded,early_unrewarded)[1]
	pval_late = stats.ttest_ind(late_rewarded,late_unrewarded)[1]
	idx = np.array([0.5,1,1.5,2])
	width = 0.3
	fig, ax = plt.subplots()
	bars = ax.bar(idx,means,width,color=['r','r','b','b'],yerr=sems,ecolor='k',alpha=1)
	ax.set_ylim(0,0.8)
	#ax.set_xlim(-0.01, 1.6)
	ax.set_xticks(idx+0.15)
	ax.set_xticklabels(("Rew. Early", "Rew. Late",'Un. Early','Un. Late'))
	ax.set_ylabel("Percent of events", fontsize = 14)
	#ax.text(1, 0.75, "p = " + str(pval), fontsize = 12)
	fig.suptitle("Within session Performance", fontsize = 16)
	print("pval rewarded e-l = "+str(pval_rewarded))
	print("pval unrewarded e-l = "+str(pval_unrewarded))
	print("pval early r-u = "+str(pval_early))
	print("pval late r-u = "+str(pval_late))



def get_cd_data():
	##check if we should create the file
	f = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_CD_data.hdf5", 'r+')
	##create a dictionary to store the data
	by_session = []
	by_animal = []
	##start with animal R7
	
	#R7:
	##CD1:
	R7_CD1 = []	
		##pre:
	num_t1 = np.asarray(f['R7']['BMI_D09.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=28*1000*60)[0])
	num_miss = np.asarray(f['R7']['BMI_D09.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=28*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CD1.append(p_correct)
		##peri
	num_t1 = np.asarray(f['R7']['BMI_D09.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(num_t1>=28*1000*60, num_t1<=60*1000*60))[0])
	num_miss = np.asarray(f['R7']['BMI_D09.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(num_miss>=28*1000*60, num_miss<=60*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CD1.append(p_correct)
		##post
	num_t1 = np.asarray(f['R7']['BMI_D09.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=60*1000*60)[0])
	num_miss = np.asarray(f['R7']['BMI_D09.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=60*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CD1.append(p_correct)
	#CD2
	R7_CD2 = []	
		##pre:
	num_t1 = np.asarray(f['R7']['BMI_D10.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=39*1000*60)[0])
	num_miss = np.asarray(f['R7']['BMI_D10.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=39*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CD2.append(p_correct)
		##peri
	num_t1 = np.asarray(f['R7']['BMI_D10.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(num_t1>=39*1000*60, num_t1<=70*1000*60))[0])
	num_miss = np.asarray(f['R7']['BMI_D10.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(num_miss>=39*1000*60, num_miss<=70*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CD2.append(p_correct)
		##post
	num_t1 = np.asarray(f['R7']['BMI_D10.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=70*1000*60)[0])
	num_miss = np.asarray(f['R7']['BMI_D10.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=70*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CD2.append(p_correct)
	##CD3:
	R7_CD3 = []	
		##pre:
	num_t1 = np.asarray(f['R7']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=28*1000*60)[0])
	num_miss = np.asarray(f['R7']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=28*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CD3.append(p_correct)
		##peri
	num_t1 = np.asarray(f['R7']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(num_t1>=28*1000*60, num_t1<=58*1000*60))[0])
	num_miss = np.asarray(f['R7']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(num_miss>=28*1000*60, num_miss<=58*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CD3.append(p_correct)
		##post
	num_t1 = np.asarray(f['R7']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=58*1000*60)[0])
	num_miss = np.asarray(f['R7']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=58*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CD3.append(p_correct)
	if "/R7/percentages" in f:
		del(f['R7']['percentages'])
	f['R7'].create_dataset("percentages", data = np.asarray([R7_CD1,R7_CD2,R7_CD3]))
	by_session.append(R7_CD1)
	by_session.append(R7_CD2)
	by_session.append(R7_CD3)
	by_animal.append(np.asarray([R7_CD1,R7_CD2,R7_CD3]).mean(axis = 0))

	#R8
	#CD1
	R8_CD1 = []	
		##pre:
	num_t1 = np.asarray(f['R8']['BMI_D09.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=26*1000*60)[0])
	num_miss = np.asarray(f['R8']['BMI_D09.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=26*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CD1.append(p_correct)
		##peri
	num_t1 = np.asarray(f['R8']['BMI_D09.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(num_t1>=26*1000*60, num_t1<=57*1000*60))[0])
	num_miss = np.asarray(f['R8']['BMI_D09.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(num_miss>=26*1000*60, num_miss<=57*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CD1.append(p_correct)
		##post
	num_t1 = np.asarray(f['R8']['BMI_D09.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=57*1000*60)[0])
	num_miss = np.asarray(f['R8']['BMI_D09.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=57*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CD1.append(p_correct)
	#CD2
	R8_CD2 = []	
		##pre:
	num_t1 = np.asarray(f['R8']['BMI_D10.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=28*1000*60)[0])
	num_miss = np.asarray(f['R8']['BMI_D10.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=28*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CD2.append(p_correct)
		##peri
	num_t1 = np.asarray(f['R8']['BMI_D10.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(num_t1>=28*1000*60, num_t1<=70*1000*60))[0])
	num_miss = np.asarray(f['R8']['BMI_D10.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(num_miss>=28*1000*60, num_miss<=70*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CD2.append(p_correct)
		##post
	num_t1 = np.asarray(f['R8']['BMI_D10.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=58*1000*60)[0])
	num_miss = np.asarray(f['R8']['BMI_D10.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=58*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CD2.append(p_correct)
	if "/R8/percentages" in f:
		del(f['R8']['percentages'])
	f['R8'].create_dataset("percentages", data = np.asarray([R8_CD1,R8_CD2]))
	by_session.append(R8_CD1)
	by_session.append(R8_CD2)
	by_animal.append(np.asarray([R8_CD1,R8_CD2]).mean(axis = 0))

	#R11
	#CD1
	R11_CD1 = []	
		##pre:
	num_t1 = np.asarray(f['R11']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=34*1000*60)[0])
	num_t2 = np.asarray(f['R11']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=34*1000*60)[0])
	num_miss = np.asarray(f['R11']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=34*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	R11_CD1.append(p_correct)
		##peri
	num_t1 = np.asarray(f['R11']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=34*1000*60)[0])
	num_t2 = np.asarray(f['R11']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=34*1000*60)[0])
	num_miss = np.asarray(f['R11']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=34*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	R11_CD1.append(p_correct)
		##post
	num_t1 = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=40*1000*60)[0])
	num_t2 = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=40*1000*60)[0])
	num_miss = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=40*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	R11_CD1.append(p_correct)
	#CD2
	R11_CD2 = []	
		##pre:
	num_t1 = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=34*1000*60)[0])
	num_t2 = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=34*1000*60)[0])
	num_miss = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=34*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	R11_CD2.append(p_correct)
		##peri
	num_t1 = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=34*1000*60)[0])
	num_t2 = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=34*1000*60)[0])
	num_miss = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=34*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	R11_CD2.append(p_correct)
		##post
	num_t1 = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=34*1000*60)[0])
	num_t2 = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=34*1000*60)[0])
	num_miss = np.asarray(f['R11']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=34*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	R11_CD2.append(p_correct)
	if "/R11/percentages" in f:
		del(f['R11']['percentages'])
	f['R11'].create_dataset("percentages", data = np.asarray([R11_CD1,R11_CD2]))
	by_session.append(R11_CD1)
	by_session.append(R11_CD2)
	by_animal.append(np.asarray([R11_CD1,R11_CD2]).mean(axis = 0))

	#V01
	#CD1
	V01_CD1 = []	
		##pre:
	num_t1 = np.asarray(f['V01']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V01']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V01']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V01_CD1.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V01']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V01']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	num_miss = np.asarray(f['V01']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V01_CD1.append(p_correct)
		##post
	num_t1 = np.asarray(f['V01']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V01']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V01']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V01_CD1.append(p_correct)
	#CD2
	V01_CD2 = []	
		##pre:
	num_t1 = np.asarray(f['V01']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V01']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V01']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V01_CD2.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V01']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V01']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	num_miss = np.asarray(f['V01']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V01_CD2.append(p_correct)
		##post
	num_t1 = np.asarray(f['V01']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V01']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V01']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V01_CD2.append(p_correct)
	if "/V01/percentages" in f:
		del(f['V01']['percentages'])
	f['V01'].create_dataset("percentages", data = np.asarray([V01_CD1,V01_CD2]))
	by_session.append(V01_CD1)
	by_session.append(V01_CD2)
	by_animal.append(np.asarray([V01_CD1,V01_CD2]).mean(axis = 0))

	#V02
	#CD1
	V02_CD1 = []	
		##pre:
	num_t1 = np.asarray(f['V02']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V02']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V02']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V02_CD1.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V02']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V02']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	num_miss = np.asarray(f['V02']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V02_CD1.append(p_correct)
		##post
	num_t1 = np.asarray(f['V02']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V02']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V02']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V02_CD1.append(p_correct)
	#CD2
	V02_CD2 = []	
		##pre:
	num_t1 = np.asarray(f['V02']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V02']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V02']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V02_CD2.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V02']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V02']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	num_miss = np.asarray(f['V02']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V02_CD2.append(p_correct)
		##post
	num_t1 = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	#num_t2 = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V02_CD2.append(p_correct)
	if "/V02/percentages" in f:
		del(f['V02']['percentages'])
	f['V02'].create_dataset("percentages", data = np.asarray([V02_CD1,V02_CD2]))
	by_session.append(V02_CD1)
	by_session.append(V02_CD2)
	by_animal.append(np.asarray([V02_CD1,V02_CD2]).mean(axis = 0))

	#V03
	#CD1
	V03_CD1 = []	
		##pre:
	num_t1 = np.asarray(f['V03']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V03']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V03']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CD1.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V03']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V03']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	num_miss = np.asarray(f['V03']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CD1.append(p_correct)
		##post
	num_t1 = np.asarray(f['V03']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V03']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V03']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CD1.append(p_correct)
	#CD2
	V03_CD2 = []	
		##pre:
	num_t1 = np.asarray(f['V03']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V03']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V03']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CD2.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V03']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V03']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	num_miss = np.asarray(f['V03']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CD2.append(p_correct)
		##post
	num_t1 = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CD2.append(p_correct)
	if "/V03/percentages" in f:
		del(f['V03']['percentages'])
	f['V03'].create_dataset("percentages", data = np.asarray([V03_CD1,V03_CD2]))
	by_session.append(V03_CD1)
	by_session.append(V03_CD2)
	by_animal.append(np.asarray([V03_CD1,V03_CD2]).mean(axis = 0))

	#V04
	#CD1
	V04_CD1 = []	
		##pre:
	num_t1 = np.asarray(f['V04']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V04']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	#num_miss = np.asarray(f['V04']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CD1.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V04']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V04']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	#num_miss = np.asarray(f['V04']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CD1.append(p_correct)
		##post
	num_t1 = np.asarray(f['V04']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V04']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V04']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CD1.append(p_correct)
	#CD2
	V04_CD2 = []	
		##pre:
	num_t1 = np.asarray(f['V04']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V04']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V04']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CD2.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V04']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V04']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	num_miss = np.asarray(f['V04']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CD2.append(p_correct)
		##post
	num_t1 = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CD2.append(p_correct)
	if "/V04/percentages" in f:
		del(f['V04']['percentages'])
	f['V04'].create_dataset("percentages", data = np.asarray([V04_CD1,V04_CD2]))
	by_session.append(V04_CD1)
	by_session.append(V04_CD2)
	by_animal.append(np.asarray([V04_CD1,V04_CD2]).mean(axis = 0))

	#V05
	#CD1
	V05_CD1 = []	
		##pre:
	num_t1 = np.asarray(f['V05']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V05']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V05']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CD1.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V05']['BMI_D11.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V05']['BMI_D11.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	num_miss = np.asarray(f['V05']['BMI_D11.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CD1.append(p_correct)
		##post
	num_t1 = np.asarray(f['V05']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V05']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V05']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CD1.append(p_correct)
	#CD2
	V05_CD2 = []	
		##pre:
	num_t1 = np.asarray(f['V05']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V05']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V05']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CD2.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V05']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1>=30*1000*60)[0])
	num_t2 = np.asarray(f['V05']['BMI_D12.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2>=30*1000*60)[0])
	num_miss = np.asarray(f['V05']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss>=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CD2.append(p_correct)
		##post
	num_t1 = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(num_t1<=30*1000*60)[0])
	num_t2 = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(num_t2<=30*1000*60)[0])
	num_miss = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(num_miss<=30*1000*60)[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CD2.append(p_correct)
	if "/V05/percentages" in f:
		del(f['V05']['percentages'])
	f['V05'].create_dataset("percentages", data = np.asarray([V05_CD1,V05_CD2]))
	by_session.append(V05_CD1)
	by_session.append(V05_CD2)
	by_animal.append(np.asarray([V05_CD1,V05_CD2]).mean(axis = 0))

	#V11
	#CD1
	V11_CD1 = []	
		##pre:
	num_t1 = np.asarray(f['V11']['BMI_D07.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(num_t1>=15*1000*60, num_t1<=45*60*1000))[0])
	num_t2 = np.asarray(f['V11']['BMI_D07.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(num_t2>=15*1000*60, num_t2<=45*60*1000))[0])
	num_miss = np.asarray(f['V11']['BMI_D07.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(num_miss>=15*1000*60, num_miss<=45*60*1000))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V11_CD1.append(p_correct)
		##peri
	num_t1 = np.asarray(f['V11']['BMI_D08.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(num_t1>=15*1000*60, num_t1<=45*60*1000))[0])
	num_t2 = np.asarray(f['V11']['BMI_D08.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(num_t2>=15*1000*60, num_t2<=45*60*1000))[0])
	#num_miss = np.asarray(f['V11']['BMI_D08.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(num_miss>=15*1000*60, num_miss<=45*60*1000))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V11_CD1.append(p_correct)
		##post
	num_t1 = np.asarray(f['V11']['BMI_D09.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(num_t1>=15*1000*60, num_t1<=45*60*1000))[0])
	num_t2 = np.asarray(f['V11']['BMI_D09.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(num_t2>=15*1000*60, num_t2<=45*60*1000))[0])
	num_miss = np.asarray(f['V11']['BMI_D09.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(num_miss>=15*1000*60, num_miss<=45*60*1000))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V11_CD1.append(p_correct)
	if "/V11/percentages" in f:
		del(f['V11']['percentages'])
	f['V11'].create_dataset("percentages", data = np.asarray(V11_CD1))
	by_session.append(V11_CD1)
	by_animal.append(np.asarray(V11_CD1))

	if "/by_session" in f:
		del(f['by_session'])
	f.create_dataset("by_session", data = np.asarray(by_session))
	if "/by_animal" in f:
		del(f['by_animal'])
	f.create_dataset("by_animal", data = np.asarray(by_animal))

	f.close()
	print("Done!")
	return np.asarray(by_session), np.asarray(by_animal)


def plot_cd_data():
	f = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_CD_data.hdf5", 'r')
	data = np.asarray(f['by_animal'])
	f.close()
	means = np.array([data[:,0].mean(), data[:,1].mean(), data[:,2].mean()])
	sem = np.array([data[:,0].std(), data[:,1].std(), data[:,2].std()])/np.sqrt(data.shape[0])
	t_p_cd,p_val_p_cd = stats.ttest_rel(data[:,0], data[:,1])
	t_cd_r,p_val_cd_r = stats.ttest_rel(data[:,1], data[:,2])
	t_p_r,p_val_p_r = stats.ttest_rel(data[:,0], data[:,2])
	idx = np.arange(3)
	width = 1.0
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, width, color = ['k','forestgreen','k'], yerr = sem, ecolor = 'k', alpha = 0.5)
	ax.set_ylim(0,0.9)
	ax.set_xlim(-0.5, 3.5)
	ax.set_xticks(idx+0.5)
	ax.set_xticklabels(("T", "CD", "R"))
	for i in range(data.shape[0]):
		plt.plot((idx+0.5), data[i,:], alpha = 0.5, color = np.random.rand(3,), marker = 'o', linewidth = 2)
	ax.set_ylabel("Percent correct", fontsize = 14)
	ax.set_xlabel("Condition", fontsize = 14)
	ax.text(0.3, 0.81, "p = " + str(p_val_p_cd), fontsize = 12)
	ax.text(2.3, 0.81, "p = " + str(p_val_cd_r), fontsize = 12)
	ax.text(1.3, 0.85, "p = " + str(p_val_p_r), fontsize = 12)
	fig.suptitle("Performance during Contingency Degradation", fontsize = 16)
	print("pval P-CD = "+str(p_val_p_cd))
	print("tval P-CD = "+str(t_p_cd))
	print("pval CD-R = "+str(p_val_cd_r))
	print("tval CD-R = "+str(t_cd_r))
	print("pval P-R = "+str(p_val_p_r))
	print("tval P-R = "+str(t_p_r))
	print("mean P = "+str(means[0]))
	print("mean CD = "+str(means[1]))
	print("mean R = "+str(means[2]))

def get_no_feedback_data():
	f_in = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 'r')
	all_t1 = []
	all_t2 = []
	all_miss = []
	
	all_t1.append(np.asarray(f_in['V02']['BMI_D15.plx']['event_arrays']['t1']))
	all_t1.append(np.asarray(f_in['V03']['BMI_D15.plx']['event_arrays']['t1']))
	all_t1.append(np.asarray(f_in['V04']['BMI_D15.plx']['event_arrays']['t1']))
	all_t1.append(np.asarray(f_in['V05']['BMI_D15.plx']['event_arrays']['t1']))
	all_t1.append(np.asarray(f_in['V11']['BMI_D11.plx']['event_arrays']['t1']))
	all_t1.append(np.asarray(f_in['V11']['BMI_D12.plx']['event_arrays']['t1']))
	all_t1.append(np.asarray(f_in['V13']['BMI_D06.plx']['event_arrays']['t1']))

	all_t2.append(np.asarray(f_in['V02']['BMI_D15.plx']['event_arrays']['t2']))
	all_t2.append(np.asarray(f_in['V03']['BMI_D15.plx']['event_arrays']['t2']))
	all_t2.append(np.asarray(f_in['V04']['BMI_D15.plx']['event_arrays']['t2']))
	all_t2.append(np.asarray(f_in['V05']['BMI_D15.plx']['event_arrays']['t2']))
	all_t2.append(np.asarray(f_in['V11']['BMI_D11.plx']['event_arrays']['t2']))
	all_t2.append(np.asarray(f_in['V11']['BMI_D12.plx']['event_arrays']['t2']))
	all_t2.append(np.asarray(f_in['V13']['BMI_D06.plx']['event_arrays']['t2']))

	all_miss.append(np.asarray(f_in['V02']['BMI_D15.plx']['event_arrays']['miss']))
	all_miss.append(np.asarray(f_in['V03']['BMI_D15.plx']['event_arrays']['miss']))
	all_miss.append(np.asarray(f_in['V04']['BMI_D15.plx']['event_arrays']['miss']))
	all_miss.append(np.asarray(f_in['V05']['BMI_D15.plx']['event_arrays']['miss']))
	all_miss.append(np.asarray(f_in['V11']['BMI_D11.plx']['event_arrays']['miss']))
	all_miss.append(np.asarray(f_in['V11']['BMI_D12.plx']['event_arrays']['miss']))
	all_miss.append(np.asarray(f_in['V13']['BMI_D06.plx']['event_arrays']['miss']))

	f_in.close()

	duration = 0
	for i in range(len(all_t1)):
		if all_t1[i].max() > duration:
			duration = all_t1[i].max()
		if all_t2[i].max() > duration:
			duration = all_t2[i].max()
		if all_miss[i].max() > duration:
			duration = all_miss[i].max()
	duration = int(np.ceil(duration/1000.0))
	for i in range(len(all_t1)):
		all_t1[i] = ss.event_times_to_binary(all_t1[i], duration)
		all_t2[i] = ss.event_times_to_binary(all_t2[i], duration)
		all_miss[i] = ss.event_times_to_binary(all_miss[i], duration)

	all_t1 = np.asarray(all_t1)
	all_t2 = np.asarray(all_t2)
	all_miss = np.asarray(all_miss)
	
	N = duration * 1000
	num_sessions = all_t1.shape[0]
	Nwin = 1000*60*3
	Nstep = 1000*30
	winstart = np.arange(0,N-Nwin, Nstep)
	nw = winstart.shape[0]
	result = np.zeros((num_sessions, nw))
	result2 = np.zeros((num_sessions, nw))
	for session in range(num_sessions):
		t1_counts = all_t1[session,:]
		t2_counts = all_t2[session,:]
		total_counts = all_t1[session,:] + all_t2[session,:] + all_miss[session,:]
		for n in range(nw):
			idx = np.arange(winstart[n], winstart[n]+Nwin)
			t1_win = t1_counts[idx]
			t2_win = t2_counts[idx]
			total_win = total_counts[idx]
			if total_win.sum() != 0:
				result[session, n] = float(t1_win.sum())/total_win.sum()
				result2[session, n] = float(t2_win.sum())/total_win.sum()
	f_out = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_NF_data.hdf5", 'w-')
	f_out.create_dataset("p_t1", data = result[:,8:])
	f_out.create_dataset("p_t2", data = result2[:,8:])
	f_out.close()
	print("Complete!")

def plot_no_feedback_data():
	source_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_NF_data.hdf5"
	f = h5py.File(source_file, 'r')
	all_t1 = np.asarray(f['p_t1'][:,::4])
	all_t2 = np.asarray(f['p_t2'][:,::4])
	f.close()
	mean = np.nanmean(all_t1, axis = 0)
	std = np.nanstd(all_t1, axis = 0)
	sem = std/np.sqrt(all_t1.shape[0])

	mean_t2 = np.nanmean(all_t2, axis = 0)
	std_t2 = np.nanstd(all_t2, axis = 0)
	sem_t2 = std_t2/np.sqrt(all_t2.shape[0])
	
	x_axis = np.linspace(0,50, all_t2.shape[1])

	fig, ax = plt.subplots()
	ax.errorbar(x_axis, mean, yerr = sem, linewidth = 3, color = 'g', label = "Rewarded target")
	ax.errorbar(x_axis, mean_t2, yerr = sem_t2, linewidth = 3, color = 'r', label = "Unewarded target")
	#ax.set_xlim([0,10])
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Time, mins", fontsize = 16)
	ax.set_ylabel("Percent of events", fontsize = 16)
	fig.suptitle("Performance without feedback", fontsize = 18)
	ax.legend()

	t1 = all_t1.mean(axis = 1)
	t2 = all_t2.mean(axis = 1)
	means = [t1.mean(), t2.mean()]
	stds = [t1.std(), t2.std()]
	sems = stds/np.sqrt(t1.size)
	tval,pval = stats.ttest_ind(t1, t2)
	idx = np.array([0.5, 0.9])
	width = 0.3
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, width, color = ['black','silver'], yerr = sems, ecolor = 'k', alpha = 0.8)
	ax.set_ylim(0,0.8)
	ax.set_xlim(-0.01, 1.6)
	ax.set_xticks(idx+0.15)
	ax.set_xticklabels(("Rewarded", "Unrewarded"))
	ax.set_ylabel("Percent correct", fontsize = 14)
	ax.text(1, 0.75, "p = " + str(pval), fontsize = 12)
	fig.suptitle("Performance without feedback", fontsize = 16)

	fig, ax2 = plt.subplots(1)
	rew = np.vstack((t1,t2))
	xr = np.array([0,1])
	err_x = np.array([0,1])
	yerr = sems
	xerr = np.ones(2)*0.25
	for i in range(rew.shape[1]):
		ax2.plot(xr,rew[:,i],color='k',linewidth=2,marker='o',linestyle='none')
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),['Rewarded','Unrewarded'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.3,1.3)

	print("pval = "+str(pval))
	print("tval = "+str(tval))
	print("mean rew = "+str(means[0]))
	print("mean unrew = "+str(means[1]))

def get_cr_data():
	source_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5"
	destination_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_CR_data.hdf5"
	f = h5py.File(source_file, 'r')
	f_out = h5py.File(destination_file, 'a')
	animals_list = ["R7", "R8", "V01", "V02", "V03", "V04", "V05"]
	for animal in animals_list:
		if "/"+animal in f_out:
			pass
		else:
			f_out.create_group(animal)
	chunks_by_session = []
	chunks_by_animal = []
	t1_arrays_by_session = []
	total_arrays_by_session = []
	t1_arrays_by_animal = []
	total_arrays_by_animal = []

	#R7:
	R7_CR_t1_arrays = []
	R7_CR_total_arrays = []
	##CR1:
	R7_CR1 = []	
		##pre:
	t1 = np.asarray(f['R7']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=8*1000*60, t1<=25*1000*60))[0])
	miss = np.asarray(f['R7']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=8*1000*60, miss<=25*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max()))/1000.0))
	R7_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	R7_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
		##peri
	t1 = np.asarray(f['R7']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=26*1000*60, t1<=36*1000*60))[0])
	miss = np.asarray(f['R7']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=26*1000*60, miss<=36*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['R7']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=40*1000*60, t1<=60*1000*60))[0])
	miss = np.asarray(f['R7']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=40*1000*60, miss<=60*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CR1.append(p_correct)
	#CD2
	R7_CR2 = []	
		##pre:
	t1 = np.asarray(f['R7']['BMI_D15.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=60*1000*60, num_t1<=85*1000*60))[0])
	miss = np.asarray(f['R7']['BMI_D15.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=60*1000*60, num_miss<=85*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CR2.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max()))/1000.0))
	R7_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	R7_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))

		##peri
	t1 = np.asarray(f['R7']['BMI_D15.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=90*1000*60, num_t1<=107*1000*60))[0])
	miss = np.asarray(f['R7']['BMI_D15.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=90*1000*60, num_miss<=107*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CR2.append(p_correct)
		##post
	t1 = np.asarray(f['R7']['BMI_D15.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=110*1000*60, num_t1<=133*1000*60))[0])
	miss = np.asarray(f['R7']['BMI_D15.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=110*1000*60, num_miss<=133*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R7_CR2.append(p_correct)
	if "/R7/percentages" in f_out:
		del(f_out['R7']['percentages'])
	f_out['R7'].create_dataset("percentages", data = np.asarray([R7_CR1,R7_CR2]))
	chunks_by_session.append(R7_CR1)
	chunks_by_session.append(R7_CR2)
	chunks_by_animal.append(np.asarray([R7_CR1,R7_CR2]).mean(axis = 0))
	t1_arrays_by_animal.append(R7_CR_t1_arrays)
	total_arrays_by_animal.append(R7_CR_total_arrays)

	#R8:
	R8_CR_t1_arrays = []
	R8_CR_total_arrays = []
	##CR1:
	R8_CR1 = []	
		##pre:
	t1 = np.asarray(f['R8']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=17*1000*60, t1<=37*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=17*1000*60, miss<=37*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max()))/1000.0))
	R8_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	R8_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
		##peri
	t1 = np.asarray(f['R8']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=38*1000*60, t1<=50*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=38*1000*60, miss<=50*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['R8']['BMI_D12.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=55*1000*60, t1<=72*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D12.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=55*1000*60, miss<=72*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR1.append(p_correct)
	#CD2
	R8_CR2 = []	
		##pre:
	t1 = np.asarray(f['R8']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=5*1000*60, num_t1<=20*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=5*1000*60, num_miss<=20*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR2.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max()))/1000.0))
	R8_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	R8_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))

		##peri
	t1 = np.asarray(f['R8']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=27*1000*60, num_t1<=38*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=27*1000*60, num_miss<=38*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR2.append(p_correct)
		##post
	t1 = np.asarray(f['R8']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=57*1000*60, num_t1<=72*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=57*1000*60, num_miss<=72*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR2.append(p_correct)

		#CD2
	R8_CR3 = []	
		##pre:
	t1 = np.asarray(f['R8']['BMI_D15.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=16*1000*60, num_t1<=44*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D15.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=16*1000*60, num_miss<=44*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR3.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max()))/1000.0))
	R8_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	R8_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))

		##peri
	t1 = np.asarray(f['R8']['BMI_D15.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=57*1000*60, num_t1<=66*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D15.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=57*1000*60, num_miss<=66*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR3.append(p_correct)
		##post
	t1 = np.asarray(f['R8']['BMI_D15.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=68*1000*60, num_t1<=85*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D15.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=68*1000*60, num_miss<=85*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR3.append(p_correct)

		#CD2
	R8_CR4 = []	
		##pre:
	t1 = np.asarray(f['R8']['BMI_D16.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=6*1000*60, num_t1<=36*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D16.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=6*1000*60, num_miss<=36*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR4.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max()))/1000.0))
	R8_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	R8_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))

		##peri
	t1 = np.asarray(f['R8']['BMI_D16.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=50*1000*60, num_t1<=60*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D16.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=50*1000*60, num_miss<=60*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR4.append(p_correct)
		##post
	t1 = np.asarray(f['R8']['BMI_D16.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=62*1000*60, num_t1<=72*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D16.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=62*1000*60, num_miss<=72*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR4.append(p_correct)
	if "/R8/percentages" in f_out:
		del(f_out['R8']['percentages'])
	f_out['R8'].create_dataset("percentages", data = np.asarray([R8_CR1,R8_CR2,R8_CR3,R8_CR4]))
	chunks_by_session.append(R8_CR1)
	chunks_by_session.append(R8_CR2)
	chunks_by_session.append(R8_CR3)
	chunks_by_session.append(R8_CR4)
	chunks_by_animal.append(np.asarray([R8_CR1,R8_CR2,R8_CR3,R8_CR4]).mean(axis = 0))
	t1_arrays_by_animal.append(R8_CR_t1_arrays)
	total_arrays_by_animal.append(R8_CR_total_arrays)

	#V01:
	V01_CR_t1_arrays = []
	V01_CR_total_arrays = []
	##CR1:
	V01_CR1 = []	
		##pre:
	t1 = np.asarray(f['V01']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=11*1000*60, t1<=20*1000*60))[0])
	t2 = np.asarray(f['V01']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=11*1000*60, t2<=20*1000*60))[0])
	miss = np.asarray(f['V01']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=11*1000*60, miss<=20*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V01_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max(), t2.max()))/1000.0))
	V01_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	V01_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
		##peri
	t1 = np.asarray(f['V01']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=20*1000*60, t1<=24*1000*60))[0])
	t2 = np.asarray(f['V01']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=20*1000*60, t2<=24*1000*60))[0])
	miss = np.asarray(f['V01']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=20*1000*60, miss<=24*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V01_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['V01']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=25*1000*60, t1<=35*1000*60))[0])
	t2 = np.asarray(f['V01']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=25*1000*60, t2<=35*1000*60))[0])
	miss = np.asarray(f['V01']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=25*1000*60, miss<=35*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V01_CR1.append(p_correct)
	if "/V01/percentages" in f_out:
		del(f_out['V01']['percentages'])
	f_out['V01'].create_dataset("percentages", data = np.asarray([V01_CR1]))
	chunks_by_session.append(V01_CR1)
	chunks_by_animal.append(np.asarray([V01_CR1]).mean(axis = 0))
	t1_arrays_by_animal.append(V01_CR_t1_arrays)
	total_arrays_by_animal.append(V01_CR_total_arrays)

		#V02:
	V02_CR_t1_arrays = []
	V02_CR_total_arrays = []
	##CR1:
	V02_CR1 = []	
		##pre:
	t1 = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=15*1000*60, t1<=25*1000*60))[0])
	#t2 = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['t2'])
	#num_t2 = len(np.where(t2<=30*1000*60)[0])	
	miss = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=15*1000*60, miss<=25*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	V02_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max()))/1000.0))
	V02_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	V02_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
		##peri
	t1 = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=28*1000*60, t1<=35*1000*60))[0])
	#t2 = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['t2'])
	#num_t2 = len(np.where(np.logical_and(t2>=30*1000*60, t2<=40*1000*60))[0])
	miss = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=28*1000*60, miss<=35*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	V02_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=37*1000*60, t1<=48*1000*60))[0])
	#t2 = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['t2'])
	#num_t2 = len(np.where(t2>=45*1000*60)[0])
	miss = np.asarray(f['V02']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=37*1000*60, miss<=48*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	V02_CR1.append(p_correct)
	if "/V02/percentages" in f_out:
		del(f_out['V02']['percentages'])
	f_out['V02'].create_dataset("percentages", data = np.asarray([V02_CR1]))
	chunks_by_session.append(V02_CR1)
	chunks_by_animal.append(np.asarray([V02_CR1]).mean(axis = 0))
	t1_arrays_by_animal.append(V02_CR_t1_arrays)
	total_arrays_by_animal.append(V02_CR_total_arrays)

	#V03:
	V03_CR_t1_arrays = []
	V03_CR_total_arrays = []
	##CR1:
	V03_CR1 = []	
		##pre:
	t1 = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=11*1000*60, t1<=27*1000*60))[0])
	t2 = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=11*1000*60, t2<=27*1000*60))[0])	
	miss = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=11*1000*60, miss<=27*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max(), t2.max()))/1000.0))
	V03_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	V03_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
		##peri
	t1 = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=30*1000*60, t1<=36*1000*60))[0])
	t2 = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=30*1000*60, t2<=36*1000*60))[0])
	miss = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=30*1000*60, miss<=36*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=42*1000*60, t1<=55*1000*60))[0])
	t2 = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=42*1000*60, t2<=55*1000*60))[0])
	miss = np.asarray(f['V03']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=42*1000*60, miss<=55*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CR1.append(p_correct)
	if "/V03/percentages" in f_out:
		del(f_out['V03']['percentages'])
	f_out['V03'].create_dataset("percentages", data = np.asarray([V03_CR1]))
	chunks_by_session.append(V03_CR1)
	chunks_by_animal.append(np.asarray([V03_CR1]).mean(axis = 0))
	t1_arrays_by_animal.append(V03_CR_t1_arrays)
	total_arrays_by_animal.append(V03_CR_total_arrays)

	#V04:
	V04_CR_t1_arrays = []
	V04_CR_total_arrays = []
	##CR1:
	V04_CR1 = []	
		##pre:
	t1 = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=6*1000*60, t1<=17*1000*60))[0])
	t2 = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=6*1000*60, t2<=17*1000*60))[0])
	miss = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=6*1000*60, miss<=17*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max(), t2.max()))/1000.0))
	V04_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	V04_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
		##peri
	t1 = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=25*1000*60, t1<=32*1000*60))[0])
	t2 = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=25*1000*60, t2<=32*1000*60))[0])
	miss = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=25*1000*60, miss<=32*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=42*1000*60, t1<=58*1000*60))[0])
	t2 = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['t1'])
	num_t2 = len(np.where(np.logical_and(t2>=42*1000*60, t2<=58*1000*60))[0])
	miss = np.asarray(f['V04']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=42*1000*60, miss<=58*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CR1.append(p_correct)
	if "/V04/percentages" in f_out:
		del(f_out['V04']['percentages'])
	f_out['V04'].create_dataset("percentages", data = np.asarray([V04_CR1]))
	chunks_by_session.append(V04_CR1)
	chunks_by_animal.append(np.asarray([V04_CR1]).mean(axis = 0))
	t1_arrays_by_animal.append(V04_CR_t1_arrays)
	total_arrays_by_animal.append(V04_CR_total_arrays)

	#V05:
	V05_CR_t1_arrays = []
	V05_CR_total_arrays = []
	##CR1:
	V05_CR1 = []	
		##pre:
	t1 = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=15*1000*60, t1<=27*1000*60))[0])
	t2 = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=15*1000*60, t2<=27*1000*60))[0])
	miss = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=15*1000*60, miss<=27*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max(), t2.max()))/1000.0))
	V05_CR_t1_arrays.append(ss.event_times_to_binary(t1, duration))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	V05_CR_total_arrays.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
		##peri
	t1 = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=29*1000*60, t1<=41*1000*60))[0])
	t2 = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=29*1000*60, t2<=41*1000*60))[0])
	miss = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=29*1000*60, miss<=41*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=52*1000*60, t1<=65*1000*60))[0])
	t2 = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=52*1000*60, t2<=65*1000*60))[0])
	miss = np.asarray(f['V05']['BMI_D13.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=52*1000*60, miss<=65*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CR1.append(p_correct)
	if "/V05/percentages" in f_out:
		del(f_out['V05']['percentages'])
	f_out['V05'].create_dataset("percentages", data = np.asarray([V05_CR1]))
	chunks_by_session.append(V05_CR1)
	chunks_by_animal.append(np.asarray([V05_CR1]).mean(axis = 0))
	t1_arrays_by_animal.append(V05_CR_t1_arrays)
	total_arrays_by_animal.append(V05_CR_total_arrays)

	f.close()

	if "/chunks_by_session" in f_out:
		del(f_out['chunks_by_session'])
	f_out.create_dataset("chunks_by_session", data = np.asarray(chunks_by_session))
	if "/chunks_by_animal" in f_out:
		del(f_out['chunks_by_animal'])
	f_out.create_dataset("chunks_by_animal", data = np.asarray(chunks_by_animal))

	duration = 0
	for i in range(len(t1_arrays_by_session)):
		if t1_arrays_by_session[i].size > duration:
			duration = t1_arrays_by_session[i].size

	for i in range(len(t1_arrays_by_session)):
		if t1_arrays_by_session[i].size < duration:
			add = np.empty((duration-t1_arrays_by_session[i].size))
			add[:] = np.nan
			t1_arrays_by_session[i] = np.hstack((t1_arrays_by_session[i], add))
			total_arrays_by_session[i] = np.hstack((total_arrays_by_session[i], add))

	t1_arrays_by_session = np.asarray(t1_arrays_by_session)
	total_arrays_by_session = np.asarray(total_arrays_by_session)
	if "/t1_arrays_by_session" in f_out:
		del(f_out['t1_arrays_by_session'])
	f_out.create_dataset('t1_arrays_by_session', data = t1_arrays_by_session)

	if "/total_arrays_by_session" in f_out:
		del(f_out['total_arrays_by_session'])
	f_out.create_dataset('total_arrays_by_session', data = total_arrays_by_session)

	for a in range(len(t1_arrays_by_animal)):
		for i in range(len(t1_arrays_by_animal[a])):
			if t1_arrays_by_animal[a][i].size < duration:
				add = np.empty((duration-t1_arrays_by_animal[a][i].size))
				add[:] = np.nan
				t1_arrays_by_animal[a][i] = np.hstack((t1_arrays_by_animal[a][i], add))
				total_arrays_by_animal[a][i] = np.hstack((total_arrays_by_animal[a][i], add))
		t1_arrays_by_animal[a] = np.asarray(t1_arrays_by_animal[a]).mean(axis = 0)
		total_arrays_by_animal[a] = np.asarray(total_arrays_by_animal[a]).mean(axis = 0)

	t1_arrays_by_animal = np.asarray(t1_arrays_by_animal)
	total_arrays_by_animal = np.asarray(total_arrays_by_animal)
	if "/t1_arrays_by_animal" in f_out:
		del(f_out['t1_arrays_by_animal'])
	f_out.create_dataset('t1_arrays_by_animal', data = t1_arrays_by_animal)

	if "/total_arrays_by_animal" in f_out:
		del(f_out['total_arrays_by_animal'])
	f_out.create_dataset('total_arrays_by_animal', data = total_arrays_by_animal)

	N = duration
	num_sessions = t1_arrays_by_animal.shape[0]
	Nwin = 1000*60*3
	Nstep = 1000*30
	winstart = np.arange(0,N-Nwin, Nstep)
	nw = winstart.shape[0]
	result = np.zeros((num_sessions, nw))
	for session in range(num_sessions):
		t1_counts = t1_arrays_by_animal[session,:]
		total_counts = total_arrays_by_animal[session,:]
		for n in range(nw):
			idx = np.arange(winstart[n], winstart[n]+Nwin)
			t1_win = t1_counts[idx]
			total_win = total_counts[idx]
			if total_win.sum() != 0:
				result[session, n] = float(t1_win.sum())/total_win.sum()
	if "/p_t1_by_animal" in f_out:
		del(f_out['p_t1_by_animal'])
	f_out.create_dataset('p_t1_by_animal', data = result)

	N = duration
	num_sessions = t1_arrays_by_session.shape[0]
	Nwin = 1000*60*3
	Nstep = 1000*30
	winstart = np.arange(0,N-Nwin, Nstep)
	nw = winstart.shape[0]
	result = np.zeros((num_sessions, nw))
	for session in range(num_sessions):
		t1_counts = t1_arrays_by_session[session,:]
		total_counts = total_arrays_by_session[session,:]
		for n in range(nw):
			idx = np.arange(winstart[n], winstart[n]+Nwin)
			t1_win = t1_counts[idx]
			total_win = total_counts[idx]
			if total_win.sum() != 0:
				result[session, n] = float(t1_win.sum())/total_win.sum()
	if "/p_t1_by_session" in f_out:
		del(f_out['p_t1_by_session'])
	f_out.create_dataset('p_t1_by_session', data = result)

	f_out.close()
	print("Done!")

def get_light_data():
	source_file = r"C:\Users\Ryan\Documents\data\R7_thru_V13_all_data.hdf5"
	destination_file = r"C:\Users\Ryan\Documents\data\R7_thru_V13_light_data.hdf5"
	f = h5py.File(source_file, 'r')
	f_out = h5py.File(destination_file, 'a')
	animals_list = ["R8", "V02", "V03", "V04", "V05"]
	for animal in animals_list:
		if "/"+animal in f_out:
			pass
		else:
			f_out.create_group(animal)
	chunks_by_session = []
	t1_arrays_by_session = []
	total_arrays_by_session = []

	#R8:
	##CR1:
	R8_CR1 = []	
		##pre:
	t1 = np.asarray(f['R8']['BMI_D17.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=6*1000*60, t1<=31*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D17.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=6*1000*60, miss<=31*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max()))/1000.0))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration))
		##peri
	t1 = np.asarray(f['R8']['BMI_D17.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=35*1000*60, t1<=40*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D17.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=35*1000*60, miss<=40*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['R8']['BMI_D17.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=40*1000*60, t1<=55*1000*60))[0])
	miss = np.asarray(f['R8']['BMI_D17.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=55*1000*60, miss<=40*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss)
	R8_CR1.append(p_correct)
	if "/R8/percentages" in f_out:
		del(f_out['R8']['percentages'])
	f_out['R8'].create_dataset("percentages", data = np.asarray([R8_CR1]))
	chunks_by_session.append(R8_CR1)


		#V02:
	V02_CR_t1_arrays = []
	V02_CR_total_arrays = []
	##CR1:
	V02_CR1 = []	
		##pre:
	t1 = np.asarray(f['V02']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=17*1000*60, t1<=26*1000*60))[0])
	t2 = np.asarray(f['V02']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=17*1000*60, t2<=26*1000*60))[0])
	miss = np.asarray(f['V02']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=17*1000*60, miss<=26*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V02_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max(), t2.max()))/1000.0))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
		##peri
	t1 = np.asarray(f['V02']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=31*1000*60, t1<=50*1000*60))[0])
	t2 = np.asarray(f['V02']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=31*1000*60, t2<=50*1000*60))[0])
	miss = np.asarray(f['V02']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=31*1000*60, miss<=50*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V02_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['V02']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=54*1000*60, t1<=67*1000*60))[0])
	t2 = np.asarray(f['V02']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=54*1000*60, t2<=67*1000*60))[0])
	miss = np.asarray(f['V02']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=54*1000*60, miss<=67*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V02_CR1.append(p_correct)
	if "/V02/percentages" in f_out:
		del(f_out['V02']['percentages'])
	f_out['V02'].create_dataset("percentages", data = np.asarray([V02_CR1]))
	chunks_by_session.append(V02_CR1)

	#V03:
	V03_CR_t1_arrays = []
	V03_CR_total_arrays = []
	##CR1:
	V03_CR1 = []	
		##pre:
	t1 = np.asarray(f['V03']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=16*1000*60, t1<=28*1000*60))[0])
	t2 = np.asarray(f['V03']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=16*1000*60, t2<=28*1000*60))[0])
	miss = np.asarray(f['V03']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=16*1000*60, miss<=28*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max(), t2.max()))/1000.0))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
		##peri
	t1 = np.asarray(f['V03']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=30*1000*60, t1<=37*1000*60))[0])
	t2 = np.asarray(f['V03']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=30*1000*60, t2<=37*1000*60))[0])
	miss = np.asarray(f['V03']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=30*1000*60, miss<=37*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['V03']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=42*1000*60, t1<=62*1000*60))[0])
	t2 = np.asarray(f['V03']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=42*1000*60, t2<=62*1000*60))[0])
	miss = np.asarray(f['V03']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=42*1000*60, miss<=62*1000*60))[0])

	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V03_CR1.append(p_correct)
	if "/V03/percentages" in f_out:
		del(f_out['V03']['percentages'])
	f_out['V03'].create_dataset("percentages", data = np.asarray([V03_CR1]))
	chunks_by_session.append(V03_CR1)

	#V04:
	V04_CR_t1_arrays = []
	V04_CR_total_arrays = []
	##CR1:
	V04_CR1 = []	
		##pre:
	t1 = np.asarray(f['V04']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=20*1000*60, t1<=34*1000*60))[0])
	t2 = np.asarray(f['V04']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=20*1000*60, t2<=34*1000*60))[0])	
	miss = np.asarray(f['V04']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=20*1000*60, miss<=34*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max(), t2.max()))/1000.0))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
		##peri
	t1 = np.asarray(f['V04']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=36*1000*60, t1<=39*1000*60))[0])
	t2 = np.asarray(f['V04']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=36*1000*60, t2<=39*1000*60))[0])
	miss = np.asarray(f['V04']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=36*1000*60, miss<=39*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['V04']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=40*1000*60, t1<=49*1000*60))[0])
	t2 = np.asarray(f['V04']['BMI_D14.plx']['event_arrays']['t1'])
	num_t2 = len(np.where(np.logical_and(t2>=40*1000*60, t2<=49*1000*60))[0])	
	miss = np.asarray(f['V04']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=40*1000*60, miss<=49*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V04_CR1.append(p_correct)
	if "/V04/percentages" in f_out:
		del(f_out['V04']['percentages'])
	f_out['V04'].create_dataset("percentages", data = np.asarray([V04_CR1]))
	chunks_by_session.append(V04_CR1)

	#V05:
	V05_CR_t1_arrays = []
	V05_CR_total_arrays = []
	##CR1:
	V05_CR1 = []	
		##pre:
	t1 = np.asarray(f['V05']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=4*1000*60, t1<=9*1000*60))[0])
	t2 = np.asarray(f['V05']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=4*1000*60, t2<=9*1000*60))[0])	
	miss = np.asarray(f['V05']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=4*1000*60, miss<=9*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CR1.append(p_correct)
	duration = int(np.ceil(max((t1.max(), miss.max(), t2.max()))/1000.0))
	t1_arrays_by_session.append(ss.event_times_to_binary(t1, duration))
	total_arrays_by_session.append(ss.event_times_to_binary(t1, duration)+ss.event_times_to_binary(miss, duration)+ss.event_times_to_binary(t2, duration))
		##peri
	t1 = np.asarray(f['V05']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=10*1000*60, t1<=23*1000*60))[0])
	t2 = np.asarray(f['V05']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=10*1000*60, t2<=23*1000*60))[0])
	miss = np.asarray(f['V05']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=10*1000*60, miss<=23*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CR1.append(p_correct)
		##post
	t1 = np.asarray(f['V05']['BMI_D14.plx']['event_arrays']['t1'])
	num_t1 = len(np.where(np.logical_and(t1>=30*1000*60, t1<=55*1000*60))[0])
	t2 = np.asarray(f['V05']['BMI_D14.plx']['event_arrays']['t2'])
	num_t2 = len(np.where(np.logical_and(t2>=30*1000*60, t2<=55*1000*60))[0])	
	miss = np.asarray(f['V05']['BMI_D14.plx']['event_arrays']['miss'])
	num_miss = len(np.where(np.logical_and(miss>=30*1000*60, miss<=55*1000*60))[0])
	p_correct = float(num_t1)/(num_t1+num_miss+num_t2)
	V05_CR1.append(p_correct)
	if "/V05/percentages" in f_out:
		del(f_out['V05']['percentages'])
	f_out['V05'].create_dataset("percentages", data = np.asarray([V05_CR1]))
	chunks_by_session.append(V05_CR1)

	f.close()

	if "/chunks_by_session" in f_out:
		del(f_out['chunks_by_session'])
	f_out.create_dataset("chunks_by_session", data = np.asarray(chunks_by_session))

	duration = 0
	for i in range(len(t1_arrays_by_session)):
		if t1_arrays_by_session[i].size > duration:
			duration = t1_arrays_by_session[i].size

	for i in range(len(t1_arrays_by_session)):
		if t1_arrays_by_session[i].size < duration:
			add = np.empty((duration-t1_arrays_by_session[i].size))
			add[:] = np.nan
			t1_arrays_by_session[i] = np.hstack((t1_arrays_by_session[i], add))
			total_arrays_by_session[i] = np.hstack((total_arrays_by_session[i], add))

	t1_arrays_by_session = np.asarray(t1_arrays_by_session)
	total_arrays_by_session = np.asarray(total_arrays_by_session)
	if "/t1_arrays_by_session" in f_out:
		del(f_out['t1_arrays_by_session'])
	f_out.create_dataset('t1_arrays_by_session', data = t1_arrays_by_session)

	if "/total_arrays_by_session" in f_out:
		del(f_out['total_arrays_by_session'])
	f_out.create_dataset('total_arrays_by_session', data = total_arrays_by_session)

	N = duration
	num_sessions = t1_arrays_by_session.shape[0]
	Nwin = 1000*60*3
	Nstep = 1000*30
	winstart = np.arange(0,N-Nwin, Nstep)
	nw = winstart.shape[0]
	result = np.zeros((num_sessions, nw))
	for session in range(num_sessions):
		t1_counts = t1_arrays_by_session[session,:]
		total_counts = total_arrays_by_session[session,:]
		for n in range(nw):
			idx = np.arange(winstart[n], winstart[n]+Nwin)
			t1_win = t1_counts[idx]
			total_win = total_counts[idx]
			if total_win.sum() != 0:
				result[session, n] = float(t1_win.sum())/total_win.sum()
	if "/p_t1_by_session" in f_out:
		del(f_out['p_t1_by_session'])
	f_out.create_dataset('p_t1_by_session', data = result)

	f_out.close()
	print("Done!")
	
def plot_cr_data():
	f = h5py.File(r"Z:\Data\processed_data\V1_BMI_final\raw_data\R7_thru_V13_CR_data.hdf5", 'r')
	data = np.asarray(f['chunks_by_animal'])

	means = np.array([data[:,0].mean(), data[:,1].mean(), data[:,2].mean()])
	sems = np.array([data[:,0].std(), data[:,1].mean(), data[:,2].std()])/np.sqrt(data.shape[0])
	t_val,p_val = stats.ttest_rel(data[:,0], data[:,2])
	t_val_cr,p_val_cr = stats.ttest_rel(data[:,0], data[:,1])
	t_val_r,p_val_r = stats.ttest_rel(data[:,1], data[:,2])
	idx = np.array([0, 1, 2])-0.4
	width = 0.8
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, width, color = ['lightblue','orange', 'royalblue'], yerr = sems, ecolor = 'k', alpha = 0.8)
	ax.set_ylim(0,1)
	ax.set_xlim(-0.01, 1.6)
	ax.set_xticks(idx+0.15)
	ax.set_xticklabels(("Pre-reversal", "Reversal", "Recovery"))
	ax.set_ylabel("Percent correct", fontsize = 14)
	ax.set_xlabel("Condition", fontsize = 14)
	ax.text(0.2, 0.95, "p = " + str(p_val), fontsize = 12)
	ax.text(0.7, 0.9, "p = " + str(p_val_cr), fontsize = 12)
	ax.text(1, 0.95, "p = " + str(p_val_r), fontsize = 12)
	fig.suptitle("Performance during Contingency Reversal", fontsize = 16)

	all_t1 = np.asarray(f['p_t1_by_session'][7:,0:125:5])
	f.close()
	mean = np.nanmean(all_t1, axis = 0)
	std = np.nanstd(all_t1, axis = 0)
	sem = std/np.sqrt(all_t1.shape[0])
	
	x_axis = np.linspace(0,75, all_t1.shape[1])

	fig, ax = plt.subplots()
	ax.plot(x_axis, mean, linewidth = 2, color = 'r', label = "Rewarded target")
	ax.fill_between(x_axis, mean-sem, mean+sem, 
		alpha = 0.5, facecolor = 'r')
	#ax.set_xlim([0,10])
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)

	plt.vlines(28, 0, 1, linestyle = 'dashed')	
	ax.set_xlabel("Time, mins", fontsize = 16)
	ax.set_ylabel("Percent of events", fontsize = 16)
	fig.suptitle("Performance during CR", fontsize = 18)
	ax.legend()

	fig, ax2 = plt.subplots(1)
	rew = data.T
	xr = np.array([0,1,2])
	err_x = np.array([0,1,2])
	yerr = sems
	xerr = np.ones(3)*0.25
	bars = ax2.bar(idx, means, width, color = ['b','r', 'b'])
	for i in range(rew.shape[1]):
		ax2.plot(xr,rew[:,i],color='k',linewidth=2,marker='o')
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,3),['P','Rev','R'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.6,2.6)
	ax2.set_ylim(0,1.3)

	print("pval P-Rev = "+str(p_val_cr))
	print("tval P-Rev = "+str(t_val_cr))
	print("pval Rev-R = "+str(p_val_r))
	print("tval Rev-R = "+str(t_val_r))
	print("pval P-R = "+str(p_val))
	print("tval P-R = "+str(t_val))
	print("mean P = "+str(means[0]))
	print("mean Rev = "+str(means[1]))
	print("mean R = "+str(means[2]))


def plot_light_data():
	f = h5py.File("/Volumes/Untitled/Ryan/V1_BMI/Neuron/R7_thru_V13_light_data.hdf5", 'r')
	data = np.asarray(f['chunks_by_session'])

	means = np.array([data[:,0].mean(), data[:,1].mean()])
	sems = np.array([data[:,0].std(), data[:,1].std()])/np.sqrt(data.shape[0])
	t_val,p_val = stats.ttest_rel(data[:,0], data[:,1])
	idx = np.array([0.5, 0.9])
	width = 0.3
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, width, color = ['black','yellow'], yerr = sems, ecolor = 'k', alpha = 0.8)
	ax.set_ylim(0,1)
	ax.set_xlim(-0.01, 1.6)
	ax.set_xticks(idx+0.15)
	ax.set_xticklabels(("Dark", "Light"))
	ax.set_ylabel("Percent correct", fontsize = 14)
	ax.set_xlabel("Condition", fontsize = 14)
	ax.text(0.1, 0.95, "p = " + str(p_val), fontsize = 12)
	fig.suptitle("Performance during Light Change", fontsize = 16)

	all_t1 = np.asarray(f['p_t1_by_session'][:,0:125])
	f.close()
	mean = np.nanmean(all_t1, axis = 0)
	std = np.nanstd(all_t1, axis = 0)
	sem = std/np.sqrt(all_t1.shape[0])
	
	x_axis = np.linspace(0,75, all_t1.shape[1])

	fig, ax = plt.subplots()
	ax.errorbar(x_axis, mean, yerr = sem, linewidth = 3, color = 'g', label = "Rewarded target")
	ax.set_xlim([0,60])
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.axvspan(0, 33, alpha = 0.5, color = 'k')
	ax.axvspan(33, 60, alpha = 0.5, color = 'y')
	ax.set_xlabel("Time, mins", fontsize = 16)
	ax.set_ylabel("Percent of events", fontsize = 16)
	fig.suptitle("Performance during Light Change", fontsize = 18)
	ax.legend()

	fig, ax2 = plt.subplots(1)
	rew = data[:,0:2].T
	xr = np.array([0,1])
	err_x = np.array([0,1])
	yerr = sems
	xerr = np.ones(2)*0.25
	for i in range(rew.shape[1]):
		ax2.plot(xr,rew[:,i],color='k',linewidth=2,marker='o')
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),['Dark','Light'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.3,1.3)

	print("pval = "+str(p_val))
	print("tval = "+str(t_val))
	print("mean early = "+str(means[0]))
	print("mean light = "+str(means[1]))

##specifically retreives data for LATE in sessions
def get_triggered_spike_rates():
	try:
		ds.save_multi_group_triggered_data(r"D:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 
			r"D:\Ryan\processed_data\V1_BMI_final\raw_data\e1_t1_spikes_late.hdf5", "t1", ["e1_units", "spikes"], [6000,6000],
			chunk = [0,10])
	except IOError:
		print("File exists; skipping")
	try:
		ds.save_multi_group_triggered_data(r"D:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 
			r"D:\Ryan\processed_data\V1_BMI_final\raw_data\e2_t1_spikes_late.hdf5", "t1", ["e2_units", "spikes"], [6000,6000],
			chunk = [0,10])
	except IOError:
		print("File exists; skipping")
	try:
		ds.save_multi_group_triggered_data(r"D:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 
			r"D:\Ryan\processed_data\V1_BMI_final\raw_data\V1_t1_spikes_late.hdf5", "t1", ["V1_units", "spikes"], [6000,6000],
			chunk = [0,10])
	except IOError:
		print("File exists; skipping")
	# try:
	# 	ds.save_multi_group_triggered_data(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 
	# 		r"J:\Ryan\processed_data\V1_BMI_final\raw_data\e1_t2_spikes_late.hdf5", "t2", ["e1_units", "spikes"], [6000,6000],
	# 		chunk = [0,10])
	# except IOError:
	# 	print "File exists; skipping"
	# try:	
	# 	ds.save_multi_group_triggered_data(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 
	# 		r"J:\Ryan\processed_data\V1_BMI_final\raw_data\e2_t2_spikes_late.hdf5", "t2", ["e2_units", "spikes"], [6000,6000],
	# 		chunk = [0,10])
	# except IOError:
	# 	print "File exists; skipping"
	# try:	
	# 	ds.save_multi_group_triggered_data(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 
	# 		r"J:\Ryan\processed_data\V1_BMI_final\raw_data\V1_t2_spikes_late.hdf5", "t2", ["V1_units", "spikes"], [6000,6000],
	# 		chunk = [0,10])
	# except IOError:
	# 	print "File exists; skipping"

	f_out = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_triggered_data_late.hdf5", 'a')
	
	f_in = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\e1_t1_spikes_late.hdf5", 'r')
	if "/e1_t1_spikes" in f_out:
		del(f_out['e1_t1_spikes'])
	f_out.create_dataset("e1_t1_spikes", data = np.asarray(f_in['e1_units_t1_spikes']))
	f_in.close()

	f_in = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\e2_t1_spikes_late.hdf5", 'r')
	if "/e2_t1_spikes" in f_out:
		del(f_out['e2_t1_spikes'])
	f_out.create_dataset("e2_t1_spikes", data = np.asarray(f_in['e2_units_t1_spikes']))
	f_in.close()

	f_in = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\V1_t1_spikes_late.hdf5", 'r')
	if "/V1_t1_spikes" in f_out:
		del(f_out['V1_t1_spikes'])
	f_out.create_dataset("V1_t1_spikes", data = np.asarray(f_in['V1_units_t1_spikes']))
	f_in.close()

	# f_in = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\e1_t2_spikes_late.hdf5", 'r')
	# if "/e1_t2_spikes" in f_out:
	# 	del(f_out['e1_t2_spikes'])
	# f_out.create_dataset("e1_t2_spikes", data = np.asarray(f_in['e1_units_t2_spikes']))
	# f_in.close()

	# f_in = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\e2_t2_spikes_late.hdf5", 'r')
	# if "/e2_t2_spikes" in f_out:
	# 	del(f_out['e2_t2_spikes'])
	# f_out.create_dataset("e2_t2_spikes", data = np.asarray(f_in['e2_units_t2_spikes']))
	# f_in.close()

	# f_in = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\V1_t2_spikes_late.hdf5", 'r')
	# if "/V1_t2_spikes" in f_out:
	# 	del(f_out['V1_t2_spikes'])
	# f_out.create_dataset("V1_t2_spikes", data = np.asarray(f_in['V1_units_t2_spikes']))
	#f_in.close()

	e1_t1 = ss.windowRate(np.asarray(f_out['e1_t1_spikes']), [100, 50]).mean(axis = 1)
	e2_t1 = ss.windowRate(np.asarray(f_out['e2_t1_spikes']), [100, 50]).mean(axis = 1)
	V1_t1 = ss.windowRate(np.asarray(f_out['V1_t1_spikes']), [100, 50]).mean(axis = 1)
	# e1_t2 = ss.windowRate(np.asarray(f_out['e1_t2_spikes']), [100, 50]).mean(axis = 1)
	# e2_t2 = ss.windowRate(np.asarray(f_out['e2_t2_spikes']), [100, 50]).mean(axis = 1)
	# V1_t2 = ss.windowRate(np.asarray(f_out['V1_t2_spikes']), [100, 50]).mean(axis = 1)

	if "/e1_t1_mean_smoothed" in f_out:
		del(f_out['e1_t1_mean_smoothed'])
	f_out.create_dataset("e1_t1_mean_smoothed", data = e1_t1)

	if "/e2_t1_mean_smoothed" in f_out:
		del(f_out['e2_t1_mean_smoothed'])
	f_out.create_dataset("e2_t1_mean_smoothed", data = e2_t1)

	if "/V1_t1_mean_smoothed" in f_out:
		del(f_out['V1_t1_mean_smoothed'])
	f_out.create_dataset("V1_t1_mean_smoothed", data = V1_t1)

	# if "/e1_t2_mean_smoothed" in f_out:
	# 	del(f_out['e1_t2_mean_smoothed'])
	# f_out.create_dataset("e1_t2_mean_smoothed", data = e1_t2)

	# if "/e2_t2_mean_smoothed" in f_out:
	# 	del(f_out['e2_t2_mean_smoothed'])
	# f_out.create_dataset("e2_t2_mean_smoothed", data = e2_t2)

	# if "/V1_t2_mean_smoothed" in f_out:
	# 	del(f_out['V1_t2_mean_smoothed'])
	# f_out.create_dataset("V1_t2_mean_smoothed", data = V1_t2)

	f_out.close()
	print("Complete!")


def plot_triggered_spike_rates():
	f = h5py.File("/Volumes/Ryan/Data/processed_data/V1_BMI_final/raw_data/R7_thru_V13_triggered_data.hdf5", 'r')
	e1_t1 = stats.zscore(np.asarray(f['e1_t1_mean_smoothed']))
	e2_t1 = stats.zscore(np.asarray(f['e2_t1_mean_smoothed']))
	V1_t1 = stats.zscore(np.asarray(f['V1_t1_mean_smoothed']))
	# e1_t2 = stats.zscore(np.asarray(f['e1_t2_mean_smoothed']))
	# e2_t2 = stats.zscore(np.asarray(f['e2_t2_mean_smoothed']))
	# V1_t2 = stats.zscore(np.asarray(f['V1_t2_mean_smoothed']))
	f.close()

	x = np.linspace(-6, 6, e1_t1.shape[0])

	fig, ax = plt.subplots()
	ax.plot(x,e1_t1, color = 'deepskyblue', linewidth = 3, label = "E1 units")
	ax.plot(x,e2_t1, color = 'orange', linewidth = 3, label = "E2 units")
	ax.plot(x,V1_t1, color = 'k', linewidth = 3, label = "Indirect units")
	ax.set_ylim(-6, 8)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Time (s)", fontsize = 16)
	ax.set_ylabel("Z-score", fontsize = 16)
	fig.suptitle("Rewarded target", fontsize = 18)
	ax.legend()

	# fig, ax = plt.subplots()
	# ax.plot(x,e1_t2, color = 'deepskyblue', linewidth = 3, label = "E1 units")
	# ax.plot(x,e2_t2, color = 'orange', linewidth = 3, label = "E2 units")
	# ax.plot(x,V1_t2, color = 'k', linewidth = 3, label = "Indirect units")
	# ax.set_ylim(-6, 8)
	# for tick in ax.xaxis.get_major_ticks():
	# 	tick.label.set_fontsize(14)
	# for tick in ax.yaxis.get_major_ticks():
	# 	tick.label.set_fontsize(14)
	# ax.set_xlabel("Time (s)", fontsize = 16)
	# ax.set_ylabel("Z-score", fontsize = 16)
	# fig.suptitle("Unrewarded target", fontsize = 18)
	# ax.legend()

def get_mean_frs():
	path = r"C:\Users\Ryan\Documents\data\R7_thru_V13_all_data.hdf5"
	animal_list = ["R13", "R11", "V02", "V03", "V04", "V05", "V11", "V13", "R7", "R8"]
	session_dict_late = {
	"R13":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"R11":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V02":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V03":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx"],
	"V04":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx"],
	"V05":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V11":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D11.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V13":["BMI_D05.plx", "BMI_D06.plx", "BMI_D04.plx"],
	"R7":["BMI_D05.plx", "BMI_D06.plx"],
	"R8":["BMI_D05.plx", "BMI_D06.plx"]
	}

	all_e1 = []
	all_e2 = []
	all_ind = []

	longest = 0
	for animal in animal_list:
		for session in session_dict_late[animal]:
			e1, e2, ind = ds.get_ensemble_arrays(path, animal = animal, session = session)
			if e1.shape[0] > longest:
				longest = e1.shape[0]
			for n in range(e1.shape[1]):
				all_e1.append(zscore(ss.windowRate(e1[:,n], [500,100])))
			for n in range(e2.shape[1]):
				all_e2.append(zscore(ss.windowRate(e2[:,n], [500,100])))
			for n in range(ind.shape[1]):
				all_ind.append(zscore(ss.windowRate(ind[:,n], [500,100])))
	for i in range(len(all_e1)):	
		if all_e1[i].shape[0] < longest:
			add = np.empty((longest-all_e1[i].shape[0]))
			add[:] = np.nan
			all_e1[i] = np.hstack((all_e1[i], add))
	for i in range(len(all_e2)):	
		if all_e2[i].shape[0] < longest:
			add = np.empty((longest-all_e2[i].shape[0]))
			add[:] = np.nan
			all_e2[i] = np.hstack((all_e2[i], add))
	for i in range(len(all_ind)):	
		if all_ind[i].shape[0] < longest:
			add = np.empty((longest-all_ind[i].shape[0]))
			add[:] = np.nan
			all_ind[i] = np.hstack((all_ind[i], add))
	all_e1 = np.asarray(all_e1)
	all_e2 = np.asarray(all_e2)
	all_ind = np.asarray(all_ind)

	e1_early = all_e1[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	e1_late = all_e1[:,45*60*10:50*60*10].mean(axis = 1)/(5.0*60)

	e2_early = all_e2[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	e2_late = all_e2[:,50*60*10:55*60*10].mean(axis = 1)/(5.0*60)

	ind_early = all_ind[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	ind_late = all_ind[:,50*60*10:55*60*10].mean(axis = 1)/(5.0*60)

	f_out = h5py.File(r"C:\Users\Ryan\Documents\data\R7_thru_V13_spike_rates_z.hdf5",'w-')
	
	f_out.create_dataset("all_e1", data = all_e1)
	f_out.create_dataset("all_e2", data = all_e2)
	f_out.create_dataset("all_ind", data = all_ind)

	f_out.create_dataset("e1_early", data = e1_early)
	f_out.create_dataset("e2_early", data = e2_early)
	f_out.create_dataset("ind_early", data = ind_early)

	f_out.create_dataset("e1_late", data = e1_late)
	f_out.create_dataset("e2_late", data = e2_late)
	f_out.create_dataset("ind_late", data = ind_late)

	f_out.close()

def get_mean_frs_nozscore():
	path = r"C:\Users\Ryan\Documents\data\R7_thru_V13_all_data.hdf5"
	animal_list = ["R13", "R11", "V02", "V03", "V04", "V05", "V11", "V13", "R7", "R8"]
	session_dict_late = {
	"R13":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"R11":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V02":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V03":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx"],
	"V04":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx"],
	"V05":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V11":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D11.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V13":["BMI_D05.plx", "BMI_D06.plx", "BMI_D04.plx"],
	"R7":["BMI_D05.plx", "BMI_D06.plx"],
	"R8":["BMI_D05.plx", "BMI_D06.plx"]
	}

	all_e1 = []
	all_e2 = []
	all_ind = []
	##save an array of which session/animal corresponds to which array idx
	session_ids = []
	longest = 0
	for animal in animal_list:
		for session in session_dict_late[animal]:
			session_ids.append(animal+"_"+session)
			e1, e2, ind = ds.get_ensemble_arrays(path, animal = animal, session = session)
			if e1.shape[0] > longest:
				longest = e1.shape[0]
			for n in range(e1.shape[1]):
				all_e1.append((ss.windowRate(e1[:,n], [500,100])))
			for n in range(e2.shape[1]):
				all_e2.append((ss.windowRate(e2[:,n], [500,100])))
			for n in range(ind.shape[1]):
				all_ind.append((ss.windowRate(ind[:,n], [500,100])))
	for i in range(len(all_e1)):	
		if all_e1[i].shape[0] < longest:
			add = np.empty((longest-all_e1[i].shape[0]))
			add[:] = np.nan
			all_e1[i] = np.hstack((all_e1[i], add))
	for i in range(len(all_e2)):	
		if all_e2[i].shape[0] < longest:
			add = np.empty((longest-all_e2[i].shape[0]))
			add[:] = np.nan
			all_e2[i] = np.hstack((all_e2[i], add))
	for i in range(len(all_ind)):	
		if all_ind[i].shape[0] < longest:
			add = np.empty((longest-all_ind[i].shape[0]))
			add[:] = np.nan
			all_ind[i] = np.hstack((all_ind[i], add))
	all_e1 = np.asarray(all_e1)
	all_e2 = np.asarray(all_e2)
	all_ind = np.asarray(all_ind)

	e1_early = all_e1[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	e1_late = all_e1[:,45*60*10:50*60*10].mean(axis = 1)/(5.0*60)

	e2_early = all_e2[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	e2_late = all_e2[:,50*60*10:55*60*10].mean(axis = 1)/(5.0*60)

	ind_early = all_ind[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	ind_late = all_ind[:,50*60*10:55*60*10].mean(axis = 1)/(5.0*60)

	f_out = h5py.File(r"C:\Users\Ryan\Documents\data\R7_thru_V13_spike_rates_binned.hdf5",'w-')
	
	f_out.create_dataset("all_e1", data = all_e1)
	f_out.create_dataset("all_e2", data = all_e2)
	f_out.create_dataset("all_ind", data = all_ind)

	f_out.create_dataset("e1_early", data = e1_early)
	f_out.create_dataset("e2_early", data = e2_early)
	f_out.create_dataset("ind_early", data = ind_early)

	f_out.create_dataset("e1_late", data = e1_late)
	f_out.create_dataset("e2_late", data = e2_late)
	f_out.create_dataset("ind_late", data = ind_late)
	f_out.create_dataset("session_ids",data=np.asarray(session_ids))

	f_out.close()


def plot_fr_data():
	f = h5py.File(r"C:\Users\Ryan\Documents\data\R7_thru_V13_spike_rates_z.hdf5", 'r')

	e1_early = np.asarray(f['e1_early'])
	e1_late = np.asarray(f['e1_late'])
	e2_early = np.asarray(f['e2_early'])
	e2_late = np.asarray(f['e2_late'])
	ind_early = np.asarray(f['ind_early'])
	ind_late = np.asarray(f['ind_late'])
	f.close()

	labels = np.array(['e1_early', 'e1_late', 'e2_early', 'e2_late', 'ind_early', 'ind_late'])
	means = np.array([np.nanmean(e1_early), np.nanmean(e1_late), np.nanmean(e2_early), 
		np.nanmean(e2_late),np.nanmean(ind_early), np.nanmean(ind_late)])
	sem = np.array([np.nanstd(e1_early)/np.sqrt(94), np.nanstd(e1_late)/np.sqrt(94), 
		np.nanstd(e2_early)/np.sqrt(92), np.nanstd(e2_late)/np.sqrt(92),
		np.nanstd(ind_early)/np.sqrt(182), np.nanstd(ind_late)/np.sqrt(182)])

	p_val_e1 = stats.ttest_rel(e1_early, e1_late, nan_policy='omit')[1]
	p_val_e2 = stats.ttest_rel(e2_early, e2_late, nan_policy='omit')[1]
	p_val_ind = stats.ttest_rel(ind_early, ind_late, nan_policy='omit')[1]

	p_val_e1_e2_early = stats.ttest_ind(e1_early, e2_early, nan_policy='omit')[1]
	p_val_e1_e2_late = stats.ttest_ind(e1_late, e2_late, nan_policy='omit')[1]
	p_val_e1_ind_early = stats.ttest_ind(e1_early, ind_early, nan_policy='omit')[1]
	p_val_e1_ind_late = stats.ttest_ind(e1_late, ind_late, nan_policy='omit')[1]
	p_val_e2_ind_early = stats.ttest_ind(e2_early, ind_early, nan_policy='omit')[1]
	p_val_e2_ind_late = stats.ttest_ind(e2_late, ind_late, nan_policy='omit')[1]


	idx = np.arange(6)
	width = 1.0
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, yerr = sem, ecolor = 'k')
	#ax.set_ylim(0,0.9)
	#ax.set_xlim(-0.5, 3.5)
	ax.set_xticks(idx+0.5)
	ax.set_xticklabels(labels)
	# for i in range(data.shape[0]):
	# 	plt.plot((idx+0.5), data[i,:], alpha = 0.5, color = np.random.rand(3,), marker = 'o', linewidth = 2)
	ax.set_ylabel("firing rate", fontsize = 14)
	ax.set_xlabel("Condition", fontsize = 14)

def get_ensemble_correlations():
	path_in = "/Volumes/Ryan/Data/processed_data/V1_BMI_final/raw_data/R7_thru_V13_all_data.hdf5"
	path_out = "/Volumes/data/V1_BMI/Neuron/data/correlations_5ms.hdf5"

	f_out = h5py.File(path_out, 'a')

	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13", "R7", "R8"]
	within_e1_by_animal = []
	within_e2_by_animal = []
	between_e1_e2_by_animal = []

	longest = 0
	for animal in animal_list:
		print("Select "+animal)
		within_e1, within_e2, between_e1_e2 = ds.ensemble_correlations(path_in,dt=5)
		f_out.create_group(animal)
		f_out[animal].create_dataset("all_within_e1", data = within_e1)
		f_out[animal].create_dataset("all_within_e2", data = within_e2)
		f_out[animal].create_dataset("all_between_e1_e2", data = between_e1_e2)
		f_out[animal].create_dataset("mean_within_e1", data = within_e1.mean(axis = 0))
		f_out[animal].create_dataset("mean_within_e2", data = within_e2.mean(axis = 0))
		f_out[animal].create_dataset("mean_between_e1_e2", data = between_e1_e2.mean(axis = 0))
		within_e1_by_animal.append(within_e1.mean(axis = 0))
		within_e2_by_animal.append(within_e2.mean(axis = 0))
		between_e1_e2_by_animal.append(between_e1_e2.mean(axis = 0))
		if within_e1.shape[1] > longest:
			longest = within_e1.shape[1]
			#print "longest = " + str(longest)
	
	for i in range(len(within_e1_by_animal)): 
		if within_e1_by_animal[i].size < longest:
			add = np.empty((longest-within_e1_by_animal[i].size))
			add[:] = np.nan
			within_e1_by_animal[i] = np.hstack((within_e1_by_animal[i], add))
			within_e2_by_animal[i] = np.hstack((within_e2_by_animal[i], add))
			between_e1_e2_by_animal[i] = np.hstack((between_e1_e2_by_animal[i], add))
	f_out.create_dataset("within_e1_by_animal", data = np.asarray(within_e1_by_animal))
	f_out.create_dataset("within_e2_by_animal", data = np.asarray(within_e2_by_animal))
	f_out.create_dataset("between_e1_e2_by_animal", data = np.asarray(between_e1_e2_by_animal))
	f_out.close()
	print("complete!")

def plot_cursor_states():
	f = h5py.File(r"Z:\Data\processed_data\V1_BMI_final\raw_data\R7_thru_V13_ensemble_state_data.hdf5", 'r')
	data = {'early':np.asarray(f['early_cvals']),'late':np.asarray(f['late_cvals'])}
	n_late, bins_late, patches_late = plt.hist(data['late'], bins = 50, facecolor = 'green', alpha = 0.4, log=False)
	plt.figure()
	n_early, bins_early, patches_early = plt.hist(data['early'], bins = bins_late, facecolor = 'red', alpha = 0.4,log=False)
	plt.figure()
	ax1=sns.distplot(np.asarray(f['late_cvals']),bins=bins_late, kde=False,color ='g')
	plt.figure()
	ax2=sns.distplot(np.asarray(f['early_cvals']),bins=bins_late, kde=False, color = 'r')
	ax1.set_yscale("log")
	ax2.set_yscale("log")

def get_dark_session_data():
	source_file = r"Z:\Data\processed_data\V1_BMI_final\raw_data\R7_thru_V13_learning_event_data2.hdf5"
	f = h5py.File(source_file, 'r+') 
	animal_list = ["V01", "V02", "V03", "V04", "V05", "V11", "V13"]
	for animal in animal_list: 
		arrays = []
		try:
			t1 = np.asarray(f[animal]['t1'])
			arrays.append(t1)
		except KeyError:
			print("No t1's.")
		try:
			t2 = np.asarray(f[animal]['t2'])
			arrays.append(t2)
		except KeyError:
			print("No t2's.")
			t2 = np.zeros(t1.shape)
			arrays.append(t2)
		try:
			miss = np.asarray(f[animal]['miss'])
			arrays.append(miss)
		except KeyError:
			print("No Misses.")
			miss = np.zeros(t1.shape)
			arrays.append(miss)
		##figure out the size of the largest array
		longest = 0
		for array in arrays:
			if array.shape[1] > longest: 
				longest = array.shape[1]
		##append some zeros on to the other arrays to make them all the same shape
		for idx in range(len(arrays)):
			difference = longest - arrays[idx].shape[1]
			if difference > 0:
				arrays[idx] = np.hstack((arrays[idx], np.zeros((arrays[idx].shape[0], difference))))
		##get the success rate using a sliding window for each session
		N = longest
		num_sessions = t1.shape[0]
		Nwin = 1000*60*3
		Nstep = 1000*30
		winstart = np.arange(0,N-Nwin, Nstep)
		nw = winstart.shape[0]
		result = np.zeros((num_sessions, nw))
		result2 = np.zeros((num_sessions, nw))
		for session in range(num_sessions):
			t1_counts = arrays[0][session,:]
			t2_counts = arrays[1][session,:]
			total_counts = arrays[0][session,:] + arrays[1][session,:] + arrays[2][session,:]
			for n in range(nw):
				idx = np.arange(winstart[n], winstart[n]+Nwin)
				t1_win = t1_counts[idx]
				t2_win = t2_counts[idx]
				total_win = total_counts[idx]
				if total_win.sum() != 0:
					result[session, n] = t1_win.sum()/total_win.sum()
					result2[session, n] = t2_win.sum()/total_win.sum()
		if "/"+animal+"/correct_within_dark_only" in f:
			del(f[animal]['correct_within_dark_only'])
		f[animal].create_dataset("correct_within_dark_only", data = result)
		if "/"+animal+"/t2_within_dark_only" in f:
			del(f[animal]['t2_within_dark_only'])
		f[animal].create_dataset("t2_within_dark_only", data = result2)
	##figure out the shape of the combined dataset
	total_sessions = 0
	longest_session = 0
	for animal in animal_list:
		total_sessions += f[animal]['correct_within_dark_only'].shape[0]
		if f[animal]['correct_within_dark_only'].shape[1] > longest_session:
			longest_session = f[animal]['correct_within_dark_only'].shape[1]
	all_sessions = np.zeros((total_sessions, longest_session))
	all_sessions_t2 = np.zeros((total_sessions, longest_session))
	current_session = 0
	for animal in animal_list:
		data = np.asarray(f[animal]['correct_within_dark_only'])
		data_t2 = np.asarray(f[animal]['t2_within_dark_only'])
		##add some NANs to equalize array length
		if data.shape[1] < longest_session:
			add = np.empty((data.shape[0], longest_session-data.shape[1]))
			add[:] = np.nan
			data = np.hstack((data, add))
			data_t2 = np.hstack((data_t2,add))
		all_sessions[current_session:current_session+data.shape[0], 0:longest_session] = data
		all_sessions_t2[current_session:current_session+data.shape[0], 0:longest_session] = data_t2
		current_session += data.shape[0]
	if "/dark_sessions" in f:
		del(f['dark_sessions'])
	f.create_dataset('dark_sessions', data = all_sessions)
	if "/dark_sessions_t2" in f:
		del(f['dark_sessions_t2'])
	f.create_dataset('dark_sessions_t2', data = all_sessions_t2)
	f.close()

def get_light_session_data():
	source_file = r"Z:\Data\processed_data\V1_BMI_final\raw_data\R7_thru_V13_learning_event_data2.hdf5"
	f = h5py.File(source_file, 'r+') 
	animal_list = ["R13", "R11", "R7", "R8"]
	for animal in animal_list: 
		arrays = []
		try:
			t1 = np.asarray(f[animal]['t1'])
			arrays.append(t1)
		except KeyError:
			print("No t1's.")
		try:
			t2 = np.asarray(f[animal]['t2'])
			arrays.append(t2)
		except KeyError:
			print("No t2's.")
			t2 = np.zeros(t1.shape)
			arrays.append(t2)
		try:
			miss = np.asarray(f[animal]['miss'])
			arrays.append(miss)
		except KeyError:
			print("No Misses.")
			miss = np.zeros(t1.shape)
			arrays.append(miss)
		##figure out the size of the largest array
		longest = 0
		for array in arrays:
			if array.shape[1] > longest: 
				longest = array.shape[1]
		##append some zeros on to the other arrays to make them all the same shape
		for idx in range(len(arrays)):
			difference = longest - arrays[idx].shape[1]
			if difference > 0:
				arrays[idx] = np.hstack((arrays[idx], np.zeros((arrays[idx].shape[0], difference))))
		##get the success rate using a sliding window for each session
		N = longest
		num_sessions = t1.shape[0]
		Nwin = 1000*60*3
		Nstep = 1000*30
		winstart = np.arange(0,N-Nwin, Nstep)
		nw = winstart.shape[0]
		result = np.zeros((num_sessions, nw))
		result2 = np.zeros((num_sessions, nw))
		for session in range(num_sessions):
			t1_counts = arrays[0][session,:]
			t2_counts = arrays[1][session,:]
			total_counts = arrays[0][session,:] + arrays[1][session,:] + arrays[2][session,:]
			for n in range(nw):
				idx = np.arange(winstart[n], winstart[n]+Nwin)
				t1_win = t1_counts[idx]
				t2_win = t2_counts[idx]
				total_win = total_counts[idx]
				if total_win.sum() != 0:
					result[session, n] = t1_win.sum()/total_win.sum()
					result2[session, n] = t2_win.sum()/total_win.sum()
		if "/"+animal+"/correct_within_light_only" in f:
			del(f[animal]['correct_within_light_only'])
		f[animal].create_dataset("correct_within_light_only", data = result)
		if "/"+animal+"/t2_within_light_only" in f:
			del(f[animal]['t2_within_light_only'])
		f[animal].create_dataset("t2_within_light_only", data = result2)
	##figure out the shape of the combined dataset
	total_sessions = 0
	longest_session = 0
	for animal in animal_list:
		total_sessions += f[animal]['correct_within_light_only'].shape[0]
		if f[animal]['correct_within_light_only'].shape[1] > longest_session:
			longest_session = f[animal]['correct_within_light_only'].shape[1]
	all_sessions = np.zeros((total_sessions, longest_session))
	all_sessions_t2 = np.zeros((total_sessions, longest_session))
	current_session = 0
	for animal in animal_list:
		data = np.asarray(f[animal]['correct_within_light_only'])
		data_t2 = np.asarray(f[animal]['t2_within_light_only'])
		##add some NANs to equalize array length
		if data.shape[1] < longest_session:
			add = np.empty((data.shape[0], longest_session-data.shape[1]))
			add[:] = np.nan
			data = np.hstack((data, add))
			data_t2 = np.hstack((data_t2,add))
		all_sessions[current_session:current_session+data.shape[0], 0:longest_session] = data
		all_sessions_t2[current_session:current_session+data.shape[0], 0:longest_session] = data_t2
		current_session += data.shape[0]
	if "/light_sessions" in f:
		del(f['light_sessions'])
	f.create_dataset('light_sessions', data = all_sessions)
	if "/light_sessions_t2" in f:
		del(f['light_sessions_t2'])
	f.create_dataset('light_sessions_t2', data = all_sessions_t2)
	f.close()

def plot_within_session_light_dark():
	source_file = r"Z:\Data\processed_data\V1_BMI_final\raw_data\R7_thru_V13_learning_event_data2.hdf5"
	f = h5py.File(source_file, 'r')
	light_data = np.asarray(f['light_sessions'][0:-6,:])
	dark_data = np.vstack((np.asarray(f['dark_sessions']), light_data[-6:,0:150]))
	f.close()
	light_mean = np.nanmean(light_data, axis = 0)
	light_std = np.nanstd(light_data, axis = 0)
	light_sem = light_std/np.sqrt(light_data.shape[0])
	dark_mean = np.nanmean(dark_data, axis = 0)
	dark_std = np.nanstd(dark_data, axis = 0)
	dark_sem = dark_std/np.sqrt(dark_data.shape[0])
	x_axis = np.linspace(0,115,light_mean.shape[0])
	fig, ax = plt.subplots()
	ax.plot(x_axis, light_mean, linewidth = 3, color = 'yellow', label = "train light")
	plt.fill_between(x_axis, light_mean-light_sem, light_mean+light_sem, 
		alpha = 0.5, facecolor = 'yellow')
	ax.plot(x_axis[0:dark_mean.size], dark_mean, linewidth = 3, color = 'k', label = "train dark")
	plt.fill_between(x_axis[0:dark_mean.size], dark_mean-dark_sem, dark_mean+dark_sem, 
		alpha = 0.5, facecolor = 'k')
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Time in session, mins", fontsize = 16)
	ax.set_ylabel("Percent rewarded", fontsize = 16)
	fig.suptitle("Light vs dark training", fontsize = 18)
	ax.set_xlim((0,50))
	ax.fill_between(x_axis, .25,.39, alpha = 0.1, facecolor = 'cyan')
	ax.legend()
	##now do a test of differences and plot the bar graph
	light_early = light_data[:,0:8].mean(axis=1)
	light_late = light_data[:,76:83].mean(axis=1)
	dark_early = dark_data[:,0:8].mean(axis=1)
	dark_late = dark_data[:,76:83].mean(axis=1)
	p_val_light = stats.ttest_rel(light_early,light_late)[1]
	p_val_dark = stats.ttest_rel(dark_early,dark_late)[1]
	p_val_light_dark = stats.ttest_ind(dark_late,light_late)[1]
	#plot the bars
	means = [light_early.mean(),light_late.mean(),dark_early.mean(),dark_late.mean()]
	sem = [light_early.std()/light_early.size,light_late.std()/light_late.size,
			dark_early.std()/dark_early.size,dark_late.std()/dark_late.size]
	idx = np.arange(4)
	width = 1.0
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, yerr = sem, ecolor = 'k',color=['y','y','k','k'])
	labels = ['light_early','light_late','dark_early','dark_late']
	ax.set_xticks(idx+0.5)
	ax.set_xticklabels(labels)
	ax.set_ylabel("percentage correct", fontsize = 14)
	ax.set_xlabel("Condition", fontsize = 14)
	print("pval light early-late= "+str(p_val_light))
	print("pval dark early-late= "+str(p_val_dark))
	print("pval light-dark-late= "+str(p_val_light_dark))


def plot_light_change_sessions():
	f = h5py.File("/Volumes/Untitled/Ryan/V1_BMI/Neuron/R7_thru_V13_light_data.hdf5",'r')
	data = np.asarray(f['scaled_p_correct_all'][0:5])
	f.close()
	mean = np.nanmean(data,axis=0)
	std = np.nanstd(data,axis=0)
	sem = std/np.sqrt(data.shape[0])
	x_axis = np.linspace(0,120,mean.shape[0])
	fig, ax = plt.subplots()
	ax.plot(x_axis, mean, linewidth = 2, color = 'r')
	plt.fill_between(x_axis, mean-sem, mean+sem, 
		alpha = 0.5, facecolor = 'r')
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Time in session, mins", fontsize = 16)
	ax.set_ylabel("Percent rewarded", fontsize = 16)
	ax.set_ylim((0,1))
	fig.suptitle("Light change training", fontsize = 18)

	##for the line graphs
	light = np.array([data[0,45:60].mean(),data[1,35:50].mean(),
		data[2,35:50].mean(),data[3,45:60].mean()])
	dark = np.array([data[0,65:80].mean(),data[1,65:80].mean(),
		data[2,60:75].mean(),data[3,68:83].mean()])
	light_mean = light.mean()
	dark_mean = dark.mean()
	light_sem = light.std()/np.sqrt(4)
	dark_sem = dark.std()/np.sqrt(4)
	means = np.array([light_mean,dark_mean])
	sems = np.array([light_sem,dark_sem])
	t_val,p_val = stats.ttest_rel(light,dark)

	fig, ax2 = plt.subplots(1)
	rew = np.vstack((light,dark))
	xr = np.array([0,1])
	err_x = np.array([0,1])
	yerr = sems
	xerr = np.ones(2)*0.25
	for i in range(rew.shape[1]):
		ax2.plot(xr,rew[:,i],color='k',linewidth=2,marker='o')
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),['Dark','Light'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.3,1.3)

	print("pval = "+str(p_val))
	print("tval = "+str(t_val))
	print("mean light = "+str(means[0]))
	print("mean dark = "+str(means[1]))


def save_V1_ds_ff_cohgram_data():	
	f = h5py.File("/home/lab/Documents/data/t1_triggered.hdf5",'r')
	sessions = f.keys()
	V1_data = []
	DMS_data = []
	session_names = []
	for s in sessions:
		try:
			v1 = None
			dms = None
			name = None
			v1 = np.asarray(f[s]['V1_lfp'][:,2000:,:])
			dms = np.asarray(f[s]['Str_lfp'][:,2000:,:])
			name = s
		except KeyError:
			pass
		if (v1 != None and dms != None):
			if (v1.shape[0] > 2 and dms.shape[0] == v1.shape[0]): ##need at least 2 trials
				V1_data.append(v1)
				DMS_data.append(dms)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_v1_dms_lfp_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("v1", data=V1_data[i])
		gp.create_dataset("dms",data=DMS_data[i])
	g.close()
	DMS_data = None; V1_data = None
	g = h5py.File("/home/lab/Documents/data/paired_v1_dms_lfp_t1.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/v1_dms_cohgrams_t12.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		v1_data = np.asarray(group['v1'])
		dms_data = np.asarray(group['dms'])
		data = []
		for v in range(v1_data.shape[2]):
			for d in range(dms_data.shape[2]):
				lfp_1 = v1_data[:,:,v].T
				lfp_2 = dms_data[:,:,d].T
				data.append([lfp_1,lfp_2])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(ss.mp_cohgrams,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def save_V1_ds_ff_cohgram_data_ctrl():	##use this for non-task relevant periods
	f = h5py.File("/home/lab/Documents/data/non_task_times.hdf5",'r')
	sessions = f.keys()
	V1_data = []
	DMS_data = []
	session_names = []
	for s in sessions:
		try:
			v1 = None
			dms = None
			name = None
			v1 = np.asarray(f[s]['V1_lfp'][:,2000:,:])
			dms = np.asarray(f[s]['Str_lfp'][:,2000:,:])
			name = s
		except KeyError:
			pass
		if (v1 != None and dms != None):
			if (v1.shape[0] > 2 and dms.shape[0] == v1.shape[0]): ##need at least 2 trials
				V1_data.append(v1)
				DMS_data.append(dms)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_v1_dms_lfp_ctrl.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("v1", data=V1_data[i])
		gp.create_dataset("dms",data=DMS_data[i])
	g.close()
	DMS_data = None; V1_data = None
	g = h5py.File("/home/lab/Documents/data/paired_v1_dms_lfp_ctrl.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/v1_dms_cohgrams_ctrl.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		v1_data = np.asarray(group['v1'])
		dms_data = np.asarray(group['dms'])
		data = []
		for v in range(v1_data.shape[2]):
			for d in range(dms_data.shape[2]):
				lfp_1 = v1_data[:,:,v].T
				lfp_2 = dms_data[:,:,d].T
				data.append([lfp_1,lfp_2])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(ss.mp_cohgrams,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def save_e1_V1_sf_cohgram_data():	
	f = h5py.File("/home/lab/Documents/data/t1_triggered.hdf5",'r')
	sessions = f.keys()
	e1_data = []
	V1_data = []
	session_names = []
	for s in sessions:
		try:
			e1 = None
			v1 = None
			name = None
			e1 = np.asarray(f[s]['e1_units'][:,:,:]) 
			v1 = np.asarray(f[s]['V1_lfp'][:,:,:])
			name = s
		except KeyError:
			pass
		if (e1 != None and v1 != None):
			if (e1.shape[0] > 2 and v1.shape[0] == e1.shape[0]): ##need at least 2 trials
				e1_data.append(e1)
				V1_data.append(v1)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_e1_v1_sf_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("e1", data=e1_data[i])
		gp.create_dataset("v1",data=V1_data[i])
	g.close()
	e1_data = None; V1_data = None
	g = h5py.File("/home/lab/Documents/data/paired_e1_v1_sf_t1.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/e1_v1_cohgrams_t1.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		e1_data = np.asarray(group['e1'])
		v1_data = np.asarray(group['v1'])
		data = []
		for v in range(e1_data.shape[2]):
			for d in range(v1_data.shape[2]):
				spikes = e1_data[:,:,v].T
				lfp = v1_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sfc,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def save_e1_V1_sf_cohgram_data_ctrl():	
	f = h5py.File("/home/lab/Documents/data/non_task_times.hdf5",'r')
	sessions = f.keys()
	e1_data = []
	V1_data = []
	session_names = []
	for s in sessions:
		try:
			e1 = None
			v1 = None
			name = None
			e1 = np.asarray(f[s]['e1_units'][:,:,:]) 
			v1 = np.asarray(f[s]['V1_lfp'][:,:,:])
			name = s
		except KeyError:
			pass
		if (e1 != None and v1 != None):
			if (e1.shape[0] > 2 and v1.shape[0] == e1.shape[0]): ##need at least 2 trials
				e1_data.append(e1)
				V1_data.append(v1)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_e1_v1_sf_ctrl.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("e1", data=e1_data[i])
		gp.create_dataset("v1",data=V1_data[i])
	g.close()
	e1_data = None; V1_data = None
	g = h5py.File("/home/lab/Documents/data/paired_e1_v1_sf_ctrl.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/e1_v1_cohgrams_ctrl.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		e1_data = np.asarray(group['e1'])
		v1_data = np.asarray(group['v1'])
		data = []
		for v in range(e1_data.shape[2]):
			for d in range(v1_data.shape[2]):
				spikes = e1_data[:,:,v].T
				lfp = v1_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sfc,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def save_e2_V1_sf_cohgram_data():	
	f = h5py.File("/home/lab/Documents/data/t1_triggered.hdf5",'r')
	sessions = f.keys()
	e1_data = []
	V1_data = []
	session_names = []
	for s in sessions:
		try:
			e1 = None
			v1 = None
			name = None
			e1 = np.asarray(f[s]['e2_units'][:,:,:]) 
			v1 = np.asarray(f[s]['V1_lfp'][:,:,:])
			name = s
		except KeyError:
			pass
		if (e1 != None and v1 != None):
			if (e1.shape[0] > 2 and v1.shape[0] == e1.shape[0]): ##need at least 2 trials
				e1_data.append(e1)
				V1_data.append(v1)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_e2_v1_sf_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("e1", data=e1_data[i])
		gp.create_dataset("v1",data=V1_data[i])
	g.close()
	e1_data = None; V1_data = None
	g = h5py.File("/home/lab/Documents/data/paired_e2_v1_sf_t1.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/e2_v1_cohgrams_t1.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		e1_data = np.asarray(group['e1'])
		v1_data = np.asarray(group['v1'])
		data = []
		for v in range(e1_data.shape[2]):
			for d in range(v1_data.shape[2]):
				spikes = e1_data[:,:,v].T
				lfp = v1_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sfc,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def save_e2_V1_sf_cohgram_data_ctrl():	
	f = h5py.File("/home/lab/Documents/data/non_task_times.hdf5",'r')
	sessions = f.keys()
	e1_data = []
	V1_data = []
	session_names = []
	for s in sessions:
		try:
			e1 = None
			v1 = None
			name = None
			e1 = np.asarray(f[s]['e2_units'][:,:,:]) 
			v1 = np.asarray(f[s]['V1_lfp'][:,:,:])
			name = s
		except KeyError:
			pass
		if (e1 != None and v1 != None):
			if (e1.shape[0] > 2 and v1.shape[0] == e1.shape[0]): ##need at least 2 trials
				e1_data.append(e1)
				V1_data.append(v1)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_e2_v1_sf_ctrl.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("e1", data=e1_data[i])
		gp.create_dataset("v1",data=V1_data[i])
	g.close()
	e1_data = None; V1_data = None
	g = h5py.File("/home/lab/Documents/data/paired_e2_v1_sf_ctrl.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/e2_v1_cohgrams_ctrl.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		e1_data = np.asarray(group['e1'])
		v1_data = np.asarray(group['v1'])
		data = []
		for v in range(e1_data.shape[2]):
			for d in range(v1_data.shape[2]):
				spikes = e1_data[:,:,v].T
				lfp = v1_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sfc,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def save_indirect_V1_sf_cohgram_data():	
	f = h5py.File("/home/lab/Documents/data/t1_triggered.hdf5",'r')
	sessions = f.keys()
	e1_data = []
	V1_data = []
	session_names = []
	for s in sessions:
		try:
			e1 = None
			v1 = None
			name = None
			e1 = np.asarray(f[s]['V1_units'][:,:,:]) 
			v1 = np.asarray(f[s]['V1_lfp'][:,:,:])
			name = s
		except KeyError:
			pass
		if (e1 != None and v1 != None):
			if (e1.shape[0] > 2 and v1.shape[0] == e1.shape[0]): ##need at least 2 trials
				e1_data.append(e1)
				V1_data.append(v1)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_ind_v1_sf_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("e1", data=e1_data[i])
		gp.create_dataset("v1",data=V1_data[i])
	g.close()
	e1_data = None; V1_data = None
	g = h5py.File("/home/lab/Documents/data/paired_ind_v1_sf_t1.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/ind_v1_cohgrams_t1.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		e1_data = np.asarray(group['e1'])
		v1_data = np.asarray(group['v1'])
		data = []
		for v in range(e1_data.shape[2]):
			for d in range(v1_data.shape[2]):
				spikes = e1_data[:,:,v].T
				lfp = v1_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sfc,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def save_V1_ds_ff_coherence_data():	
	f = h5py.File("/home/lab/Documents/data/t1_triggered.hdf5",'r')
	sessions = f.keys()
	V1_data = []
	DMS_data = []
	session_names = []
	for s in sessions:
		try:
			v1 = None
			dms = None
			name = None
			v1 = np.asarray(f[s]['V1_lfp'][:,2000:,:])
			dms = np.asarray(f[s]['Str_lfp'][:,2000:,:])
			name = s
		except KeyError:
			pass
		if (v1 != None and dms != None):
			if (v1.shape[0] > 2 and dms.shape[0] == v1.shape[0]): ##need at least 2 trials
				V1_data.append(v1)
				DMS_data.append(dms)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_v1_dms_lfp_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("v1", data=V1_data[i])
		gp.create_dataset("dms",data=DMS_data[i])
	g.close()
	DMS_data = None; V1_data = None
	g = h5py.File("/home/lab/Documents/data/paired_v1_dms_lfp_t1.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/v1_dms_cohgrams_t12.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		v1_data = np.asarray(group['v1'])
		dms_data = np.asarray(group['dms'])
		data = []
		for v in range(v1_data.shape[2]):
			for d in range(dms_data.shape[2]):
				lfp_1 = v1_data[:,:,v].T
				lfp_2 = dms_data[:,:,d].T
				data.append([lfp_1,lfp_2])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(ss.mp_cohgrams,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def save_V1_ds_ff_coherence_data_ctrl():	
	f = h5py.File("/home/lab/Documents/data/non_task_times.hdf5",'r')
	sessions = f.keys()
	V1_data = []
	DMS_data = []
	session_names = []
	for s in sessions:
		try:
			v1 = None
			dms = None
			name = None
			v1 = np.asarray(f[s]['V1_lfp'][:,2000:,:])
			dms = np.asarray(f[s]['Str_lfp'][:,2000:,:])
			name = s
		except KeyError:
			pass
		if (v1 != None and dms != None):
			if (v1.shape[0] > 2 and dms.shape[0] == v1.shape[0]): ##need at least 2 trials
				V1_data.append(v1)
				DMS_data.append(dms)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_v1_dms_lfp_ctrl.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("v1", data=V1_data[i])
		gp.create_dataset("dms",data=DMS_data[i])
	g.close()
	DMS_data = None; V1_data = None
	g = h5py.File("/home/lab/Documents/data/paired_v1_dms_lfp_ctrl.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/v1_dms_cohgrams_ctrl.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		v1_data = np.asarray(group['v1'])
		dms_data = np.asarray(group['dms'])
		data = []
		for v in range(v1_data.shape[2]):
			for d in range(dms_data.shape[2]):
				lfp_1 = v1_data[:,:,v].T
				lfp_2 = dms_data[:,:,d].T
				data.append([lfp_1,lfp_2])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(ss.mp_cohgrams,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()


def save_e1_V1_sf_coherence_data():	
	f = h5py.File(r"C:\Users\Ryan\Documents\data\t1_triggered.hdf5",'r')
	sessions = f.keys()
	e1_data = []
	V1_data = []
	session_names = []
	for s in sessions:
		try:
			e1 = None
			v1 = None
			name = None
			e1 = np.asarray(f[s]['e1_units'][:,3000:-1000,:])
			v1 = np.asarray(f[s]['V1_lfp'][:,3000:-1000,:])
			name = s
		except KeyError:
			pass
		if (e1 != None and v1 != None):
			if (e1.shape[0] > 2 and e1.shape[0] == v1.shape[0]): ##need at least 2 trials
				e1_data.append(e1)
				V1_data.append(v1)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_e1_v1_hybrid_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("e1", data=e1_data[i])
		gp.create_dataset("v1",data=V1_data[i])
	g.close()
	e1_data = None; V1_data = None
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_e1_v1_hybrid_t1.hdf5",'r')
	results_file = h5py.File(r"C:\Users\Ryan\Documents\data\e1_v1_sfcoherence_t1.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		e1_data = np.asarray(group['e1'])
		v1_data = np.asarray(group['v1'])
		data = []
		for v in range(e1_data.shape[2]):
			for d in range(v1_data.shape[2]):
				spikes = e1_data[:,:,v].T
				lfp = v1_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sf_coherence,data)
		pool.close()
		pool.join()
		result_data = async_result.get() ##this will return a list of lists, 
		##index 0 is the coherence, index 1 is the confidence intervals
		##separate them out:
		coherence = []
		confC = []
		for i in range(len(result_data)):
			coherence.append(result_data[i][0])
			confC.append(result_data[i][1])
		results_file.create_dataset(session,data = np.asarray(coherence))
		results_file.create_dataset(session+"_err", data=np.asarray(confC))
	g.close()
	results_file.close()

def save_e2_V1_sf_coherence_data():	
	f = h5py.File(r"C:\Users\Ryan\Documents\data\t1_triggered.hdf5",'r')
	sessions = f.keys()
	e1_data = []
	V1_data = []
	session_names = []
	for s in sessions:
		try:
			e1 = None
			v1 = None
			name = None
			e1 = np.asarray(f[s]['e2_units'][:,3000:-1000,:])
			v1 = np.asarray(f[s]['V1_lfp'][:,3000:-1000,:])
			name = s
		except KeyError:
			pass
		if (e1 != None and v1 != None):
			if (e1.shape[0] > 2 and e1.shape[0] == v1.shape[0]): ##need at least 2 trials
				e1_data.append(e1)
				V1_data.append(v1)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_e2_v1_hybrid_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("e1", data=e1_data[i])
		gp.create_dataset("v1",data=V1_data[i])
	g.close()
	e1_data = None; V1_data = None
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_e2_v1_hybrid_t1.hdf5",'r')
	results_file = h5py.File(r"C:\Users\Ryan\Documents\data\e2_v1_sfcoherence_t1.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		e1_data = np.asarray(group['e1'])
		v1_data = np.asarray(group['v1'])
		data = []
		for v in range(e1_data.shape[2]):
			for d in range(v1_data.shape[2]):
				spikes = e1_data[:,:,v].T
				lfp = v1_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sf_coherence,data)
		pool.close()
		pool.join()
		result_data = async_result.get() ##this will return a list of lists, 
		##index 0 is the coherence, index 1 is the confidence intervals
		##separate them out:
		coherence = []
		confC = []
		for i in range(len(result_data)):
			coherence.append(result_data[i][0])
			confC.append(result_data[i][1])
		results_file.create_dataset(session,data = np.asarray(coherence))
		results_file.create_dataset(session+"_err", data=np.asarray(confC))
	g.close()
	results_file.close()

def save_e1_V1_sf_coherence_ctrl():	
	f = h5py.File(r"C:\Users\Ryan\Documents\data\non_task_times.hdf5",'r')
	sessions = f.keys()
	e1_data = []
	V1_data = []
	session_names = []
	for s in sessions:
		try:
			e1 = None
			v1 = None
			name = None
			e1 = np.asarray(f[s]['e1_units'][:,3000:-1000,:])
			v1 = np.asarray(f[s]['V1_lfp'][:,3000:-1000,:])
			name = s
		except KeyError:
			pass
		if (e1 != None and v1 != None):
			if (e1.shape[0] > 2 and e1.shape[0] == v1.shape[0]): ##need at least 2 trials
				e1_data.append(e1)
				V1_data.append(v1)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_e1_v1_hybrid_ctrl.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("e1", data=e1_data[i])
		gp.create_dataset("v1",data=V1_data[i])
	g.close()
	e1_data = None; V1_data = None
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_e1_v1_hybrid_ctrl.hdf5",'r')
	results_file = h5py.File(r"C:\Users\Ryan\Documents\data\e1_v1_sfcoherence_ctrl.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		e1_data = np.asarray(group['e1'])
		v1_data = np.asarray(group['v1'])
		data = []
		for v in range(e1_data.shape[2]):
			for d in range(v1_data.shape[2]):
				spikes = e1_data[:,:,v].T
				lfp = v1_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sf_coherence,data)
		pool.close()
		pool.join()
		result_data = async_result.get() ##this will return a list of lists, 
		##index 0 is the coherence, index 1 is the confidence intervals
		##separate them out:
		coherence = []
		confC = []
		for i in range(len(result_data)):
			coherence.append(result_data[i][0])
			confC.append(result_data[i][1])
		results_file.create_dataset(session,data = np.asarray(coherence))
		results_file.create_dataset(session+"_err", data=np.asarray(confC))
	g.close()
	results_file.close()

def save_e2_V1_sf_coherence_ctrl():	
	f = h5py.File(r"C:\Users\Ryan\Documents\data\non_task_times.hdf5",'r')
	sessions = f.keys()
	e1_data = []
	V1_data = []
	session_names = []
	for s in sessions:
		try:
			e1 = None
			v1 = None
			name = None
			e1 = np.asarray(f[s]['e2_units'][:,3000:-1000,:])
			v1 = np.asarray(f[s]['V1_lfp'][:,3000:-1000,:])
			name = s
		except KeyError:
			pass
		if (e1 != None and v1 != None):
			if (e1.shape[0] > 2 and e1.shape[0] == v1.shape[0]): ##need at least 2 trials
				e1_data.append(e1)
				V1_data.append(v1)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_e2_v1_hybrid_ctrl.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("e1", data=e1_data[i])
		gp.create_dataset("v1",data=V1_data[i])
	g.close()
	e1_data = None; V1_data = None
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_e2_v1_hybrid_ctrl.hdf5",'r')
	results_file = h5py.File(r"C:\Users\Ryan\Documents\data\e2_v1_sfcoherence_ctrl.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		e1_data = np.asarray(group['e1'])
		v1_data = np.asarray(group['v1'])
		data = []
		for v in range(e1_data.shape[2]):
			for d in range(v1_data.shape[2]):
				spikes = e1_data[:,:,v].T
				lfp = v1_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sf_coherence,data)
		pool.close()
		pool.join()
		result_data = async_result.get() ##this will return a list of lists, 
		##index 0 is the coherence, index 1 is the confidence intervals
		##separate them out:
		coherence = []
		confC = []
		for i in range(len(result_data)):
			coherence.append(result_data[i][0])
			confC.append(result_data[i][1])
		results_file.create_dataset(session,data = np.asarray(coherence))
		results_file.create_dataset(session+"_err", data=np.asarray(confC))
	g.close()
	results_file.close()

def save_direct_ds_sf_cohgram():	
	f = h5py.File("/home/lab/Documents/data/t1_triggered.hdf5",'r')
	sessions = f.keys()
	direct_data = []
	DS_data = []
	session_names = []
	for s in sessions:
		try:
			direct = None
			ds = None
			name = None
			direct1 = np.asarray(f[s]['e1_units'][:,:,:])
			direct2 = np.asarray(f[s]['e2_units'][:,:,:])
			direct = np.dstack((direct1,direct2)) 
			ds = np.asarray(f[s]['Str_lfp'][:,:,:])
			name = s
		except KeyError:
			pass
		if (direct != None and ds != None):
			if (direct.shape[0] > 2 and direct.shape[0] == ds.shape[0]): ##need at least 2 trials
				direct_data.append(direct)
				DS_data.append(ds)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_direct_ds_sf_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("direct", data=direct_data[i])
		gp.create_dataset("ds",data=DS_data[i])
	g.close()
	direct_data = None; DS_data = None
	g = h5py.File("/home/lab/Documents/data/paired_direct_ds_sf_t1.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/direct_ds_sfcohgrams_t1.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		direct_data = np.asarray(group['direct'])
		ds_data = np.asarray(group['ds'])
		data = []
		for v in range(direct_data.shape[2]):
			for d in range(ds_data.shape[2]):
				spikes = direct_data[:,:,v].T
				lfp = ds_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sfc,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def save_direct_ds_sf_cohgram_ctrl():	
	f = h5py.File("/home/lab/Documents/data/non_task_times.hdf5",'r')
	sessions = f.keys()
	direct_data = []
	DS_data = []
	session_names = []
	for s in sessions:
		try:
			direct = None
			ds = None
			name = None
			direct1 = np.asarray(f[s]['e1_units'][:,:,:])
			direct2 = np.asarray(f[s]['e2_units'][:,:,:])
			direct = np.dstack((direct1,direct2)) 
			ds = np.asarray(f[s]['Str_lfp'][:,:,:])
			name = s
		except KeyError:
			pass
		if (direct != None and ds != None):
			if (direct.shape[0] > 2 and direct.shape[0] == ds.shape[0]): ##need at least 2 trials
				direct_data.append(direct)
				DS_data.append(ds)
				session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File("/home/lab/Documents/data/paired_direct_ds_sf_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
		gp=g.create_group(name)
		gp.create_dataset("direct", data=direct_data[i])
		gp.create_dataset("ds",data=DS_data[i])
	g.close()
	direct_data = None; DS_data = None
	g = h5py.File("/home/lab/Documents/data/paired_direct_ds_sf_t1.hdf5",'r')
	results_file = h5py.File("/home/lab/Documents/data/direct_ds_sfcohgrams_t1.hdf5",'w-')
	##shape is trials x time x channels
	##let's just do a pairwise comparison of EVERYTHING
	##do this one sesssion at a time to not overload the memory
	for session in session_names:
		group = g[session]
		direct_data = np.asarray(group['direct'])
		ds_data = np.asarray(group['ds'])
		data = []
		for v in range(direct_data.shape[2]):
			for d in range(ds_data.shape[2]):
				spikes = direct_data[:,:,v].T
				lfp = ds_data[:,:,d].T
				data.append([spikes,lfp])
		pool = mp.Pool(processes=mp.cpu_count())
		async_result = pool.map_async(SFC.mp_sfc,data)
		pool.close()
		pool.join()
		cohgrams = async_result.get()
		results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()

def plot_jaws_late():
	f = h5py.File(r"C:\Users\Ryan\Documents\data\jaws_late_v_50_new.hdf5",'r')
	s50_mean = np.asarray(f['stim_50']).mean(axis=1)
	s50_sem = np.asarray(f['stim_50']).std(axis=1)/np.sqrt(f['stim_50'].shape[1])
	sLate_mean = np.asarray(f['stim_Late']).mean(axis=1)
	sLate_sem = np.asarray(f['stim_Late']).std(axis=1)/np.sqrt(f['stim_Late'].shape[1])
	f.close()
	xLate = np.arange(1,sLate_mean.size+1)
	x50 = np.arange(1,s50_mean.size+1)
	fig, ax = plt.subplots(1)
	ax.plot(x50,s50_mean,linewidth=2,color='k',linestyle='-',label='stim 50')
	ax.plot(x50,s50_mean,linewidth=2,color='r',linestyle='--')
	ax.plot(xLate[0:46],sLate_mean[0:46],linewidth=2,color='k',label='stim late')
	ax.plot(xLate[45:],sLate_mean[45:],linewidth=2,color='r')
	ax.fill_between(xLate[0:46],sLate_mean[0:46]-sLate_sem[0:46],sLate_mean[0:46]+
		sLate_sem[0:46],facecolor='k',color='k',alpha=0.5)
	ax.fill_between(xLate[45:],sLate_mean[45:]-sLate_sem[45:],sLate_mean[45:]+
		sLate_sem[45:],facecolor='r',color='r',alpha=0.5)
	z1 = []
	z2 = []
	for i in range (s50_mean.size/4):
		z1.append(True)
		z1.append(True)
		z1.append(False)
		z1.append(False)
		z2.append(False)
		z2.append(False)
		z2.append(True)
		z2.append(True)
	z1.append(True)
	z1.append(True)
	z2.append(False)
	z2.append(False)
	ax.fill_between(x50,s50_mean-s50_sem,s50_mean+s50_sem,facecolor='k',color='k',
		alpha=0.5,where=z1)
	ax.fill_between(x50,s50_mean-s50_sem,s50_mean+s50_sem,facecolor='r',color='r',
		alpha=0.5,where=z2)
	ax.set_xlabel("Trial number",fontsize=14)
	ax.set_ylabel("Percentage correct",fontsize=14)
	ax.set_title("Jaws D13 and D15",fontsize=16)


def plot_jaws_v_gfp_learning():
	f = h5py.File(r"C:\Users\Ryan\Documents\data\jaws_ctrl_learning.hdf5",'r')
	jaws = np.asarray(f['jaws'])
	gfp = np.asarray(f['gfp'])
	f.close()
	jaws_mean = jaws.mean(axis=1)
	jaws_sem = jaws.std(axis=1)/np.sqrt(jaws.shape[1])
	gfp_mean = np.nanmean(gfp,axis=1)
	gfp_sem = np.nanstd(gfp,axis=1)/np.sqrt(gfp.shape[1])
	x_axis = np.arange(1,15)
	fig, ax = plt.subplots(1)
	ax.errorbar(x_axis[0:12],jaws_mean[0:12],yerr=jaws_sem[0:12],linewidth=2,color='r')
	ax.errorbar(x_axis,gfp_mean,yerr=gfp_sem,linewidth=2,color='k')
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Percentage correct",fontsize=14)
	ax.set_title("Jaws vs GFP learning")
	ax.set_xlim(0,16)
	for animal in range(jaws.shape[1]):
		plt.plot(x_axis[0:12],jaws[0:12,animal],color='r',
			linewidth=1,alpha=0.5)
	for animal in range(gfp.shape[1]):
		plt.plot(x_axis,gfp[:,animal],color='k',linewidth=1,alpha=0.5)
	

	fig2, ax2 = plt.subplots(1)
	gfp_early = gfp[0:3,:].mean(axis=0)
	gfp_late = gfp[4:7,:].mean(axis=0)
	jaws_early = jaws[0:3,:].mean(axis=0)
	jaws_late = jaws[4:7,:].mean(axis=0)
	jaws_late_nostim = jaws[9:,:].mean(axis=0)
	means = [gfp_early.mean(),gfp_late.mean(),jaws_early.mean(),
			jaws_late.mean(),jaws_late_nostim.mean()]
	sems = [gfp_early.std()/2,gfp_late.std()/2,jaws_early.std()/2,
			jaws_late.std()/2,jaws_late_nostim.std()/2]
	idx = np.arange(0,5)-0.35
	width = 0.7
	bars = ax2.bar(idx, means, width, color = ['g','g','r','r','r'], yerr = sems, 
		ecolor = 'k', alpha = 1,linewidth=2)
	ax2.set_xticklabels(["gfp early","gfp late","jaws early","jaws late","jaws noStim"])
	ax2.set_ylabel("Percentage correct",fontsize=14)
	ax2.set_title("Learning across days- comparisons")
	t_gfp_el,gfp_early_late = stats.ttest_rel(gfp_early,gfp_late)
	t_jaws_el,jaws_early_late = stats.ttest_rel(jaws_early,jaws_late)
	t_jaws_lns,jaws_late_latenostim = stats.ttest_rel(jaws_late,jaws_late_nostim)
	gfp_jaws_late = stats.ttest_ind(gfp_late,jaws_late)[1],
	gfp_late_jaws_notstim = stats.ttest_ind(gfp_late,jaws_late_nostim)[1]
	
	##line plot version
	labels = ["gfp early","gfp late","jaws early","jaws late","jaws noStim"]
	fig, ax2 = plt.subplots(1)
	gfp = np.vstack((gfp_early,gfp_late))
	jaws = np.vstack((jaws_early,jaws_late,jaws_late_nostim))
	xg = np.array([0,1])
	xj = np.array([2,3,4])
	for i in range(gfp_early.shape[0]):
		ax2.plot(xg,gfp[:,i],color='g',linewidth=2,marker='o')
	for i in range(jaws_early.shape[0]):
		ax2.plot(xj,jaws[:,i],color='r',linewidth=2,marker='o')
	err_x = np.arange(0,5)
	yerr=sems
	xerr=np.ones(5)*0.25
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,5),labels)
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.35,4.35)
	ax2.set_ylabel("percentage correct", fontsize = 14)
	ax2.set_xlabel("Condition", fontsize = 14)
	#bars = ax2.bar(idx, means, width, color = ['g','g','r','r','r'], yerr = None, 
	#	alpha = 1,linewidth=2)

	print("mean gfp early = "+str(means[0]))
	print("mean gfp late = "+str(means[1]))
	print("P-val gfp early vs late= "+str(gfp_early_late))
	print("T-val gfp early vs late= "+str(t_gfp_el))
	print("mean jaws early = "+str(means[2]))
	print("mean jaws late = "+str(means[3]))
	print("mean jaws nostim = "+str(means[4]))
	print("P-val jaws early vs late= "+str(jaws_early_late))
	print("T-val jaws early vs late= "+str(t_jaws_el))
	print("P-val jaws late vs jaws no stim= "+str(jaws_late_latenostim))
	print("T-val jaws late vs jaws no stim= "+str(t_jaws_lns))
	print("P-val gfp late vs jaws late= "+str(gfp_jaws_late))
	print("P-val gfp late vs jaws no stim= "+str(gfp_late_jaws_notstim))


def plot_late_jaws_manips():
	f = h5py.File(r"C:\Users\Ryan\Documents\data\jaws_task_manips_bars.hdf5",'r')
	late_nostim = np.asarray(f['late_nostim'])
	late_stim = np.asarray(f['late_stim'])
	no_stim = np.asarray(f['no_stim'])
	stim_50 = np.asarray(f['stim_50'])
	f.close()
	means1 = [no_stim.mean(),stim_50.mean(),late_nostim.mean()]
	sems1 = [no_stim.std()/2,stim_50.std()/2,late_nostim.std()/2]
	means2 = [late_nostim.mean(),late_stim.mean()]
	sems2 = [late_nostim.std()/2,late_stim.std()/2]
	fig1, ax1 = plt.subplots(1)
	fig1, ax2 = plt.subplots(1)
	
	idx1 = np.arange(1,4)
	width = 0.7
	bars1 = ax1.bar(idx1, means1, width, color = ['k','r','k'], yerr = sems1, 
		ecolor = 'grey', alpha = 1,linewidth=2)
	ax1.set_xticklabels(["LED off","LED 50","LED off",])
	ax1.set_ylabel("Percentage correct",fontsize=14)
	ax1.set_title("Learning v Performance",fontsize=16)
	ax1.set_xlim(-.15,4.35)

	idx2 = np.arange(1,3)
	width = 0.7
	bars2 = ax2.bar(idx2, means2, width, color = ['k','r'], yerr = sems2, 
		ecolor = 'grey', alpha = 1,linewidth=2)
	ax2.set_xticklabels(["Train LED off","Test LED on"])
	ax2.set_ylabel("Percentage correct",fontsize=14)
	ax2.set_title("Learning v Performance",fontsize=16)
	ax2.set_xlim(-.15,3.35)
	#ax2.scatter(np.ones(4)*1.35,late_nostim,color='grey',s=20,zorder=2)
	#ax2.scatter(np.ones(4)*2.35,late_stim,color='grey',s=20,zorder=2)
	t_perf,p_perf = stats.ttest_rel(late_nostim,late_stim)
	t_no_v_50,p_no_v_50 = stats.ttest_rel(no_stim,stim_50)
	t_50_v_no,p_50_v_no = stats.ttest_rel(stim_50,late_nostim)
	t_no_v_no,p_no_v_no = stats.ttest_rel(no_stim,late_nostim)
	
	labels1 = ["LED off","LED 50","LED off"]
	rew = np.vstack((no_stim,stim_50,late_nostim))
	fig, ax2 = plt.subplots(1)
	xr = np.array([0,1,2])
	err_x = np.array([0,1,2])
	yerr = sems1
	xerr = np.ones(3)*0.25
	for i in range(rew.shape[1]):
		ax2.plot(xr,rew[:,i],color='k',linewidth=2,marker='o')
	ax2.errorbar(err_x,means1,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),labels1)
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.3,2.3)
	ax2.set_ylim(-0.1,0.8)

	labels2 = [" train LED off","test LED on"]
	rew = np.vstack((late_nostim,late_stim))
	fig, ax2 = plt.subplots(1)
	xr = np.array([0,1])
	err_x = np.array([0,1])
	yerr = sems2
	xerr = np.ones(2)*0.25
	for i in range(rew.shape[1]):
		ax2.plot(xr,rew[:,i],color='k',linewidth=2,marker='o')
	ax2.errorbar(err_x,means2,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),labels2)
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.3,1.3)
	ax2.set_ylim(-0,1)

	print("mean LED off 1 = "+str(means1[0]))
	print("mean LED 50 = "+str(means1[1]))
	print("mean LED off 2 = "+str(means1[2]))
	print("mean train LED off = "+str(means2[0]))
	print("mean test LED on = "+str(means2[1]))
	print("pval train LED off v train LED on= "+str(p_perf))
	print("tval train LED off v train LED on= "+str(t_perf))
	print("pval LED off 1 vs Stim50 = "+str(p_no_v_50))
	print("tval LED off 1 v stim_50= "+str(t_no_v_50))
	print("pval stim_50 v LED off 2= "+str(p_50_v_no))
	print("tval stim_50 v LED off 2= "+str(t_50_v_no))
	print("pval LED off 1 v LED off 2= "+str(p_no_v_no))
	print("tval LED off v LED off 2= "+str(t_no_v_no))



########## NEW STUFF FOR NATURE NEURO ##################

##plots within session data by animal
def plot_within_session_by_animal():
	source_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_learning_event_data2.hdf5"
	f = h5py.File(source_file, 'r+') 
	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13", "R8"]
	##run through each animal and take the mean over all sessions
	longest = 0 #to record the length of the longest array
	rewarded_means = []
	unrewarded_means = []
	for animal in animal_list:
		##get the rewarded targets
		rewarded = np.asarray(f[animal]['correct_within_sessions']).mean(axis=0)
		rewarded_means.append(rewarded)
		#3unrewarded
		unrewarded = np.asarray(f[animal]['t2_within_sessions']).mean(axis=0)
		unrewarded_means.append(unrewarded)
		##keep track of longest sessions
		l = max(longest,rewarded.shape[0],unrewarded.shape[0])
		if l > longest:
			longest = l
	f.close()
	##standardize the lengths of all arrays
	##append some zeros on to the other arrays to make them all the same shape
	for idx in range(len(rewarded_means)):
		difference = longest - rewarded_means[idx].shape[0]
		if difference > 0:
			rewarded_means[idx] = np.hstack((rewarded_means[idx],np.zeros(difference)))
	for idx in range(len(unrewarded_means)):
		difference = longest - unrewarded_means[idx].shape[0]
		if difference > 0:
			unrewarded_means[idx] = np.hstack((unrewarded_means[idx],np.zeros(difference)))
	rewarded_means = np.asarray(rewarded_means)
	unrewarded_means = np.asarray(unrewarded_means)
	##now let's get the mean performance for the first 5 min and last 5 min
	early_rewarded = rewarded_means[:,0:5].mean(axis=1)
	late_rewarded = rewarded_means[:,60:65].mean(axis=1)
	early_unrewarded = unrewarded_means[:,0:5].mean(axis=1)
	late_unrewarded = unrewarded_means[:,60:65].mean(axis=1)
	##do dem stats
	#data = [early_rewarded,late_rewarded,early_unrewarded,late_unrewarded]
	##make a pandas dataframe to do the swarmplot
	data = collections.OrderedDict()
	data["Early rew."]=early_rewarded
	data["Late rew."]=late_rewarded
	data["Early unrew."]=early_unrewarded
	data["Late unrew."]=late_unrewarded
	df = pd.DataFrame(data=data,index=animal_list)
	##some stats for errorbars
	means = np.array([early_rewarded.mean(), late_rewarded.mean(),early_unrewarded.mean(),late_unrewarded.mean()])
	stds = np.array([early_rewarded.std(), late_rewarded.std(),early_unrewarded.std(),late_unrewarded.std()])
	yerr = stds/np.sqrt(len(animal_list))
	##xerr is just used to plot the mean
	xerr = np.ones(4)*0.1
	err_x = np.arange(0,4) ##x-vals for errorbars
	##turn off grid
	sns.set_style("whitegrid", {'axes.grid' : False})
	ax = sns.stripplot(data=df,jitter=True)
	ax.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	for ticklabel in ax.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax.get_yticklabels():
		ticklabel.set_fontsize(14)
	plt.draw()
	##plot it another way
	fig, ax2 = plt.subplots(1)
	rew = np.vstack((early_rewarded,late_rewarded))
	unrew = np.vstack((early_unrewarded,late_unrewarded))
	xr = np.array([0,1])
	xu = np.array([2,3])
	for i in range(len(animal_list)):
		ax2.plot(xr,rew[:,i],color='r',linewidth=2,marker='o',alpha=0.5)
		ax2.plot(xu,unrew[:,i],color='b',linewidth=2,marker='o',alpha=0.5)
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,4),data.keys())
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.1,3.1)
	t_rewarded,pval_rewarded = stats.ttest_rel(early_rewarded, late_rewarded)
	t_unrewarded,pval_unrewarded = stats.ttest_rel(early_unrewarded,late_unrewarded)
	t_early,pval_early = stats.ttest_ind(early_rewarded,early_unrewarded)
	t_late,pval_late = stats.ttest_ind(late_rewarded,late_unrewarded)
	# fig, ax = plt.subplots()
	# boxes = ax.boxplot(data)
	# ax.set_ylim(-0.1,1.1)
	# # #ax.set_xlim(-0.01, 1.6)
	# # ax.set_xticks(idx+0.15)
	# ax.set_xticklabels(("Rew. Early", "Rew. Late",'Un. Early','Un. Late'))
	# ax.set_ylabel("Percent of events", fontsize = 14)
	# #ax.text(1, 0.75, "p = " + str(pval), fontsize = 12)
	# fig.suptitle("Within session Performance", fontsize = 16)
	print("pval rewarded e-l = "+str(pval_rewarded))
	print("pval unrewarded e-l = "+str(pval_unrewarded))
	print("pval early r-u = "+str(pval_early))
	print("pval late r-u = "+str(pval_late))
	print("tval rewarded e-l = "+str(t_rewarded))
	print("tval unrewarded e-l = "+str(t_unrewarded))
	print("tval early r-u = "+str(t_early))
	print("tval late r-u = "+str(t_late))
	print("mean rew early = "+str(means[0]))
	print("sem rew early = "+str(yerr[0]))
	print("mean rew late = "+str(means[1]))
	print("sem rew late = "+str(yerr[1]))
	print("mean unrew early = "+str(means[2]))
	print("sem unrew early = "+str(yerr[2]))
	print("mean unrew late = "+str(means[3]))
	print("sem unrew late = "+str(yerr[3]))


def plot_within_session_light_dark2():
	source_file = "/Volumes/Untitled/Ryan/V1_BMI/Neuron/R7_thru_V13_learning_event_data2.hdf5"
	f = h5py.File(source_file, 'r')
	light_animals = ["R13", "R11", "R7", "R8"]
	dark_animals = ["V01", "V02", "V03", "V04", "V05", "V11", "V13"]
	light_data = []
	dark_data = []
	longest = 0
	for l in light_animals:
		dat = np.asarray(f[l]['correct_within_light_only']).mean(axis=0)
		light_data.append(dat)
		t = max(longest,dat.shape[0])
		if t > longest:
			longest = t
	for d in dark_animals:
		dat = np.asarray(f[d]['correct_within_dark_only']).mean(axis=0)
		dark_data.append(dat)
		t = max(longest,dat.shape[0])
		if t > longest:
			longest = t
	f.close()
	##standardize the lengths of all arrays
	##append some zeros on to the other arrays to make them all the same shape
	for idx in range(len(light_data)):
		difference = longest - light_data[idx].shape[0]
		if difference > 0:
			light_data[idx] = np.hstack((light_data[idx],np.zeros(difference)))
	for idx in range(len(dark_data)):
		difference = longest -dark_data[idx].shape[0]
		if difference > 0:
			dark_data[idx] = np.hstack((dark_data[idx],np.zeros(difference)))
	dark_data = np.asarray(dark_data)
	light_data = np.asarray(light_data)
	light_mean = np.nanmean(light_data, axis = 0)
	light_std = np.nanstd(light_data, axis = 0)
	light_sem = light_std/np.sqrt(light_data.shape[0])
	dark_mean = np.nanmean(dark_data, axis = 0)
	dark_std = np.nanstd(dark_data, axis = 0)
	dark_sem = dark_std/np.sqrt(dark_data.shape[0])
	x_axis = np.linspace(0,115,light_mean.shape[0])
	fig, ax = plt.subplots()
	ax.plot(x_axis, light_mean, linewidth = 3, color = 'yellow', label = "train light")
	plt.fill_between(x_axis, light_mean-light_sem, light_mean+light_sem, 
		alpha = 0.5, facecolor = 'yellow')
	ax.plot(x_axis[0:dark_mean.size], dark_mean, linewidth = 3, color = 'k', label = "train dark")
	plt.fill_between(x_axis[0:dark_mean.size], dark_mean-dark_sem, dark_mean+dark_sem, 
		alpha = 0.5, facecolor = 'k')
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel("Time in session, mins", fontsize = 16)
	ax.set_ylabel("Percent rewarded", fontsize = 16)
	fig.suptitle("Light vs dark training", fontsize = 18)
	ax.set_xlim((0,50))
	ax.fill_between(x_axis, .25,.39, alpha = 0.1, facecolor = 'cyan')
	ax.legend()
	##now do a test of differences and plot the bar graph
	light_early = light_data[:,0:8].mean(axis=1)
	light_late = light_data[:,60:65].mean(axis=1)
	dark_early = dark_data[:6,0:8].mean(axis=1)
	dark_late = np.array([dark_data[0,55:60],dark_data[1,65:70],dark_data[2,65:70],
		dark_data[3,98:103],dark_data[4,104:109],dark_data[5,65:70]]).mean(axis=1)
	t_light,p_val_light = stats.ttest_rel(light_early,light_late)
	t_dark,p_val_dark = stats.ttest_rel(dark_early,dark_late)
	t_light_dark,p_val_light_dark = stats.ttest_ind(dark_late,light_late)
	#plot the bars
	means = [light_early.mean(),light_late.mean(),dark_early.mean(),dark_late.mean()]
	sem = [light_early.std()/np.sqrt(light_early.size),light_late.std()/np.sqrt(light_late.size),
			dark_early.std()/np.sqrt(dark_early.size),dark_late.std()/np.sqrt(dark_late.size)]
	labels = ['light_early','light_late','dark_early','dark_late']
	fig, ax2 = plt.subplots(1)
	rew = np.vstack((light_early,light_late))
	unrew = np.vstack((dark_early,dark_late))
	xr = np.array([0,1])
	xu = np.array([2,3])
	for i in range(light_early.shape[0]):
		ax2.plot(xr,rew[:,i],color='y',linewidth=2,marker='o',alpha=0.8)
	for i in range(dark_early.shape[0]):
		ax2.plot(xu,unrew[:,i],color='k',linewidth=2,marker='o',alpha=0.5)
	err_x = np.arange(0,4)
	yerr=sem
	xerr=np.ones(4)*0.25
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,4),labels)
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.35,3.35)
	ax2.set_ylabel("percentage correct", fontsize = 14)
	ax2.set_xlabel("Condition", fontsize = 14)
	print("pval light early-late= "+str(p_val_light))
	print("pval dark early-late= "+str(p_val_dark))
	print("pval light-dark-late= "+str(p_val_light_dark))
	print("tval light early-late= "+str(t_light))
	print("tval dark early-late= "+str(t_dark))
	print("tval light-dark-late= "+str(t_light_dark))
	print("mean light early = "+str(means[0]))
	print("sem light early = "+str(sem[0]))
	print("mean light late = "+str(means[1]))
	print("sem light late = "+str(sem[1]))
	print("mean dark early = "+str(means[2]))
	print("sem dark early = "+str(sem[2]))
	print("mean dark late = "+str(means[3]))
	print("sem dark late = "+str(sem[3]))


def get_mean_frs2():
	path = r"C:\Users\Ryan\Documents\data\R7_thru_V13_all_data.hdf5"
	animal_list = ["R13", "R11", "V02", "V03", "V04", "V05", "V11", "V13", "R7", "R8"]
	session_dict_late = {
	"R13":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"R11":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V02":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V03":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx"],
	"V04":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx"],
	"V05":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D08.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V11":["BMI_D05.plx", "BMI_D06.plx", "BMI_D07.plx", "BMI_D11.plx", "BMI_D09.plx", "BMI_D10.plx"],
	"V13":["BMI_D05.plx", "BMI_D06.plx", "BMI_D04.plx"],
	"R7":["BMI_D05.plx", "BMI_D06.plx"],
	"R8":["BMI_D05.plx", "BMI_D06.plx"]
	}

	e1_master = []
	e2_master = []
	ind_master = []

	longest2 = 0
	for animal in animal_list:
		all_e1 = []
		all_e2 = []
		all_ind = []
		longest = 0
		for session in session_dict_late[animal]:
			e1, e2, ind = ds.get_ensemble_arrays(path, animal = animal, session = session)
			if e1.shape[0] > longest:
				longest = e1.shape[0]
			for n in range(e1.shape[1]):
				all_e1.append(ss.windowRate(e1[:,n], [500,100]))
			for n in range(e2.shape[1]):
				all_e2.append(ss.windowRate(e2[:,n], [500,100]))
			for n in range(ind.shape[1]):
				all_ind.append(ss.windowRate(ind[:,n], [500,100]))
		for i in range(len(all_e1)):	
			if all_e1[i].shape[0] < longest:
				add = np.empty((longest-all_e1[i].shape[0]))
				add[:] = np.nan
				all_e1[i] = np.hstack((all_e1[i], add))
		for i in range(len(all_e2)):	
			if all_e2[i].shape[0] < longest:
				add = np.empty((longest-all_e2[i].shape[0]))
				add[:] = np.nan
				all_e2[i] = np.hstack((all_e2[i], add))
		for i in range(len(all_ind)):	
			if all_ind[i].shape[0] < longest:
				add = np.empty((longest-all_ind[i].shape[0]))
				add[:] = np.nan
				all_ind[i] = np.hstack((all_ind[i], add))
		all_e1 = np.nanmean(np.asarray(all_e1),axis=0)
		all_e2 = np.nanmean(np.asarray(all_e2),axis=0)
		all_ind = np.nanmean(np.asarray(all_ind),axis=0)
		e1_master.append(all_e1)
		e2_master.append(all_e2)
		ind_master.append(all_ind)
		if longest > longest2:
			longest2 = longest

	for i in range(len(e1_master)):	
		if e1_master[i].shape[0] < longest2:
			add = np.empty((longest2-e1_master[i].shape[0]))
			add[:] = np.nan
			e1_master[i] = np.hstack((e1_master[i], add))
	e1_master = np.asarray(e1_master)
	
	for i in range(len(e2_master)):	
		if e2_master[i].shape[0] < longest2:
			add = np.empty((longest2-e2_master[i].shape[0]))
			add[:] = np.nan
			e2_master[i] = np.hstack((e2_master[i], add))
	e2_master = np.asarray(e1_master)
	
	ind_master[5] = ind_master[6]
	for i in range(len(ind_master)):	
		if ind_master[i].shape[0] < longest2:
			add = np.empty((longest2-ind_master[i].shape[0]))
			add[:] = np.nan
			ind_master[i] = np.hstack((ind_master[i], add))
	ind_master = np.asarray(ind_master)

	


	e1_early = e1_master[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	e1_late = e1_master[:,45*60*10:50*60*10].mean(axis = 1)/(5.0*60)

	e2_early = e2_master[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	e2_late = e2_master[:,50*60*10:55*60*10].mean(axis = 1)/(5.0*60)

	ind_early = ind_master[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	ind_late = ind_master[:,50*60*10:55*60*10].mean(axis = 1)/(5.0*60)

	f_out = h5py.File(r"C:\Users\Ryan\Documents\data\R7_thru_V13_spike_rates_by_animal.hdf5",'w-')
	
	f_out.create_dataset("all_e1", data = all_e1)
	f_out.create_dataset("all_e2", data = all_e2)
	f_out.create_dataset("all_ind", data = all_ind)

	f_out.create_dataset("e1_early", data = e1_early)
	f_out.create_dataset("e2_early", data = e2_early)
	f_out.create_dataset("ind_early", data = ind_early)

	f_out.create_dataset("e1_late", data = e1_late)
	f_out.create_dataset("e2_late", data = e2_late)
	f_out.create_dataset("ind_late", data = ind_late)

	f_out.close()

def plot_fr_data2():
	f = h5py.File(r"C:\Users\Ryan\Documents\data\R7_thru_V13_spike_rates_by_animal.hdf5", 'r')

	e1_early = np.asarray(f['e1_early'])*300
	e1_late = np.asarray(f['e1_late'])*300
	e2_early = np.asarray(f['e2_early'])*300
	e2_late = np.asarray(f['e2_late'])*300
	ind_early = np.asarray(f['ind_early'])*300
	ind_late = np.asarray(f['ind_late'])*300
	f.close()

	labels = np.array(['e1_early', 'e1_late', 'e2_early', 'e2_late', 'ind_early', 'ind_late'])
	means = np.array([np.nanmean(e1_early), np.nanmean(e1_late), np.nanmean(e2_early), 
		np.nanmean(e2_late),np.nanmean(ind_early), np.nanmean(ind_late)])
	sem = np.array([np.nanstd(e1_early)/np.sqrt(94), np.nanstd(e1_late)/np.sqrt(94), 
		np.nanstd(e2_early)/np.sqrt(92), np.nanstd(e2_late)/np.sqrt(92),
		np.nanstd(ind_early)/np.sqrt(182), np.nanstd(ind_late)/np.sqrt(182)])

	t_val_e1,p_val_e1 = stats.ttest_rel(e1_early, e1_late, nan_policy='omit')
	t_val_e2,p_val_e2 = stats.ttest_rel(e2_early, e2_late, nan_policy='omit')
	t_val_ind,p_val_ind = stats.ttest_rel(ind_early, ind_late, nan_policy='omit')

	t_val_e1_e2_early,p_val_e1_e2_early = stats.ttest_ind(e1_early, e2_early, nan_policy='omit')
	t_val_e1_e2_late,p_val_e1_e2_late = stats.ttest_ind(e1_late, e2_late, nan_policy='omit')
	t_val_e1_ind_early,p_val_e1_ind_early = stats.ttest_ind(e1_early, ind_early, nan_policy='omit')
	p_val_e1_ind_late = stats.ttest_ind(e1_late, ind_late, nan_policy='omit')[1]
	p_val_e2_ind_early = stats.ttest_ind(e2_early, ind_early, nan_policy='omit')[1]
	p_val_e2_ind_late = stats.ttest_ind(e2_late, ind_late, nan_policy='omit')[1]


	idx = np.arange(6)-0.45
	width = 1.0
	fig, ax2 = plt.subplots()
	bars = ax2.bar(idx, means, yerr = sem, ecolor = 'k')
	#ax.set_ylim(0,0.9)
	#ax.set_xlim(-0.5, 3.5)
	# ax.set_xticks(idx+0.5)
	# ax.set_xticklabels(labels)
	# # for i in range(data.shape[0]):
	# # 	plt.plot((idx+0.5), data[i,:], alpha = 0.5, color = np.random.rand(3,), marker = 'o', linewidth = 2)
	# ax.set_ylabel("firing rate", fontsize = 14)
	# ax.set_xlabel("Condition", fontsize = 14)

	fig, ax2 = plt.subplots(1)
	e1 = np.vstack((e1_early,e1_late))
	e2 = np.vstack((e2_early,e2_late))
	ind = np.vstack((ind_early,ind_late))
	x1 = np.array([0,1])
	x2 = np.array([2,3])
	xi = np.array([4,5])
	for i in range(e1_early.shape[0]):
		ax2.plot(x1,e1[:,i],color='g',linewidth=2,marker='o',alpha=0.8)
	for i in range(e2_early.shape[0]):
		ax2.plot(x2,e2[:,i],color='b',linewidth=2,marker='o',alpha=0.5)
	for i in range(ind_early.shape[0]):
		ax2.plot(xi,ind[:,i],color='k',linewidth=2,marker='o',alpha=0.5)
	err_x = np.arange(0,6)
	yerr=sem
	xerr=np.ones(6)*0.25
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,6),labels)
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.5,5.35)
	ax2.set_ylabel("Fr, Hz", fontsize = 14)
	ax2.set_xlabel("Condition", fontsize = 14)
	print("pval e1 early-late= "+str(p_val_e1))
	print("tval e1 early-late= "+str(t_val_e1))
	print("pval e2 early-late= "+str(p_val_e2))
	print("tval e2 early-late= "+str(t_val_e2))
	print("pval ind early-late= "+str(p_val_ind))
	print("tval ind early-late= "+str(t_val_ind))
	print("mean e1 early = "+str(means[0]))
	print("mean e1 late = "+str(means[1]))
	print("mean e2 early = "+str(means[2]))
	print("mean e2 late = "+str(means[3]))
	print("mean ind early = "+str(means[4]))
	print("mean ind late = "+str(means[5]))


def plot_within_session_ratios():
	source_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_learning_event_data2.hdf5"
	f = h5py.File(source_file, 'r+') 
	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13","R8"]
	##run through each animal and take the mean over all sessions
	longest = 0 #to record the length of the longest array
	##start with early sessions
	rewarded_means = []
	unrewarded_means = []
	for animal in animal_list:
		##get the rewarded targets
		rewarded = np.asarray(f[animal]['correct_within_sessions'][0:3,:]).mean(axis=0)
		rewarded_means.append(rewarded)
		#3unrewarded
		unrewarded = np.asarray(f[animal]['t2_within_sessions'][0:3,:]).mean(axis=0)
		unrewarded_means.append(unrewarded)
		##keep track of longest sessions
		l = max(longest,rewarded.shape[0],unrewarded.shape[0])
		if l > longest:
			longest = l
	##standardize the lengths of all arrays
	##append some zeros on to the other arrays to make them all the same shape
	for idx in range(len(rewarded_means)):
		difference = longest - rewarded_means[idx].shape[0]
		if difference > 0:
			rewarded_means[idx] = np.hstack((rewarded_means[idx],np.zeros(difference)))
	for idx in range(len(unrewarded_means)):
		difference = longest - unrewarded_means[idx].shape[0]
		if difference > 0:
			unrewarded_means[idx] = np.hstack((unrewarded_means[idx],np.zeros(difference)))
	rewarded_means = np.asarray(rewarded_means)
	unrewarded_means = np.asarray(unrewarded_means)
	unrewarded_means[-1] = unrewarded_means[-1]+0.33 ##correction for earlier code that didn't count T2
	##now let's get the performance RATIOS 
	early_rewarded = rewarded_means[:,0:5].mean(axis=1)
	late_rewarded = rewarded_means[:,60:65].mean(axis=1)
	early_unrewarded = unrewarded_means[:,0:5].mean(axis=1)
	late_unrewarded = unrewarded_means[:,60:65].mean(axis=1)
	early_early_ratio = early_rewarded/early_unrewarded
	early_late_ratio = late_rewarded/late_unrewarded
	
	##now for late sessions
	longest = 0 #to record the length of the longest array
	##start with early sessions
	rewarded_means = []
	unrewarded_means = []
	for animal in animal_list:
		##get the rewarded targets
		rewarded = np.asarray(f[animal]['correct_within_sessions'][-3:,:]).mean(axis=0)
		rewarded_means.append(rewarded)
		#3unrewarded
		unrewarded = np.asarray(f[animal]['t2_within_sessions'][-3:,:]).mean(axis=0)
		unrewarded_means.append(unrewarded)
		##keep track of longest sessions
		l = max(longest,rewarded.shape[0],unrewarded.shape[0])
		if l > longest:
			longest = l
	##standardize the lengths of all arrays
	##append some zeros on to the other arrays to make them all the same shape
	for idx in range(len(rewarded_means)):
		difference = longest - rewarded_means[idx].shape[0]
		if difference > 0:
			rewarded_means[idx] = np.hstack((rewarded_means[idx],np.zeros(difference)))
	for idx in range(len(unrewarded_means)):
		difference = longest - unrewarded_means[idx].shape[0]
		if difference > 0:
			unrewarded_means[idx] = np.hstack((unrewarded_means[idx],np.zeros(difference)))
	rewarded_means = np.asarray(rewarded_means)
	unrewarded_means = np.asarray(unrewarded_means)
	unrewarded_means[-1] = unrewarded_means[-1]+0.33 ##correction for earlier code that didn't count T2
	##now let's get the performance RATIOS 
	early_rewarded = np.array([
		rewarded_means[0,0:10].mean(),
		rewarded_means[1,0:10].mean(),
		rewarded_means[2,0:10].mean(),
		rewarded_means[3,0:10].mean(),
		rewarded_means[4,0:10].mean(),
		rewarded_means[5,0:5].mean(),
		rewarded_means[6,0:10].mean(),
		rewarded_means[7,0:5].mean(),
		rewarded_means[8,0:5].mean(),
		rewarded_means[9,0:10].mean()])
	late_rewarded = np.array([
		rewarded_means[0,80:90].mean(),
		rewarded_means[1,70:80].mean(),
		rewarded_means[2,50:60].mean(),
		rewarded_means[3,70:80].mean(),
		rewarded_means[4,55:65].mean(),
		rewarded_means[5,65:70].mean(),
		rewarded_means[6,80:90].mean(),
		rewarded_means[7,65:70].mean(),
		rewarded_means[8,70:80].mean(),
		rewarded_means[9,60:70].mean()])
	early_unrewarded = np.array([
		unrewarded_means[0,0:10].mean(),
		unrewarded_means[1,0:10].mean(),
		unrewarded_means[2,0:10].mean(),
		unrewarded_means[3,0:10].mean(),
		unrewarded_means[4,0:10].mean(),
		unrewarded_means[5,0:5].mean(),
		unrewarded_means[6,0:10].mean(),
		unrewarded_means[7,0:5].mean(),
		unrewarded_means[8,0:5].mean(),
		unrewarded_means[9,0:10].mean()])
	late_unrewarded = np.array([
		unrewarded_means[0,80:90].mean(),
		unrewarded_means[1,70:80].mean(),
		unrewarded_means[2,50:60].mean(),
		unrewarded_means[3,70:80].mean(),
		unrewarded_means[4,55:65].mean(),
		unrewarded_means[5,65:70].mean(),
		unrewarded_means[6,80:90].mean(),
		unrewarded_means[7,65:70].mean(),
		unrewarded_means[8,70:80].mean(),
		unrewarded_means[9,60:70].mean()])
	late_early_ratio = early_rewarded/early_unrewarded
	late_late_ratio = late_rewarded/late_unrewarded
	f.close()

	##do dem stats
	#data = [early_rewarded,late_rewarded,early_unrewarded,late_unrewarded]
	##make a pandas dataframe to do the swarmplot
	data = collections.OrderedDict()
	data["Early-early."]=early_early_ratio
	data["Early-late"]=early_late_ratio
	data["Late-early."]=late_early_ratio
	data["Late-late."]=late_late_ratio
	df = pd.DataFrame(data=data,index=animal_list)
	##some stats for errorbars
	means = np.array([early_early_ratio.mean(), early_late_ratio.mean(),
		late_early_ratio.mean(),late_late_ratio.mean()])
	stds = np.array([early_early_ratio.std(), early_late_ratio.std(),
		late_early_ratio.std(),late_late_ratio.std()])
	yerr = stds/np.sqrt(len(animal_list))
	##xerr is just used to plot the mean
	xerr = np.ones(4)*0.1
	err_x = np.arange(0,4) ##x-vals for errorbars
	##turn off grid
	sns.set_style("whitegrid", {'axes.grid' : False})
	ax = sns.stripplot(data=df,jitter=True)
	ax.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	for ticklabel in ax.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax.get_yticklabels():
		ticklabel.set_fontsize(14)
	plt.draw()
	##plot it another way
	fig, ax2 = plt.subplots(1)
	rew = np.vstack((early_early_ratio,early_late_ratio))
	unrew = np.vstack((late_early_ratio,late_late_ratio))
	xr = np.array([0,1])
	xu = np.array([2,3])
	for i in range(len(animal_list)):
		ax2.plot(xr,rew[:,i],color='cyan',linewidth=2,marker='o',alpha=0.5)
		ax2.plot(xu,unrew[:,i],color='b',linewidth=2,marker='o',alpha=0.5)
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,4),data.keys())
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.1,3.1)
	tval_early,pval_early = stats.ttest_rel(early_early_ratio, early_late_ratio)
	tval_late,pval_late = stats.ttest_rel(late_early_ratio,late_late_ratio)
	# t_early,pval_early = stats.ttest_ind(early_rewarded,early_unrewarded)
	# t_late,pval_late = stats.ttest_ind(late_rewarded,late_unrewarded)
	# fig, ax = plt.subplots()
	# boxes = ax.boxplot(data)
	# ax.set_ylim(-0.1,1.1)
	# # #ax.set_xlim(-0.01, 1.6)
	# # ax.set_xticks(idx+0.15)
	# ax.set_xticklabels(("Rew. Early", "Rew. Late",'Un. Early','Un. Late'))
	# ax.set_ylabel("Percent of events", fontsize = 14)
	# #ax.text(1, 0.75, "p = " + str(pval), fontsize = 12)
	# fig.suptitle("Within session Performance", fontsize = 16)
	print("pval early e-l = "+str(pval_early))
	print("tval early e-l = "+str(tval_early))
	print("pval late e-l = "+str(pval_late))
	print("tval late e-l = "+str(tval_late))
	print("mean early-early = "+str(means[0]))
	print("mean early-late = "+str(means[1]))
	print("mean late-early = "+str(means[2]))
	print("mean late-late = "+str(means[3]))


#######################################################
#########################################################
##REBUTTAL
######################################################

"""
A function to get the relationship between mean firing rates
in E1 VS performance, and mean firing ratets in E2 vs performance
"""
def get_frs_vs_performance():
	##the main source file
	root_dir = r"D:\Ryan\V1_BMI"
	save_path = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\all_frs_vs_performance.hdf5"
	animal_list = ru.animals.keys()
	##the data to save
	##go session by session for each animal and save the mean fr of e1 units and e2
	##units over the whole session, as well as the mean performance over the whole session
	for animal in animal_list:
		session_list = ru.animals[animal][1].keys() ##the names of all the sessions for this animal
		for session in session_list:
			##figure out if this data is already saved
			f_out = h5py.File(save_path,'a')
			try:
				session_exists = f_out[animal][session]
				print(animal+" "+session+" data exists; moving on")
				f_out.close()
			except KeyError:
				session_e1 = []
				session_p1 = []
				session_e2 = []
				session_p2 = []
				plxfile = os.path.join(root_dir,animal,session)
				print("working on "+animal+" "+session)
				e1_names = ru.animals[animal][1][session]['units']['e1_units']
				e2_names = ru.animals[animal][1][session]['units']['e2_units']
				try:
					t1_id = ru.animals[animal][1][session]['events']['t1'][0]
				except KeyError:
					t1_id = None
				try:
					t2_id = ru.animals[animal][1][session]['events']['t2'][0]
				except KeyError:
					t2_id = None
				try:
					miss_id = ru.animals[animal][1][session]['events']['miss'][0]
				except KeyError:
					miss_id = None		
				##open the file
				if animal.startswith('m'):
					adrange = range(100,200)
				else:
					adrange = range(1,97)
				raw_data = plxread.import_file(plxfile,AD_channels=adrange,save_wf=False,
						import_unsorted=False,verbose=False)
				try:
					lfp_id = ru.animals[animal][1][session]['lfp']['V1_lfp'][0]
				except KeyError:
					lfp_id = ru.animals[animal][1][session]['lfp']['Str_lfp'][0]
				duration = raw_data[lfp_id+'_ts'].max()
				##now get the performance for this session
				try:
					n_t1 = float(raw_data[t1_id].size)
				except KeyError:
					n_t1 = 0
				try: 
					n_t2 = float(raw_data[t2_id].size)
				except KeyError:
					n_t2 = 0
				try:
					n_miss = float(raw_data[miss_id].size)
				except KeyError:
					n_miss = 0
				try:
					score = n_t1/(n_t1+n_t2+n_miss)
				except ZeroDivisionError:
					score = np.random.ranf()
				for name in e1_names:
					try:
						n_spikes = raw_data[name].size ##the total spikes in the whole session
						if n_spikes/duration < 70:
							session_e1.append(n_spikes/duration)
							session_p1.append(score)
					except KeyError:
						print(name+" not in this file; skipping")
				for name in e2_names:
					try:
						n_spikes =  raw_data[name].size
						if n_spikes/duration < 70:
							session_e2.append(n_spikes/duration)
							session_p2.append(score)
					except KeyError:
						print(name+" not in this file; skipping")
				f_out = h5py.File(save_path,'a')
				try:
					a_group = f_out[animal]
				except KeyError:
					a_group = f_out.create_group(animal)
				s_group = a_group.create_group(session)
				s_group.create_dataset("e1_rates",data=np.asarray(session_e1))
				s_group.create_dataset("e2_rates",data=np.asarray(session_e2))
				s_group.create_dataset("p_correct_e1",data=np.asarray(session_p1))
				s_group.create_dataset("p_correct_e2",data=np.asarray(session_p2))
				f_out.close()
	##reopen the file, and extract all the data
	f = h5py.File(save_path,'r')
	e1_rates = []
	e2_rates = []
	p_correct1 = []
	p_correct2 = []
	for animal in f.keys():
		for session in f[animal].keys():
			e1_rates.append(np.nanmean(np.asarray(f[animal][session]['e1_rates'])))
			e2_rates.append(np.nanmean(np.asarray(f[animal][session]['e2_rates'])))
			p_correct1.append(np.asarray(f[animal][session]['p_correct_e1']))
			p_correct2.append(np.asarray(f[animal][session]['p_correct_e2']))
	e1_rates = np.nan_to_num(np.asarray(e1_rates))+1
	e2_rates = np.nan_to_num(np.asarray(e2_rates))+1
	for i in range(len(p_correct1)):
		p_correct1[i] = p_correct1[i].mean()
	for i in range(len(p_correct2)):
		p_correct2[i] = p_correct2[i].mean()
	p_correct1 = np.random.permutation(np.asarray(p_correct1))
	p_correct2 = np.random.permutation(np.asarray(p_correct2))
	f.close()
	##now that we have this data, plot the scatter plots and the correlations
	##first make one for E1 frs VS perforomance
	e1_idx = np.argsort(e1_rates)
	e2_idx = np.argsort(e2_rates)
	e1_rates = np.nan_to_num(e1_rates[e1_idx])+1
	e2_rates = np.nan_to_num(e2_rates[e2_idx])+1
	p_correct1 = np.nan_to_num(p_correct1[e1_idx])
	p_correct2 = np.nan_to_num(p_correct2[e2_idx])
	fig,(ax1,ax2) = plt.subplots(2,sharex=True)
	ax1.scatter(e1_rates,p_correct1,alpha=0.5,marker='o',color='g')
	ax2.scatter(e2_rates,p_correct2,alpha=0.5,marker='o',color='b')
	##the best fit lines
	par1 = np.polyfit(e1_rates,p_correct1,1,full=True)
	slope1=par1[0][0]
	intercept1=par1[0][1]
	xl1 = [min(e1_rates), max(e1_rates)]
	yl1 = [slope1*xx + intercept1  for xx in xl1]
	# coefficient of determination, plot text
	variance1 = np.var(p_correct1)
	residuals1 = np.var([(slope1*xx + intercept1 - yy)  for xx,yy in zip(e1_rates,p_correct1)])
	Rsqr1 = np.round(1-residuals1/variance1, decimals=5)
	ax1.text(.9*max(e1_rates)+.1*min(e2_rates),.9*max(p_correct1)+.1*min(p_correct1),
		'$R^2 = %0.5f$'% Rsqr1, fontsize=14)
	ax1.plot(e1_rates, np.poly1d(np.polyfit(e1_rates,p_correct1,1))(e1_rates))

	par2 = np.polyfit(e2_rates,p_correct2,1,full=True)
	slope2=par2[0][0]
	intercept2=par2[0][1]
	xl2 = [min(e2_rates), max(e2_rates)]
	yl2 = [slope2*xx + intercept2  for xx in xl2]
	# coefficient of determination, plot text
	variance2 = np.var(p_correct2)
	residuals2 = np.var([(slope2*xx + intercept2 - yy)  for xx,yy in zip(e2_rates,p_correct2)])
	Rsqr2 = np.round(1-residuals2/variance2, decimals=5)
	plt.text(.9*max(e2_rates)+.1*min(e2_rates),.9*max(p_correct2)+.1*min(p_correct2),
		'$R^2 = %0.5f$'% Rsqr2, fontsize=14)
	ax2.plot(e2_rates, np.poly1d(np.polyfit(e2_rates,p_correct2,1))(e2_rates))	

	ax1.set_title("E1 units",fontsize=14)
	ax2.set_title("E2_units",fontsize=14)
	ax1.set_ylabel("Percent correct",fontsize=14)
	ax2.set_ylabel("Percent correct",fontsize=14)
	ax2.set_xlabel("Mean FR, Hz",fontsize=14)
	p_e1 = stats.pearsonr(e1_rates,p_correct1)
	p_e2 = stats.pearsonr(e2_rates,p_correct2)
	print("pval e1 = "+str(p_e1))
	print("pval e2 = "+str(p_e2))


"""
A function to do logistic regression analysis on the indirect units, to see
how many of them are predictuve of E1 vs E2 choice.
"""
def log_regress_units(unit_types=['Str_units']):
	animal_list = None
	session_range = None
	window = [400,0]
	bin_width = 50
	##make some dictionaries to store the results
	results = {}
	##we should be able to run regression for each session as a whole.
	##first, we need to get two arrays: X; the data matrix of spike data
	##in dimensions trials x units x bins, and then y; the binary matrix
	## of target 1 and target 2 values.
	root_dir =r"D:\Ryan\V1_BMI"
	save_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\new_regressions\DMS_singles_400ms_bins.hdf5"
	##make some arrays to store
	if animal_list is None:
		animal_list = [x for x in ru.animals.keys() if not x.startswith('m')]
	for animal in animal_list:
		if session_range is None:
			session_list = ru.animals[animal][1].keys()
		else: 
			session_list = [x for x in f[animal].keys() if int(x[-6:-4]) in session_range]
		for session in session_list:
			##check to see if data for this session already exists
			try:
				f_out = h5py.File(save_file,'a')
				session_exists = f_out[animal][session]
				print(animal+" "+session+" data exists; moving to next file")
				f_out.close()
			except KeyError:
				f_out.close()
				print("analyzing data from "+animal+" "+session)
				try:
					t1_id = ru.animals[animal][1][session]['events']['t1'][0]
				except KeyError:
					t1_id = None
				try:
					t2_id = ru.animals[animal][1][session]['events']['t2'][0]
				except KeyError:
					t2_id = None
				##continue only if there are 2 targets for this file
				if t2_id is not None and t1_id is not None:
					##see if the correct unit types are present in this file
					unit_list = []
					for ut in unit_types:
						try:
							unit_ids = ru.animals[animal][1][session]['units'][ut]
							for ui in unit_ids:
								unit_list.append(ui)
						except KeyError:
							print("No "+ut+" present in this file")
					##continue only if there are units to analyze
					if len(unit_list)>0:
						plxfile = os.path.join(root_dir,animal,session)
						##load the plx file
						raw_data = plxread.import_file(plxfile,AD_channels=range(1,97),save_wf=False,
								import_unsorted=False,verbose=False)
						duration = None
						for arr in raw_data.keys():
							if arr.startswith('AD') and arr.endswith('_ts'):
								duration = int(np.ceil((raw_data[arr].max()*1000)/100)*100)+1
								break
						else: print("No A/D timestamp data found!!!")
						##make sure that this file had at least 20 trials of each type
						try:
							t1_ts = raw_data[t1_id]*1000.0
						except KeyError:
							t1_ts = np.array([])
						try: 
							t2_ts = raw_data[t2_id]*1000.0
						except KeyError:
							t2_ts = np.array([])
						if (t1_ts.size >= 15) and (t2_ts.size >= 15):
						##now load the data arrays for each unit
							X = []
							for unit in unit_list:
								try:
									spiketrain = raw_data[unit]*1000.0 #timestamps in ms
									##convert to binary array
									spiketrain = np.histogram(spiketrain,bins=duration,range=(0,duration))
									spiketrain = spiketrain[0].astype(bool).astype(int)
									spike_windows_t1 = get_data_window(spiketrain,t1_ts,window[0],window[1])
									spike_windows_t2 = get_data_window(spiketrain,t2_ts,window[0],window[1]) ##size bins x trials
									X.append(np.concatenate((spike_windows_t1,spike_windows_t2),axis=1)) ##concatenate trials
								except KeyError:
									print("Can't find "+unit+" in this file")
							##make sure we still have more than 0 units
							if len(X)>0:
								X = np.asarray(X).transpose(0,2,1) ##shape should now be units x trials x bins
								##add the number of total units to the data array
								total_units = X.shape[0]
								##now make our y dataset, which is just the trial outcome for all the trials
								t1s = np.ones(t1_ts.size)
								t2s = np.zeros(t2_ts.size)
								y = np.concatenate((t1s,t2s),axis=0)
								##now let's bin the spike data
								##I'm lazy and don't want to compute how big the binned data will be
								test = bin_spikes(X[0,0,:],bin_width)
								X_bins = np.zeros((X.shape[0],X.shape[1],test.shape[0]))
								for u in range(X.shape[0]):
									for t in range(X.shape[1]):
										X_bins[u,t,:] = bin_spikes(X[u,t,:],bin_width)
								##finally, we can actually do the regression
								accuracies, pvals = lr2.permutation_test_multi(X_bins,y,n_iter_cv=5,n_iter_p=500)
								##now just add the counts to the animal's array
								sig_units = (pvals<=0.05).sum()
								##now save these data
								f_out = h5py.File(save_file,'a')
								try:
									a_group = f_out[animal]
								except KeyError:
									a_group = f_out.create_group(animal)
								s_group = a_group.create_group(session)
								s_group.create_dataset('pvals',data=pvals)
								s_group.create_dataset('accuracies',data=accuracies)
								f_out.close()
	print("done")


"""
A function to do logistic regression analysis on the indirect units, as a group, to see
how many of them are predictuve of E1 vs E2 choice.
"""
def log_regress_grouped_units(unit_types=['Str_units']):
	animal_list = None
	session_range = None
	window = [400,0]
	bin_width = 50
	##make some dictionaries to store the results
	results = {}
	##we should be able to run regression for each session as a whole.
	##first, we need to get two arrays: X; the data matrix of spike data
	##in dimensions trials x units x bins, and then y; the binary matrix
	## of target 1 and target 2 values.
	root_dir =r"F:\data"
	save_file = r"F:\NatureNeuro\rebuttal\data\new_regressions\indirect_pop_400ms_bins.hdf5"
	##make some arrays to store
	if animal_list is None:
		animal_list = [x for x in ru.animals.keys() if not x.startswith('m')]
	for animal in animal_list:
		if session_range is None:
			session_list = ru.animals[animal][1].keys()
		else: 
			session_list = [x for x in f[animal].keys() if int(x[-6:-4]) in session_range]
		for session in session_list:
			##check to see if data for this session already exists
			try:
				f_out = h5py.File(save_file,'a')
				session_exists = f_out[animal][session]
				print(animal+" "+session+" data exists; moving to next file")
				f_out.close()
			except KeyError:
				f_out.close()
				print("analyzing data from "+animal+" "+session)
				try:
					t1_id = ru.animals[animal][1][session]['events']['t1'][0]
				except KeyError:
					t1_id = None
				try:
					t2_id = ru.animals[animal][1][session]['events']['t2'][0]
				except KeyError:
					t2_id = None
				##continue only if there are 2 targets for this file
				if t2_id is not None and t1_id is not None:
					##see if the correct unit types are present in this file
					unit_list = []
					for ut in unit_types:
						try:
							unit_ids = ru.animals[animal][1][session]['units'][ut]
							for ui in unit_ids:
								unit_list.append(ui)
						except KeyError:
							print("No "+ut+" present in this file")
					##continue only if there are units to analyze
					if len(unit_list)>0:
						plxfile = os.path.join(root_dir,animal,session)
						##load the plx file
						raw_data = plxread.import_file(plxfile,AD_channels=range(1,97),save_wf=False,
								import_unsorted=False,verbose=False)
						duration = None
						for arr in raw_data.keys():
							if arr.startswith('AD') and arr.endswith('_ts'):
								duration = int(np.ceil((raw_data[arr].max()*1000)/100)*100)+1
								break
						else: print("No A/D timestamp data found!!!")
						##make sure that this file had at least 20 trials of each type
						try:
							t1_ts = raw_data[t1_id]*1000.0
						except KeyError:
							t1_ts = np.array([])
						try: 
							t2_ts = raw_data[t2_id]*1000.0
						except KeyError:
							t2_ts = np.array([])
						if (t1_ts.size >= 15) and (t2_ts.size >= 15):
						##now load the data arrays for each unit
							X = []
							for unit in unit_list:
								try:
									spiketrain = raw_data[unit]*1000.0 #timestamps in ms
									##convert to binary array
									spiketrain = np.histogram(spiketrain,bins=duration,range=(0,duration))
									spiketrain = spiketrain[0].astype(bool).astype(int)
									spike_windows_t1 = get_data_window(spiketrain,t1_ts,window[0],window[1])
									spike_windows_t2 = get_data_window(spiketrain,t2_ts,window[0],window[1]) ##size bins x trials
									X.append(np.concatenate((spike_windows_t1,spike_windows_t2),axis=1)) ##concatenate trials
								except KeyError:
									print("Can't find "+unit+" in this file")
							##make sure we still have more than 0 units
							if len(X)>0:
								X = np.asarray(X).transpose(0,2,1) ##shape should now be units x trials x bins
								##add the number of total units to the data array
								total_units = X.shape[0]
								##now make our y dataset, which is just the trial outcome for all the trials
								t1s = np.ones(t1_ts.size)
								t2s = np.zeros(t2_ts.size)
								y = np.concatenate((t1s,t2s),axis=0)
								##now we'll just take the spike counts of X over the interval 
								X = X.sum(axis=2).T ##now it's just trials x units
								###not really sure if this is necessary, but lets just mix up the arrays
								idx = np.random.permutation(np.arange(y.shape[0]))
								y = y[idx]
								X = X[idx,:]
								##finally, we can actually do the regression
								accuracy,chance,pval = lr2.permutation_test((X,y,5,500))
								##now save the data
								f_out = h5py.File(save_file,'a')
								try:
									a_group = f_out[animal]
								except KeyError:
									a_group = f_out.create_group(animal)
								s_group = a_group.create_group(session)
								s_group.create_dataset('pval',data=np.array(pval))
								s_group.create_dataset('accuracy',data=np.array(accuracy))
								s_group.create_dataset('chance',data=np.asarray(chance))
								f_out.close()
	print("Done")


##function to plot the results from the above function
def plot_log_groups():
	datafile = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\new_regressions\indirect_pop_400ms_bins.hdf5"
	f = h5py.File(datafile,'r')
	animal_list = f.keys()
	##store the means of all the animals
	sig_vals = []
	accuracies = []
	chance = []
	for a in animal_list:
		sig = []
		acc = []
		ch = []
		for s in f[a].keys():
			sig.append(np.asarray(f[a][s]['pval'])*1)
			acc.append(np.asarray(f[a][s]['accuracy'])*1)
			ch.append(np.asarray(f[a][s]['chance'])*1)
		sig_vals.append(np.asarray(sig))
		accuracies.append(np.asarray(acc))
		chance.append(np.asarray(ch))
	f.close()
	sig_vals = equalize_arrs(sig_vals)
	accuracies = equalize_arrs(accuracies)
	chance = equalize_arrs(chance)
	x_axis = np.arange(1,accuracies.shape[1]+1)
	##count the number of animals for each session with significant predictability
	##first, the total number of animals that we have data for for this day
	total_animals = np.zeros(accuracies.shape[1])
	##and the total animals with significant predictability
	sig_animals = np.zeros(accuracies.shape[1])
	for i in range(accuracies.shape[1]):
		total_animals[i] = float(sig_vals[:,i][~np.isnan(sig_vals[:,i])].size)
		sig_animals[i] = float(np.where(sig_vals[:,i]<=0.05)[0].size)
	sig_perc = sig_animals/total_animals
	##now do the plots
	fig,(ax1,ax3) = plt.subplots(2,sharex=True)
	mean_acc = np.nanmean(accuracies,axis=0)
	serr_acc = np.nanstd(accuracies,axis=0)/np.sqrt(accuracies.shape[0])
	mean_chance = np.nanmean(chance,axis=0)
	serr_chance = np.nanstd(chance,axis=0)/np.sqrt(chance.shape[0])
	ax1.set_ylabel("Prediction\n accuracy",fontsize=14)
	ax3.set_ylabel("Percent\n significant",fontsize=14)
	ax3.set_xlabel("Training day",fontsize=14)
	ax1.set_title("Prediction accuracy of indirect population",fontsize=14)
	ax3.set_title("Percent of animals with significant prediction by direct units",fontsize=14)
	ax1.errorbar(x_axis,mean_acc,yerr=serr_acc,color='k',linewidth=2)
	ax1.fill_between(x_axis,mean_chance-serr_chance,mean_chance+serr_chance,color='cyan',alpha=0.5)
	ax3.plot(sig_perc,color='k',linewidth=2)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for i in range(accuracies.shape[0]):
		ax1.plot(x_axis,accuracies[i,:],alpha=0.5,color='k')
	##now do a summary figure for all days
	##now let's get the summary for all days
	x = np.array([1])
	err_x = np.array([0.1])
	labels = ['mean accuracy']
	acc_summ = np.nanmean(accuracies,axis=1)
	chance_summ = np.nanmean(chance,axis=1)
	acc_mean = acc_summ.mean()
	acc_err = stats.sem(acc_summ)
	chance_mean = chance_summ.mean()
	chance_err = stats.sem(chance_summ)
	fig,ax = plt.subplots(1)
	ax.errorbar(x,acc_mean,yerr=acc_err,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	for i in range(acc_summ.shape[0]):
		ax.plot(x,acc_summ[i],marker='o',color='k')
	##to plot chance rates
	chance_means = np.array([chance_mean,chance_mean,chance_mean])
	chance_errs = np.array([chance_err,chance_err,chance_err])
	ax.fill_between(np.array([0.5,1,1.5]),chance_means-chance_errs,chance_means+chance_errs,
		color='cyan',alpha=0.5)
	ax.set_xticks([1])
	ax.set_xticklabels(labels)
	ax.set_ylabel("Accuracy",fontsize=14)
	ax.set_title("Target choice predicted by indirect population",fontsize=14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)


def linear_regression_direct_indirect(unit_types=['Str_units']):
	direct_types = ['e1_units','e2_units']
	animal_list = None
	session_range = None
	window = [400,0]
	bin_width = 50
	##make some dictionaries to store the results
	results = {}
	##we should be able to run regression for each session as a whole.
	##first, we need to get two arrays: X; the data matrix of spike data
	##in dimensions trials x units x bins, and then y; the binary matrix
	## of target 1 and target 2 values.
	root_dir =r"D:\Ryan\V1_BMI"
	save_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\new_regressions\ridge_DMS_direct_400ms_bins.hdf5"
	##make some arrays to store
	if animal_list is None:
		animal_list = [x for x in ru.animals.keys() if not x.startswith('m')]
	for animal in animal_list:
		if session_range is None:
			session_list = ru.animals[animal][1].keys()
		else: 
			session_list = [x for x in f[animal].keys() if int(x[-6:-4]) in session_range]
		for session in session_list:
			##check to see if data for this session already exists
			try:
				f_out = h5py.File(save_file,'a')
				session_exists = f_out[animal][session]
				print(animal+" "+session+" data exists; moving to next file")
				f_out.close()
			except KeyError:
				f_out.close()
				print("analyzing data from "+animal+" "+session)
				try:
					t1_id = ru.animals[animal][1][session]['events']['t1'][0]
				except KeyError:
					t1_id = None
				try:
					t2_id = ru.animals[animal][1][session]['events']['t2'][0]
				except KeyError:
					t2_id = None
				##continue only if there are 2 targets for this file
				if t2_id is not None and t1_id is not None:
					##see if the correct unit types are present in this file
					unit_list = []
					for ut in unit_types:
						try:
							unit_ids = ru.animals[animal][1][session]['units'][ut]
							for ui in unit_ids:
								unit_list.append(ui)
						except KeyError:
							print("No "+ut+" present in this file")
					direct_list = []
					for ut in direct_types:
						try:
							unit_ids = ru.animals[animal][1][session]['units'][ut]
							for ui in unit_ids:
								direct_list.append(ui)
						except KeyError:
							print("No "+ut+" present in this file")
					##continue only if there are units to analyze
					if len(unit_list)>0 and len(direct_list)>0:
						plxfile = os.path.join(root_dir,animal,session)
						##load the plx file
						raw_data = plxread.import_file(plxfile,AD_channels=range(1,97),save_wf=False,
								import_unsorted=False,verbose=False)
						duration = None
						for arr in raw_data.keys():
							if arr.startswith('AD') and arr.endswith('_ts'):
								duration = int(np.ceil((raw_data[arr].max()*1000)/100)*100)+1
								break
						else: print("No A/D timestamp data found!!!")
						##make sure that this file had at least 20 trials of each type
						try:
							t1_ts = raw_data[t1_id]*1000.0
						except KeyError:
							t1_ts = np.array([])
						try: 
							t2_ts = raw_data[t2_id]*1000.0
						except KeyError:
							t2_ts = np.array([])
						if (t1_ts.size >= 1) and (t2_ts.size >= 1):
						##now load the data arrays for each direct unit
							y = []
							for unit in direct_list:
								try:
									spiketrain = raw_data[unit]*1000.0 #timestamps in ms
									##convert to binary array
									spiketrain = np.histogram(spiketrain,bins=duration,range=(0,duration))
									spiketrain = spiketrain[0].astype(bool).astype(int)
									spike_windows_t1 = get_data_window(spiketrain,t1_ts,window[0],window[1])
									spike_windows_t2 = get_data_window(spiketrain,t2_ts,window[0],window[1]) ##size bins x trials
									y.append(np.concatenate((spike_windows_t1,spike_windows_t2),axis=1)) ##concatenate trials
								except KeyError:
									print("Can't find "+unit+" in this file")
							##repeat for the other units
							X = []
							for unit in unit_list:
								try:
									spiketrain = raw_data[unit]*1000.0 #timestamps in ms
									##convert to binary array
									spiketrain = np.histogram(spiketrain,bins=duration,range=(0,duration))
									spiketrain = spiketrain[0].astype(bool).astype(int)
									spike_windows_t1 = get_data_window(spiketrain,t1_ts,window[0],window[1])
									spike_windows_t2 = get_data_window(spiketrain,t2_ts,window[0],window[1]) ##size bins x trials
									X.append(np.concatenate((spike_windows_t1,spike_windows_t2),axis=1)) ##concatenate trials
								except KeyError:
									print("Can't find "+unit+" in this file")
							##make sure we still have more than 0 units
							if len(X)>0 and len(y)>0:
								X = np.asarray(X) ##shape should now be units x bins x trials
								y = np.asarray(y)
								##add the number of total units to the data array
								total_units = X.shape[0]
								##sum the spike counts over the interval, and transpose into trials x units
								X = X.sum(axis=1).T
								y = y.sum(axis=1).T
								##FINALLY, we can run the linear regression
								R2s,pvals = lin2.permutation_test_multi(X,y,n_iter_cv=5,n_iter_p=500)
								##now save these data
								f_out = h5py.File(save_file,'a')
								try:
									a_group = f_out[animal]
								except KeyError:
									a_group = f_out.create_group(animal)
								s_group = a_group.create_group(session)
								s_group.create_dataset('pvals',data=pvals)
								s_group.create_dataset('R2s',data=R2s)
								f_out.close()
	print("Done")

##function to plot the results from the above function
def plot_lin_regression():
	datafile = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\new_regressions\ridge_indirect_direct_400ms.hdf5"
	f = h5py.File(datafile,'r')
	animal_list = f.keys()
	##store the means of all the animals
	totals = []
	sig_vals = []
	var_explained = []
	for a in animal_list:
		tot = []
		r2 = []
		sig = []
		for s in f[a].keys():
			r2.append(np.asarray(f[a][s]['R2s']).mean())
			sig.append((np.asarray(f[a][s]['pvals'])<0.05).sum())
			tot.append(np.asarray(f[a][s]['R2s']).size)
		totals.append(np.asarray(tot))
		sig_vals.append(np.asarray(sig))
		var_explained.append(np.asarray(r2))
	f.close()
	totals = equalize_arrs(totals)
	sig_vals = equalize_arrs(sig_vals)
	var_explained = equalize_arrs(var_explained)
	x_axis = np.arange(1,totals.shape[1]+1)
	sig_perc = sig_vals/totals
	##now do the plots
	fig,(ax1,ax2,ax3) = plt.subplots(3,sharex=True)
	mean_total = np.nanmean(totals,axis=0)
	serr_total = np.nanstd(totals,axis=0)/np.sqrt(totals.shape[0])
	mean_var = np.nanmean(var_explained,axis=0)
	serr_var = np.nanstd(var_explained,axis=0)/np.sqrt(var_explained.shape[0])
	mean_sig = np.nanmean(sig_perc,axis=0)
	serr_sig = np.nanstd(sig_perc,axis=0)/np.sqrt(sig_perc.shape[0])
	ax1.set_ylabel("R2",fontsize=14)
	ax2.set_ylabel("Number\n of units",fontsize=14)
	ax3.set_ylabel("Percent\n significant",fontsize=14)
	ax3.set_xlabel("Training day",fontsize=14)
	ax1.set_title("Variance of direct units explained by DMS units",fontsize=14)
	ax2.set_title ("Total number of DMS units",fontsize=14)
	ax3.set_title("Percent direct units with significant prediction by indirect units",fontsize=14)
	ax1.errorbar(x_axis,mean_var,yerr=serr_var,color='k',linewidth=2)
	ax2.errorbar(x_axis,mean_total,yerr=serr_total,color='k',linewidth=2)
	ax3.errorbar(x_axis,mean_sig,yerr=serr_sig,color='k',linewidth=2)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for i in range(totals.shape[0]):
		ax1.plot(x_axis,var_explained[i,:],alpha=0.5,color='k')
		ax2.plot(x_axis,totals[i,:],alpha=0.5,color='k')
		ax3.plot(x_axis,sig_perc[i,:],alpha=0.5,color='k')
	##now let's get the summary for all days
	x = np.array([1,2])
	err_x = np.array([0.1])
	labels = ['mean R2','perc sig']
	ps_summ = np.nanmean(sig_perc,axis=1)
	r2_summ = np.nanmean(var_explained,axis=1)
	ps_mean = ps_summ.mean()
	ps_err = stats.sem(ps_summ)
	r2_mean = r2_summ.mean()
	r2_err = stats.sem(r2_summ)
	fig,ax = plt.subplots(1)
	ax2 = ax.twinx()
	ax2.errorbar(x[0],ps_mean,yerr=ps_err,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	ax.errorbar(x[1],r2_mean,yerr=r2_err,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	for i in range(ps_summ.shape[0]):
		ax2.plot(x[0],ps_summ[i],marker='o',color='k')
	for i in range(r2_summ.shape[0]):
		ax.plot(x[1],r2_summ[i],marker='o',color='k')
	ax.set_xticks([1,2])
	ax.set_xticklabels(labels)
	ax.set_ylabel("R2",fontsize=14)
	ax2.set_ylabel("Percent significant",fontsize=14)
	ax.set_title("Direct unit activity predicted by indirect units",fontsize=14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	# ax1.set_xlim(0,12)
	#ax1.set_ylim(-1,0.4)
	# ax2.set_xlim(0,12)
	# ax3.set_xlim(0,12)

"""
A function to plot the results from the LR function (individual units)
"""
def plot_log_regression():
	datafile = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\new_regressions\indirect_singles_400ms_bins.hdf5"
	f = h5py.File(datafile,'r')
	animal_list = f.keys()
	##store the means of all the animals
	totals = []
	sigs = []
	accuracies = []
	for a in animal_list:
		sessions = np.sort(f[a].keys())
		tot = []
		acc = []
		sig = []
		##now get the data
		for s in sessions:
			a_data = np.asarray(f[a][s]['accuracies'])
			p_data = np.asarray(f[a][s]['pvals'])
			tot.append(a_data.size)
			acc.append(np.nanmean(a_data))
			sig.append((p_data<0.05).sum())
		accuracies.append(np.asarray(acc))
		totals.append(np.asarray(tot))
		sigs.append(np.asarray(sig))
	f.close()
	totals = equalize_arrs(totals)
	sigs = equalize_arrs(sigs)
	accuracies = equalize_arrs(accuracies)
	perc = (sigs/totals)*100
	mean = np.nanmean(perc,axis=0)
	serr = np.nanstd(perc,axis=0)/np.sqrt(perc.shape[0])
	fig,ax = plt.subplots(1)
	x_axis = np.arange(1,mean.size+1)
	ax.errorbar(x_axis,mean,yerr=serr,color='k',linewidth=2)
	for i in range(perc.shape[0]):
		ax.plot(x_axis,perc[i,:],color='k',linewidth=1,alpha=0.5)
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Percent of units",fontsize=14)
	ax.set_title("Indirect units predictive of target choice",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	##now for the totals and the raw sig numbers
	fig,(ax1,ax2) = plt.subplots(2,sharex=True)
	mean_total = np.nanmean(totals,axis=0)
	serr_total = np.nanstd(totals,axis=0)/np.sqrt(totals.shape[0])
	mean_sig = np.nanmean(sigs,axis=0)
	serr_sig = np.nanstd(sigs,axis=0)/np.sqrt(sigs.shape[0])
	ax1.set_ylabel("Number of units",fontsize=14)
	ax2.set_ylabel("Number of units",fontsize=14)
	ax2.set_xlabel("Training day",fontsize=14)
	ax1.set_title("Number of significant indirect units",fontsize=14)
	ax2.set_title ("Total number of indirect units",fontsize=14)
	ax1.errorbar(x_axis,mean_sig,yerr=serr_sig,color='k',linewidth=2)
	ax2.errorbar(x_axis,mean_total,yerr=serr_total,color='k',linewidth=2)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for i in range(totals.shape[0]):
		ax1.plot(x_axis,sigs[i,:],alpha=0.5,color='k')
		ax2.plot(x_axis,totals[i,:],alpha=0.5,color='k')
	##plot the accuracies
	mean = np.nanmean(accuracies,axis=0)
	serr = np.nanstd(accuracies,axis=0)/np.sqrt(accuracies.shape[0])
	fig,ax = plt.subplots(1)
	x_axis = np.arange(1,mean.size+1)
	ax.errorbar(x_axis,mean,yerr=serr,color='k',linewidth=2)
	for i in range(accuracies.shape[0]):
		ax.plot(x_axis,accuracies[i,:],color='k',linewidth=1,alpha=0.5)
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Mean accuracy",fontsize=14)
	ax.set_title("Accuracy of single indirect units for target choice",fontsize=14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	##now let's get the summary for all days
	x = np.array([1,2])
	err_x = np.array([0.1])
	labels = ['mean acc','perc sig']
	ps_summ = np.nanmean(perc,axis=1)
	r2_summ = np.nanmean(accuracies,axis=1)
	ps_mean = ps_summ.mean()
	ps_err = stats.sem(ps_summ)
	r2_mean = r2_summ.mean()
	r2_err = stats.sem(r2_summ)
	fig,ax = plt.subplots(1)
	ax2 = ax.twinx()
	ax2.errorbar(x[0],ps_mean,yerr=ps_err,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	ax.errorbar(x[1],r2_mean,yerr=r2_err,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	for i in range(ps_summ.shape[0]):
		ax2.plot(x[0],ps_summ[i],marker='o',color='k')
	for i in range(r2_summ.shape[0]):
		ax.plot(x[1],r2_summ[i],marker='o',color='k')
	ax.set_xticks([1,2])
	ax.set_xticklabels(labels)
	ax.set_ylabel("Accuracy",fontsize=14)
	ax2.set_ylabel("Percent units significant",fontsize=14)
	ax.set_title("Target choice by individual indirect units",fontsize=14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)



"""
A function to look at the online volitional target-locked modulations
compared to the ones observed during rewarded tone playback
"""
def get_rev1_bs():
	root_dir = r"K:\Ryan\V1_BMI"
	animal_list = ['V14','V15','V16'"R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13", "R7", "R8"]
	session_list = ['BMI_D06','BMI_D07']
	window = [6000,6000]
	save_file = h5py.File(r"D:\Ryan\V1_BMI\Cell\data\triggered_spike_rates.hdf5",'w-')
	##open the file 
	for animal in animal_list:
		a_group = save_file.create_group(animal)
		for session in session_list:
			s_group = a_group.create_group(session)
			file_v = os.path.join(root_dir,animal,session)+".plx" ##the file path to the volitional part
			file_pb = os.path.join(root_dir,animal,session)+"_pb.plx" ##path to the playback part
			##get the E1 and E2 unis for this session
			e1_list = ru.animals[animal][1][session+".plx"]['units']['e1_units']
			e2_list = ru.animals[animal][1][session+".plx"]['units']['e2_units']
			##start with the non-manipulation file
			v_data = plxread.import_file(file_v,AD_channels=range(1,97),save_wf=True,
				import_unsorted=False,verbose=False)
			pb_data = plxread.import_file(file_pb,AD_channels=range(1,97),save_wf=True,
				import_unsorted=False,verbose=False)
			##what is the duration of this file
			duration_v = None
			for arr in v_data.keys():
				if arr.startswith('AD') and arr.endswith('_ts'):
					duration_v = int(np.ceil((v_data[arr].max()*1000)/100)*100)+1
					break
			else: print("No A/D timestamp data found!!!")
			duration_pb = None
			for arr in pb_data.keys():
				if arr.startswith('AD') and arr.endswith('_ts'):
					duration_pb = int(np.ceil((pb_data[arr].max()*1000)/100)*100)+1
					break
			else: print("No A/D timestamp data found!!!")
			##now get the E1 unit data 
			e1_v = []
			e1_pb = []
			for unit in e1_list:
				spiketrain_v = v_data[unit] * 1000 #timestamps in ms
				##convert to binary array
				spiketrain_v = np.histogram(spiketrain_v,bins=duration_v,range=(0,duration_v))
				spiketrain_v = spiketrain_v[0].astype(bool).astype(int)
				e1_v.append(spiketrain_v)
				##repeat for pb data
				spiketrain_pb = pb_data[unit] * 1000 #timestamps in ms
				##convert to binary array
				spiketrain_pb = np.histogram(spiketrain_pb,bins=duration_pb,range=(0,duration_pb))
				spiketrain_pb = spiketrain_pb[0].astype(bool).astype(int)
				e1_pb.append(spiketrain_pb)
			##convert to a numpy array
			e1_v = np.asarray(e1_v)
			e1_pb = np.asarray(e1_pb)
			##now get the E2 unit data 
			e2_v = []
			e2_pb = []
			for unit in e2_list:
				spiketrain_v = v_data[unit] * 1000 #timestamps in ms
				##convert to binary array
				spiketrain_v = np.histogram(spiketrain_v,bins=duration_v,range=(0,duration_v))
				spiketrain_v = spiketrain_v[0].astype(bool).astype(int)
				e2_v.append(spiketrain_v)
				##repeat for pb data
				spiketrain_pb = pb_data[unit] * 1000 #timestamps in ms
				##convert to binary array
				spiketrain_pb = np.histogram(spiketrain_pb,bins=duration_pb,range=(0,duration_pb))
				spiketrain_pb = spiketrain_pb[0].astype(bool).astype(int)
				e2_pb.append(spiketrain_pb)
			##convert to a numpy array
			e2_v = np.asarray(e2_v)
			e2_pb = np.asarray(e2_pb)
			##we are also going to need the T1 and T2 timestamps fo each file
			t1_id = ru.animals[animal][1][session+".plx"]['events']['t1'][0] ##the event name in the plexon file
			t2_id = ru.animals[animal][1][session+".plx"]['events']['t2'][0]
			##get the event ts for each file
			t1_v = v_data[t1_id]*1000.0
			t2_v = v_data[t2_id]*1000.0
			t1_pb = pb_data[t1_id]*1000.0
			t2_pb = pb_data[t2_id]*1000.0
			##now get the time-locked data
			t1_e1_v = ds.get_data_window(t1_v,window[0],window[1],e1_v)
			t2_e1_v = ds.get_data_window(t2_v,window[0],window[1],e1_v)
			t1_e2_v = ds.get_data_window(t1_v,window[0],window[1],e2_v)
			t2_e2_v = ds.get_data_window(t2_v,window[0],window[1],e2_v)
			##repeat for playback data
			t1_e1_pb = ds.get_data_window(t1_pb,window[0],window[1],e1_pb)
			t2_e1_pb = ds.get_data_window(t2_pb,window[0],window[1],e1_pb)
			t1_e2_pb = ds.get_data_window(t1_pb,window[0],window[1],e2_pb)
			t2_e2_pb = ds.get_data_window(t2_pb,window[0],window[1],e2_pb)
			##save all of this data to the save file
			s_group.create_dataset('t1_e1_v',data=t1_e1_v)
			s_group.create_dataset('t2_e1_v',data=t2_e1_v)
			s_group.create_dataset('t1_e2_v',data=t1_e2_v)
			s_group.create_dataset('t2_e2_v',data=t2_e2_v)
			s_group.create_dataset('t1_e1_pb',data=t1_e1_pb)
			s_group.create_dataset('t2_e1_pb',data=t2_e1_pb)
			s_group.create_dataset('t1_e2_pb',data=t1_e2_pb)
			s_group.create_dataset('t2_e2_pb',data=t2_e2_pb)
	save_file.close()
	print('Done')

"""
a function to plot the playback data saved by the above function
"""
def plot_rev1_bs():
	datafile = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\rev1_expt.hdf5"
	f = h5py.File(datafile,'r')
	animal_list = f.keys()
	##lists to save the data averaged by animal
	e1_t1_v = []
	e1_t2_v = []
	e2_t1_v = []
	e2_t2_v = []
	##same for the playback data
	e1_t1_pb = []
	e1_t2_pb = []
	e2_t1_pb = []
	e2_t2_pb = []
	##go through each animal and take the average of the sessions 
	for animal in animal_list:
		print("working on animal "+animal)
		e1t1v = []
		e1t2v = []
		e2t1v = []
		e2t2v = []
		### the playback ones
		e1t1pb = []
		e1t2pb = []
		e2t1pb = []
		e2t2pb = []
		for session in f[animal].keys():
			print("working on session "+session)
			e1t1v.append(np.asarray(f[animal][session]['t1_e1_v']))
			e1t2v.append(np.asarray(f[animal][session]['t2_e1_v']))
			e2t1v.append(np.asarray(f[animal][session]['t1_e2_v']))
			e2t2v.append(np.asarray(f[animal][session]['t2_e2_v']))
			##repeat for playback
			e1t1pb.append(np.asarray(f[animal][session]['t1_e1_pb']))
			e1t2pb.append(np.asarray(f[animal][session]['t2_e1_pb']))
			e2t1pb.append(np.asarray(f[animal][session]['t1_e2_pb']))
			e2t2pb.append(np.asarray(f[animal][session]['t2_e2_pb']))
		##now concatenate trials from both sessions, and add to the animal average
		##also take the across animal average over both sessions and all trials
		e1_t1_v.append(np.concatenate(e1t1v,axis=2).mean(axis=2))
		e1_t2_v.append(np.concatenate(e1t2v,axis=2).mean(axis=2))
		e2_t1_v.append(np.concatenate(e2t1v,axis=2).mean(axis=2))
		e2_t2_v.append(np.concatenate(e2t2v,axis=2).mean(axis=2))
		##same for playback
		e1_t1_pb.append(np.concatenate(e1t1pb,axis=2).mean(axis=2))
		e1_t2_pb.append(np.concatenate(e1t2pb,axis=2).mean(axis=2))
		e2_t1_pb.append(np.concatenate(e2t1pb,axis=2).mean(axis=2))
		e2_t2_pb.append(np.concatenate(e2t1pb,axis=2).mean(axis=2))
	##now convert to a numpy array
	e1_t1_v = np.asarray(e1_t1_v) ##now shape is animals x units x time, averaged over sessions
	e1_t2_v = np.asarray(e1_t2_v)
	e2_t1_v = np.asarray(e2_t1_v)
	e2_t2_v = np.asarray(e2_t2_v)
	##repeat for playback
	e1_t1_pb = np.asarray(e1_t1_pb)
	e1_t2_pb = np.asarray(e1_t2_pb)
	e2_t1_pb = np.asarray(e2_t1_pb)
	e2_t2_pb = np.asarray(e2_t2_pb)
	##now I probably want to look at the averages for each animal separately
	win_d = ss.windowRate(e1_t1_v[0,:,:].T,[200,50]).shape[0]
	e1_t1_v_means = np.zeros((e1_t1_v.shape[0],win_d)) ##being lazy and not calculating this other dimension
	e1_t2_v_means = np.zeros((e1_t2_v.shape[0],win_d))
	e2_t1_v_means = np.zeros((e2_t1_v.shape[0],win_d))
	e2_t2_v_means = np.zeros((e2_t2_v.shape[0],win_d))
	####
	e1_t1_pb_means = np.zeros((e1_t1_pb.shape[0],win_d))
	e1_t2_pb_means = np.zeros((e1_t2_pb.shape[0],win_d))
	e2_t1_pb_means = np.zeros((e2_t1_pb.shape[0],win_d))
	e2_t2_pb_means = np.zeros((e2_t2_pb.shape[0],win_d))
	for a in range(e1_t1_v.shape[0]):
		e1_t1_v_means[a,:] = stats.zscore(ss.windowRate(e1_t1_v[a,:,:].T,[200,50]).mean(axis=1)) ##this is the average trace for this animal, over all units/trials
		e1_t2_v_means[a,:] = stats.zscore(ss.windowRate(e1_t2_v[a,:,:].T,[200,50]).mean(axis=1))
		e2_t1_v_means[a,:] = stats.zscore(ss.windowRate(e2_t1_v[a,:,:].T,[200,50]).mean(axis=1))
		e2_t2_v_means[a,:] = stats.zscore(ss.windowRate(e2_t2_v[a,:,:].T,[200,50]).mean(axis=1))
		###
		e1_t1_pb_means[a,:] = stats.zscore(ss.windowRate(e1_t1_pb[a,:,:].T,[200,50]).mean(axis=1)) ##this is the average trace for this animal, over all units/trials
		e1_t2_pb_means[a,:] = stats.zscore(ss.windowRate(e1_t2_pb[a,:,:].T,[200,50]).mean(axis=1))
		e2_t1_pb_means[a,:] = stats.zscore(ss.windowRate(e2_t1_pb[a,:,:].T,[200,50]).mean(axis=1))
		e2_t2_pb_means[a,:] = stats.zscore(ss.windowRate(e2_t1_pb[a,:,:].T,[200,50]).mean(axis=1))
	e2_t1_v_means = e1_t2_v_means
	##now get the means and std errs
	e1_t1_v_mean = e1_t1_v_means.mean(axis=0)
	e1_t2_v_mean = e1_t2_v_means.mean(axis=0)
	e2_t1_v_mean = e2_t1_v_means.mean(axis=0)
	e2_t2_v_mean = e2_t2_v_means.mean(axis=0)
	##now the std dev
	e1_t1_v_serr = e1_t1_v_means.std(axis=0)/np.sqrt(e1_t1_v_means.shape[0])
	e1_t2_v_serr = e1_t2_v_means.std(axis=0)/np.sqrt(e1_t2_v_means.shape[0])
	e2_t1_v_serr = e1_t2_v_means.std(axis=0)/np.sqrt(e2_t1_v_means.shape[0])
	e2_t2_v_serr = e2_t2_v_means.std(axis=0)/np.sqrt(e2_t2_v_means.shape[0])
	###repeat for the pb
	e1_t1_pb_mean = e1_t1_pb_means.mean(axis=0)
	e1_t2_pb_mean = e1_t2_pb_means.mean(axis=0)
	e2_t1_pb_mean = e2_t1_pb_means.mean(axis=0)
	e2_t2_pb_mean = e2_t2_pb_means.mean(axis=0)
	##now the std dev
	e1_t1_pb_serr = e1_t1_pb_means.std(axis=0)/np.sqrt(e1_t1_pb_means.shape[0])
	e1_t2_pb_serr = e1_t2_pb_means.std(axis=0)/np.sqrt(e1_t2_pb_means.shape[0])
	e2_t1_pb_serr = e1_t2_pb_means.std(axis=0)/np.sqrt(e2_t1_pb_means.shape[0])
	e2_t2_pb_serr = e2_t2_pb_means.std(axis=0)/np.sqrt(e2_t2_pb_means.shape[0])
	##now plot this BS!!!
	x = np.linspace(-6,6,e1_t1_v_means.shape[1])
	t1_fig = plt.figure()
	t1_fig.suptitle("Rewarded target",fontsize=14)
	ax_v = t1_fig.add_subplot(121)
	ax_v.set_title("Online control",fontsize=14,weight='bold')
	ax_v.set_ylabel("Firing rate, z-scored",fontsize=14)
	ax_v.set_xlabel("Time to target, s",fontsize=14)
	ax_v.set_xlim(-5,5)
	ax_v.set_ylim(-4,6)
	for ticklabel in ax_v.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax_v.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax_v.plot(x,e1_t1_v_mean,linewidth=2,color='g',label='E1')
	ax_v.fill_between(x,e1_t1_v_mean-e1_t1_v_serr,e1_t1_v_mean+e1_t1_v_serr,color='g',alpha=0.5)
	ax_v.plot(x,e2_t1_v_mean,linewidth=2,color='b',label='E2')
	ax_v.fill_between(x,e2_t1_v_mean+e2_t1_v_serr,e2_t1_v_mean-e2_t1_v_serr,color='b',alpha=0.5)
	ax_pb = t1_fig.add_subplot(122)
	ax_pb.set_title("Tone playback",fontsize=14,weight='bold')
	ax_pb.set_ylabel("Firing rate, z-scored",fontsize=14)
	ax_pb.set_xlabel("Time to target, s",fontsize=14)
	ax_pb.set_xlim(-5,5)
	ax_pb.set_ylim(-4,6)
	for ticklabel in ax_pb.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax_pb.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax_pb.plot(x,e1_t1_pb_mean,linewidth=2,color='g',label='E1')
	ax_pb.fill_between(x,e1_t1_pb_mean-e1_t1_pb_serr,e1_t1_pb_mean+e1_t1_pb_serr,color='g',alpha=0.5)
	ax_pb.plot(x,e2_t1_pb_mean,linewidth=2,color='b',label='E2')
	ax_pb.fill_between(x,e2_t1_pb_mean+e2_t1_pb_serr,e2_t1_pb_mean-e2_t1_pb_serr,color='b',alpha=0.5)
	ax_pb.legend()
	##repeat for target 2
	t2_fig = plt.figure()
	t2_fig.suptitle("Unrewarded target",fontsize=14)
	ax_v = t2_fig.add_subplot(121)
	ax_v.set_title("Online control",fontsize=14,weight='bold')
	ax_v.set_ylabel("Firing rate, z-scored",fontsize=14)
	ax_v.set_xlabel("Time to target, s",fontsize=14)
	ax_v.set_xlim(-5,5)
	ax_v.set_ylim(-4,6)
	for ticklabel in ax_v.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax_v.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax_v.plot(x,e1_t2_v_mean,linewidth=2,color='g',label='E1')
	ax_v.fill_between(x,e1_t2_v_mean-e1_t2_v_serr,e1_t2_v_mean+e1_t2_v_serr,color='g',alpha=0.5)
	ax_v.plot(x,e2_t2_v_mean,linewidth=2,color='b',label='E2')
	ax_v.fill_between(x,e2_t2_v_mean+e2_t2_v_serr,e2_t2_v_mean-e2_t2_v_serr,color='b',alpha=0.5)
	ax_pb = t2_fig.add_subplot(122)
	ax_pb.set_title("Tone playback",fontsize=14,weight='bold')
	ax_pb.set_ylabel("Firing rate, z-scored",fontsize=14)
	ax_pb.set_xlabel("Time to target, s",fontsize=14)
	ax_pb.set_xlim(-5,5)
	ax_pb.set_ylim(-4,6)
	for ticklabel in ax_pb.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax_pb.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax_pb.plot(x,e1_t2_pb_mean,linewidth=2,color='g',label='E1')
	ax_pb.fill_between(x,e1_t2_pb_mean-e1_t2_pb_serr,e1_t2_pb_mean+e1_t2_pb_serr,color='g',alpha=0.5)
	ax_pb.plot(x,e2_t2_pb_mean,linewidth=2,color='b',label='E2')
	ax_pb.fill_between(x,e2_t2_pb_mean+e2_t2_pb_serr,e2_t2_pb_mean-e2_t2_pb_serr,color='b',alpha=0.5)
	ax_pb.legend()
	###now I guess we can also plot the modulation depths
	e1_t1_mod_v = abs(e1_t1_v_means[:,110:120]).max(axis=1)
	e1_t2_mod_v = abs(e1_t2_v_means[:,110:120]).max(axis=1)
	e2_t1_mod_v = abs(e2_t1_v_means[:,110:120]).max(axis=1)
	e2_t2_mod_v = abs(e2_t2_v_means[:,110:120]).max(axis=1)
	###
	e1_t1_mod_pb = abs(e1_t1_pb_means[:,110:120]).max(axis=1)
	e1_t2_mod_pb = abs(e1_t2_pb_means[:,110:120]).max(axis=1)
	e2_t1_mod_pb = abs(e2_t1_pb_means[:,110:120]).max(axis=1)
	e2_t2_mod_pb = abs(e2_t2_pb_means[:,110:120]).max(axis=1)
	##now do the combination of all ensembles
	t1_mod_v = np.concatenate((e1_t1_mod_v,e2_t1_mod_v))
	t1_mod_pb = np.concatenate((e1_t1_mod_pb,e2_t1_mod_pb))
	##let's say we only care about the rewarded target
	e1_mod = np.vstack((e1_t1_mod_v,e1_t1_mod_pb)) ##now shape is conditions x animals
	e2_mod = np.vstack((e2_t1_mod_v,e2_t1_mod_pb))
	all_mod = np.vstack((t1_mod_v,t1_mod_pb))
	e1_mod_mean = e1_mod.mean(axis=1)
	all_mod_mean = all_mod.mean(axis=1)
	all_mod_serr = all_mod.std(axis=1)/np.sqrt(all_mod.shape[0])
	e1_mod_serr = e1_mod.std(axis=1)/np.sqrt(e1_t1_mod_v.shape[0])
	##and for E2
	e2_mod_mean = e1_mod.mean(axis=1)
	e2_mod_serr = e2_mod.std(axis=1)/np.sqrt(e1_t1_mod_v.shape[0])
	##now do the plotting
	t1_mod_fig = plt.figure()
	ax_e1 = t1_mod_fig.add_subplot(121)
	x2 = np.array([0,1])
	err_x = np.array([0.2,0.2])
	for i in range(e1_mod.shape[1]):
		ax_e1.plot(x2,e1_mod[:,i],color='g',linewidth=2,marker='o')
	ax_e1.errorbar(x2,e1_mod_mean,yerr=e1_mod_serr,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),['Online','Playback'])
	for ticklabel in ax_e1.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax_e1.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax_e1.set_xlim(-0.5,1.5)
	ax_e1.set_ylim(0,5.5)
	ax_e1.set_ylabel("Modulation depth",fontsize=14)
	ax_e1.set_title("Average E1 modulation depth",fontsize=14)
	##now for E2
	ax_e2 = t1_mod_fig.add_subplot(122)
	for i in range(e2_mod.shape[1]):
		ax_e2.plot(x2,e2_mod[:,i],color='b',linewidth=2,marker='o')
	ax_e2.errorbar(x2,e2_mod_mean,yerr=e2_mod_serr,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),['Online','Playback'])
	for ticklabel in ax_e2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax_e2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax_e2.set_xlim(-0.5,1.5)
	ax_e2.set_ylim(0,5.5)
	ax_e2.set_ylabel("Modulation depth",fontsize=14)
	ax_e2.set_title("Average E2 modulation depth",fontsize=14)
	all_mod_fig = plt.figure()
	ax_all = all_mod_fig.add_subplot(111)
	x2 = np.array([0,1])
	err_x = np.array([0.2,0.2])
	for i in range(e1_mod.shape[1]):
		ax_all.plot(x2,e1_mod[:,i],color='g',linewidth=2,marker='o')
	for i in range(e2_mod.shape[1]):
		ax_all.plot(x2,e2_mod[:,i],color='b',linewidth=2,marker='o')
	ax_all.errorbar(x2,all_mod_mean,yerr=all_mod_serr,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),['Online','Playback'])
	for ticklabel in ax_all.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax_all.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax_all.set_xlim(-0.5,1.5)
	ax_all.set_ylim(0,5.5)
	ax_all.set_ylabel("Modulation depth",fontsize=14)
	ax_all.set_title("Average ensemble modulation depth",fontsize=14)
	#finally do some significance testing
	pval_e1 = stats.ttest_rel(e1_t1_mod_v, e1_t1_mod_pb)[1]
	tval_e1 =stats.ttest_rel(e1_t1_mod_v, e1_t1_mod_pb)[0]
	pval_e2 = stats.ttest_rel(e2_t1_mod_v, e2_t1_mod_pb)[1]
	tval_e2 =stats.ttest_rel(e2_t1_mod_v, e2_t1_mod_pb)[0]
	pval_all = stats.ttest_rel(t1_mod_v,t1_mod_pb)[1]
	tval_all = stats.ttest_rel(t1_mod_v,t1_mod_pb)[0]
	print("E1 online mean = "+str(e1_t1_mod_v.mean()))
	print("E1 playback mean = "+str(e1_t1_mod_pb.mean()))
	print("E1 pval = "+str(pval_e1))
	print("E1 tval = "+str(tval_e1))
	print("E2 online mean = "+str(e2_t1_mod_v.mean()))
	print("E2 playback mean = "+str(e2_t1_mod_pb.mean()))
	print("E2 pval = "+str(pval_e2))
	print("E2 tval = "+str(tval_e2))
	print("all playback mean = "+str(t1_mod_pb.mean()))
	print("all online mean = "+str(t1_mod_v.mean()))
	print("all pval = "+str(pval_all))
	print("all tval = "+str(tval_all))


def get_peg_e1_e2():
	root_dir = r"D:\Ryan\V1_BMI"
	animal_list = ['V14','V15','V16']
	session_list = ['BMI_D08']
	##open the file 
	p_e1 = []
	p_e2 =[]
	p_ctrl = []
	for animal in animal_list:
		for session in session_list:
			filepath = os.path.join(root_dir,animal,session)+".plx" ##the file path to the volitional part
			##start with the non-manipulation file
			data = plxread.import_file(filepath,AD_channels=range(1,97),save_wf=True,
				import_unsorted=False,verbose=False)
			##we are going to need the T1 and T2 timestamps fo each file
			t1_id = ru.animals[animal][1][session+".plx"]['events']['t1'][0] ##the event name in the plexon file
			t2_id = ru.animals[animal][1][session+".plx"]['events']['t2'][0]
			miss_id = ru.animals[animal][1][session+".plx"]['events']['miss'][0]
			peg_e1_id = ru.animals[animal][1][session+".plx"]['events']['peg_e1'][0]
			peg_e2_id = ru.animals[animal][1][session+".plx"]['events']['peg_e2'][0]
			##get the event ts for each file
			t1 = data[t1_id]*1000.0
			t2 = data[t2_id]*1000.0
			miss = data[miss_id]*1000.0
			e1_catch = data[peg_e1_id]*1000.0
			e2_catch = data[peg_e2_id]*1000.0
			##now we'll make 2 arrays; one with the event IDs, and the other with the event TS
			ids = []
			ts = []
			for i in range(t1.size):
				ids.append('t1')
				ts.append(t1[i])
			for j in range(t2.size):
				ids.append('t2')
				ts.append(t2[j])
			for k in range(miss.size):
				ids.append('miss')
				ts.append(miss[k])
			for l in range(e1_catch.size):
				ids.append('e1_catch')
				ts.append(e1_catch[l])
			for m in range(e2_catch.size):
				ids.append('e2_catch')
				ts.append(e2_catch[m])
			##now put everything in order
			ids = np.asarray(ids)
			ts = np.asarray(ts)
			idx = np.argsort(ts)
			ids = ids[idx]
			ts = ts[idx]
			##now make sure I didn't screw up and put 2 catch trials in a row
			i = 0
			del_idx = []
			last_id = ''
			for i in range(ids.size):
				current_id = ids[i]
				if (last_id == 'e1_catch' and current_id == 'e1_catch') or (last_id == 
					'e2_catch' and current_id == 'e2_catch'):
					del_idx.append(i)
					print("repeat of "+current_id)
				last_id = current_id
			
			for d in range(len(del_idx)):
				np.delete(ids,del_idx[d])
			##OK, now we want to know the percent correct for regular trials, e1_fix trials and e2_fix trials
			##gonna combine miss and unrewarded
			correct_e1 = 0
			incorrect_e1 = 0
			correct_e2 = 0
			incorrect_e2 = 0
			correct = 0
			incorrect = 0
			i = 0
			while i < ids.size:
				if ids[i] == 'e1_catch': ##case e1 catch trial
					if ids[i+1] == 't1':
						correct_e1 += 1
					elif ids[i+1] == 't2' or ids[i+1] == 'miss':
						incorrect_e1 += 1
					i+=2
				elif ids[i] == 'e2_catch': ##case e2 catch trial
					if ids[i+1] == 't1':
						correct_e2 += 1
					elif ids[i+1] == 't2' or ids[i+1] == 'miss':
						incorrect_e2 += 1
					i+=2
				elif ids[i] == 't1':
					correct += 1
					i+=1
				elif ids[i] == 't2' or ids[i] == 'miss':
					incorrect += 1
					i+=1
				else:
					print("unrecognized event ID: "+ids[i])
					i+=1
			p_e1.append(float(correct_e1)/(incorrect_e1+correct_e1))
			p_e2.append(float(correct_e2)/(incorrect_e2+correct_e2))
			p_ctrl.append(float(correct)/(incorrect+correct))
	p_ctrl = np.asarray(p_ctrl)
	p_e1 = np.asarray(p_e1)
	p_e2 = np.asarray(p_e2)
	means = np.array([p_ctrl.mean(),p_e1.mean(),p_e2.mean()])
	sems = np.array([stats.sem(p_ctrl),stats.sem(p_e1),stats.sem(p_e2)])
	labels1 = ["Control","Fixed E1","Fixed E2"]
	plot_data = np.vstack((p_ctrl,p_e1,p_e2))
	fig, ax2 = plt.subplots(1)
	x = np.array([0,1,2])
	err_x = np.array([0,1,2])
	yerr = sems
	xerr = np.ones(3)*0.25
	colors = ['k','g','b']
	for i in range(plot_data.shape[1]):
		ax2.plot(np.zeros(3)+i,plot_data[i,:],color=colors[i],linewidth=2,marker='o',linestyle='none')
	ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,3),labels1)
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.3,2.3)
	ax2.set_ylim(-0.1,0.8)
	ax2.set_ylabel("Percent correct",fontsize=14)
	ax2.set_title("Training with fixed ensembles",fontsize=14)
	pval_e1 = stats.ttest_rel(p_ctrl, p_e1)[1]
	tval_e1 =stats.ttest_rel(p_ctrl, p_e1)[0]
	pval_e2 = stats.ttest_rel(p_ctrl, p_e2)[1]
	tval_e2 =stats.ttest_rel(p_ctrl, p_e2)[0]
	print("ctrl mean = "+str(p_ctrl.mean()))
	print("peg e1 mean = "+str(p_e1.mean()))
	print("E1 pval = "+str(pval_e1))
	print("E1 tval = "+str(tval_e1))
	print("peg e2 mean = "+str(p_e2.mean()))
	print("E2 pval = "+str(pval_e2))
	print("E2 tval = "+str(tval_e2))
	return p_ctrl,p_e1,p_e2

def get_time_locked_lfp():
	unit_type = 'V1_lfp' ##the type of units to run regression on
	root_dir = r"D:\Ryan\V1_BMI"
	animal_list = [x for x in ru.animals.keys() if not x.startswith("m")]
	session_list = None
	window = [2500,1500]
	target = 't1'
	save_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\rat_lfp_t1.hdf5"
	f_out = h5py.File(save_file,'a')
	f_out.close()
	for animal in animal_list:
		f_out = h5py.File(save_file,'a')
		try:
			a_group = f_out[animal]
			f_out.close()
		except KeyError:
			a_group = f_out.create_group(animal)
			f_out.close()
			#if session_list is None:
			session_list = ru.animals[animal][1].keys()
			for session in session_list:
				print("Working on "+animal+" "+session)
				filepath = os.path.join(root_dir,animal,session) ##the file path to the volitional part
				##start with the non-manipulation file
				data = plxread.import_file(filepath,AD_channels=list(range(1,97)),save_wf=False,
					import_unsorted=False,verbose=False)
				##we are going to need the T1 and T2 timestamps fo each file
				t1_id = ru.animals[animal][1][session]['events'][target][0] ##the event name in the plexon file
				try:
					lfp_id = ru.animals[animal][1][session]['lfp'][unit_type][0] ##we'll just take the first LFP channel since many only have one chan anyway
				except:
					unit_type2 = list(ru.animals[animal][1][session]['lfp'])[0]
					lfp_id = ru.animals[animal][1][session]['lfp'][unit_type2][0]
				t1_ts = data[t1_id]*1000
				lfp = data[lfp_id]
				if len(t1_ts)>0:
					traces = get_data_window_lfp(lfp,t1_ts,window[0],window[1])
					if traces != None:
						f_out = h5py.File(save_file,'a')
						f_out[animal].create_dataset(session,data=traces)
						f_out.close()
	# try:
	# 	with h5py.File(save_file,'a') as f:
	# 		f.__delitem__('m11/BMI_D09')
	# except KeyError:
	# 	pass
	print ('done!')

def get_mouse_lfp_specgram():
	datafile = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\mouse_lfp_t1.hdf5"
	save_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\mouse_lfp_spec_by_trial.hdf5"
	f = h5py.File(datafile,'r')
	f_out = h5py.File(save_file,'w')
	animals = f.keys()
	all_sessions = []
	for animal in animals:
		a_group = f_out.create_group(animal)
		sessions = f[animal].keys()
		for session in sessions:
			data = np.asarray(f[animal][session])
			S, t, fr, Serr = specs.lfpSpecGram(data,[0.75,0.05],Fs=1000.0,fpass=[0,150],err=None,
				sigType='lfp',norm=True, trialave=False)
			a_group.create_dataset(session,data=S)
			all_sessions.append(S)
	f_out.create_dataset('all_sessions',data=np.concatenate(all_sessions,axis=2))
	f.close()
	f_out.close()

def get_mouse_lfp_spectrum():
	datafile = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\lfp_t1.hdf5"
	save_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\mouse_spectrums.hdf5"
	f = h5py.File(datafile,'r')
	f_out = h5py.File(save_file,'w')
	idx = np.arange(0,2000)
	animals = f.keys()
	all_sessions = []
	for animal in animals:
		a_group = f_out.create_group(animal)
		sessions = f[animal].keys()
		for session in sessions:
			data = np.asarray(f[animal][session])[idx]
			S, fr, Serr = specs.mtspectrum(data,Fs=1000.0,fpass=[0,150],err=None,
				sigType='lfp')
			a_group.create_dataset(session,data=S)
			all_sessions.append(S)
	f_out.create_dataset('all_sessions',data=np.asarray(all_sessions))
	f.close()
	f_out.close()

def plot_lfp_specgram():
	source_file =  r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\mouse_lfp_spec.hdf5"
	f = h5py.File(source_file,'r')
	early_range = np.arange(6,9)
	late_range = np.arange(9,13)
	vmin=0.85
	vmax=1.35
	by_animal_early = []
	by_animal_late = []
	by_session_early = []
	by_session_late = []
	for animal in [x for x in f.keys() if not x == 'all_sessions']:
		all_sessions = []
		if early_range == None:
			sessions = f[animal].keys()
		else:
			sessions = [x for x in f[animal].keys() if int(x[-6:-4]) in early_range]
		for session in sessions:
			data = np.asarray(f[animal][session])
			all_sessions.append(data)
			by_session_early.append(data)
		by_animal_early.append(np.nanmean(np.asarray(all_sessions),axis=0))
	for animal in [x for x in f.keys() if not x == 'all_sessions']:
		all_sessions = []
		if late_range == None:
			sessions = f[animal].keys()
		else:
			sessions = [x for x in f[animal].keys() if int(x[-6:-4]) in late_range]
		for session in sessions:
			data = np.asarray(f[animal][session])
			all_sessions.append(data)
			by_session_late.append(data)
		by_animal_late.append(np.nanmean(np.asarray(all_sessions),axis=0))
	f.close()
	##now plot
	by_session_early = np.asarray(by_session_early)
	by_session_late = np.asarray(by_session_late)
	by_animal_early = np.asarray(by_animal_early)
	by_animal_late = np.asarray(by_animal_late)
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	cax1 = ax1.imshow(np.nanmean(by_session_early,axis=0).T,aspect='auto',
		origin='lower',extent=(-2,2,0,150),vmin=vmin,vmax=vmax)
	ax1.axvline(x=0,color='white',linestyle='dashed',linewidth=2)
	# cb = plt.colorbar(cax,label='coherence')
	ax1.set_xlabel("Time to rewarded target",fontsize=16)
	ax1.set_ylabel("Frequency, Hz",fontsize=16)
	ax1.set_title("Early training sessions",fontsize=16)
	#ax1.set_ylim(0,60)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	##for the late session data
	cax2 = ax2.imshow(np.nanmean(by_session_late,axis=0).T,aspect='auto',
		origin='lower',extent=(-2,2,0,150),vmin=vmin,vmax=vmax)
	ax2.axvline(x=0,color='white',linestyle='dashed',linewidth=2)
	cbaxes = fig.add_axes([0.85, 0.08, 0.08, 0.85]) 
	cb = plt.colorbar(cax2,cax=cbaxes)
	cb.set_label(label='Power',fontsize=16)
	ax2.set_xlabel("Time to rewarded target",fontsize=16)
	# ax2.set_ylabel("Frequency, Hz",fontsize=16)
	ax2.set_yticks([])
	for tick in ax2.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	ax2.set_title("Late training sessions",fontsize=16)
	#ax2.set_ylim(0,60)
	fig.suptitle("Jaws animals LFP power",fontsize=16,weight='bold')
	##now plot just the gamma band
	fig,ax = plt.subplots(1)
	gamma_early = by_session_early[:,:,10:20].mean(axis=2)
	gamma_late = by_session_late[:,:,10:20].mean(axis=2)
	early_mean = gamma_early.mean(axis=0)
	early_sem = stats.sem(gamma_early,axis=0)
	late_mean = gamma_late.mean(axis=0)
	late_sem = stats.sem(gamma_late,axis=0)
	x = np.linspace(-2,2,by_session_early.shape[1])
	ax.plot(x,early_mean,linewidth=2,color='r',label='Stim on')
	ax.plot(x,late_mean,linewidth=2,color='k',label='Stim off')
	ax.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='r',alpha=0.5)
	ax.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='k',alpha=0.5)
	ax.set_ylabel("Normalized gamma power",fontsize=16)
	ax.set_xlabel("Time to rewarded target",fontsize=16)
	ax.set_title("Gamma power during rewarded trials",fontsize=16,weight='bold')
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax.axvline(0,linewidth=2,color='k',linestyle='dashed')
	ax.legend()

##same as above but not averaged over trials
def plot_lfp_specgram2():
	source_file =  r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\mouse_lfp_spec_by_trial2.hdf5"
	f = h5py.File(source_file,'r')
	early_range = np.array([4,5])
	late_range = np.array([11,12])
	vmin=0.85
	vmax=1.15
	by_session_early = []
	by_session_late = []
	for animal in [x for x in f.keys() if not x == 'all_sessions']:
		if early_range == None:
			sessions = f[animal].keys()
		else:
			sessions = [x for x in f[animal].keys() if int(x[-6:-4]) in early_range]
		for session in sessions:
			data = np.asarray(f[animal][session])
			by_session_early.append(data)
	for animal in [x for x in f.keys() if not x == 'all_sessions']:
		if late_range == None:
			sessions = f[animal].keys()
		else:
			sessions = [x for x in f[animal].keys() if int(x[-6:-4]) in late_range]
		for session in sessions:
			data = np.asarray(f[animal][session])
			by_session_late.append(data)
	f.close()
	##now plot
	by_session_early = np.concatenate(by_session_early,axis=2).transpose(2,0,1)
	by_session_late = np.concatenate(by_session_late,axis=2).transpose(2,0,1)
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	cax1 = ax1.imshow(np.nanmean(by_session_early,axis=0).T,aspect='auto',
		origin='lower',extent=(-2,2,0,150),vmin=vmin,vmax=vmax)
	ax1.axvline(x=0,color='white',linestyle='dashed',linewidth=2)
	# cb = plt.colorbar(cax,label='coherence')
	ax1.set_xlabel("Time to rewarded target",fontsize=16)
	ax1.set_ylabel("Frequency, Hz",fontsize=16)
	ax1.set_title("Early training sessions",fontsize=16)
	ax1.set_ylim(0,100)
	ax1.set_xlim(-2,0.5)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	##for the late session data
	cax2 = ax2.imshow(np.nanmean(by_session_late,axis=0).T,aspect='auto',
		origin='lower',extent=(-2,2,0,150),vmin=vmin,vmax=vmax)
	ax2.axvline(x=0,color='white',linestyle='dashed',linewidth=2)
	cbaxes = fig.add_axes([0.85, 0.08, 0.08, 0.85]) 
	cb = plt.colorbar(cax2,cax=cbaxes)
	cb.set_label(label='Power',fontsize=16)
	ax2.set_xlabel("Time to rewarded target",fontsize=16)
	# ax2.set_ylabel("Frequency, Hz",fontsize=16)
	ax2.set_yticks([])
	for tick in ax2.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	ax2.set_title("Late training sessions",fontsize=16)
	ax2.set_ylim(0,100)
	ax2.set_xlim(-2,0.5)
	fig.suptitle("Jaws animals LFP power",fontsize=16,weight='bold')
	##now plot just the gamma band
	fig,ax = plt.subplots(1)
	gamma_early = by_session_early[:,:,5:15].mean(axis=2)
	gamma_late = by_session_late[:,:,5:15].mean(axis=2)
	early_mean = gamma_early.mean(axis=0)
	early_sem = np.std(gamma_early,axis=0)/np.sqrt(gamma_early.shape[0])
	late_mean = gamma_late.mean(axis=0)
	late_sem = np.std(gamma_late,axis=0)/np.sqrt(gamma_late.shape[0])
	x = np.linspace(-2,2,by_session_early.shape[1])
	ax.plot(x,early_mean,linewidth=2,color='r',label='Stim on')
	ax.plot(x,late_mean,linewidth=2,color='k',label='Stim off')
	ax.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='r',alpha=0.5)
	ax.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='k',alpha=0.5)
	ax.set_ylabel("Normalized gamma power",fontsize=16)
	ax.set_xlabel("Time to rewarded target",fontsize=16)
	ax.set_title("Gamma power during rewarded trials",fontsize=16,weight='bold')
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax.axvline(0,linewidth=2,color='k',linestyle='dashed')
	ax.legend()
	ax.set_xlim(-1.5,0.5)

def plot_mouse_lfp_spectrum():
	source_file =  r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\mouse_spectrums.hdf5"
	f = h5py.File(source_file,'r')
	early_range = np.arange(5,7)
	late_range = np.arange(7,13)
	by_animal_early = []
	by_animal_late = []
	by_session_early = []
	by_session_late = []
	for animal in [x for x in f.keys() if not x == 'all_sessions']:
		all_sessions = []
		if early_range == None:
			sessions = f[animal].keys()
		else:
			sessions = [x for x in f[animal].keys() if int(x[-6:-4]) in early_range]
		for session in sessions:
			data = np.asarray(f[animal][session])
			all_sessions.append(data)
			by_session_early.append(data)
		by_animal_early.append(np.nanmean(np.asarray(all_sessions),axis=0))
	for animal in [x for x in f.keys() if not x == 'all_sessions']:
		all_sessions = []
		if late_range == None:
			sessions = f[animal].keys()
		else:
			sessions = [x for x in f[animal].keys() if int(x[-6:-4]) in late_range]
		for session in sessions:
			data = np.asarray(f[animal][session])
			all_sessions.append(data)
			by_session_late.append(data)
		by_animal_late.append(np.nanmean(np.asarray(all_sessions),axis=0))
	f.close()
	##now plot
	by_session_early = np.asarray(by_session_early)
	by_session_late = np.asarray(by_session_late)
	by_animal_early = np.asarray(by_animal_early)
	by_animal_late = np.asarray(by_animal_late)
	fig,ax = plt.subplots(1)
	x = np.linspace(0,150,by_session_early.shape[1])
	early_mean = by_session_early.mean(axis=0)
	early_sem = stats.sem(by_session_early,axis=0)
	late_mean = by_session_late.mean(axis=0)
	late_sem = stats.sem(by_session_late,axis=0)
	ax.plot(x,early_mean,linewidth=2,color='r',label='Stim on')
	ax.plot(x,late_mean,linewidth=2,color='k',label='Stim off')
	ax.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='r',alpha=0.5)
	ax.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='k',alpha=0.5)
	ax.set_ylabel("LFP power",fontsize=16)
	ax.set_xlabel("Frequency, Hz",fontsize=16)
	ax.set_title("LFP power during rewarded trials",fontsize=16,weight='bold')
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax.set_yscale("log")
		#ax.set_xlim(0,100)

def save_ff_cohgram_data():
	##define some gobal parameters
	sig1 = 'PLC_lfp'
	sig2 = 'V1_lfp'
	sig_type = 'lfp'
	target = 't1'
	animal_list = ["R11","R13","V02","V03","V04","V05","V11","V13"]
	window = [6000,6000]
	session_range = None	
	root_dir = r"D:\Ryan\V1_BMI"
	save_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\V1_DS_ffc.hdf5"
	if animal_list is None:
		animal_list = ru.animals.keys()
	for animal in animal_list:
		if session_range is None:
			session_list = ru.animals[animal][1].keys()
		else: 
			session_list = [x for x in ru.animals[animal][1].keys() if int(x[-6:-4]) in session_range]
		for session in session_list:
			##check to see if we have already computed this data
			f_out = h5py.File(save_file,'r')
			try:
				session_exists = f_out[animal][session]
				f_out.close()
			except KeyError: ##go ahead and get the data
				f_out.close()
				##this should be the path to the plexon file
				plxfile = os.path.join(root_dir,animal,session)
				##try to get the list of names for the signals and the targets
				try:
					sig_list1 = ru.animals[animal][1][session][sig_type][sig1]
				except KeyError:
					sig_list1 = []
				try:
					sig_list2 = ru.animals[animal][1][session][sig_type][sig2]
				except KeyError:
					sig_list2 = []
				try:
					event = ru.animals[animal][1][session]['events'][target][0]
				except KeyError:
					event = 0
				##if we do have everything we need, continue
				if (len(sig_list1)>0 and len(sig_list2)>0 and event != 0):
					print("working on "+animal+" "+session)
					##open the raw plexon data file
					raw_data = plxread.import_file(plxfile,AD_channels=range(1,97),save_wf=False,
						import_unsorted=False,verbose=False)
					##this will be our list of lfp pairs to send for parallel coherence calculations
					trial_data = []
					target_ts = raw_data[event]*1000.0
					##now process each signal for this session to get the time-locked traces for each trial:
					for i in range(len(sig_list1)):
						signame1 = sig_list1[i]
						tempdata1 = raw_data[signame1]
						sigts1 = raw_data[signame1+"_ts"]
						#convert the ad ts to samples, and integers for indexing
						sigts1 = np.ceil((sigts1*1000)).astype(int)
						sigdata1 = np.zeros(sigts1.shape[0]+1000)
						sigdata1[sigts1] = tempdata1
						traces1 = get_data_window_lfp(sigdata1,target_ts,window[0],window[1])
						for j in range(len(sig_list2)):
							signame2 = sig_list2[j]
							tempdata2 = raw_data[signame2]
							sigts2 = raw_data[signame2+"_ts"]
							#convert the ad ts to samples, and integers for indexing
							sigts2 = np.ceil((sigts2*1000)).astype(int)
							sigdata2 = np.zeros(sigts2.shape[0]+1000)
							sigdata2[sigts2] = tempdata2
							traces2 = get_data_window_lfp(sigdata2,target_ts,window[0],window[1])
							trial_data.append([traces1,traces2])
					pool = mp.Pool(processes=mp.cpu_count())
					async_result = pool.map_async(ss.mp_cohgrams,trial_data)
					pool.close()
					pool.join()
					cohgrams = async_result.get()
					f_out = h5py.File(save_file,'a')
					try:
						a_group = f_out[animal]
					except KeyError:
						a_group = f_out.create_group(animal)
					a_group.create_dataset(session, data=np.asarray(cohgrams))
					f_out.close()
	print("Done!")

def save_sf_cohgram_data():
	##define some gobal parameters
	##MAKE SURE TO ADJUST LFP RANGES WHEN SWITCHING BETWEEN MOUSE AND RAT
	sig1 = 'PLC_units'
	sig2 = 'V1_lfp'
	sig_type1 = 'units'
	sig_type2 = 'lfp'
	target = 't1'
	animal_list = ["R13","R11"]
	window = [6000,6000]
	session_range = np.arange(0,14)	
	root_dir = r"F:\data"
	save_file = r"F:\data\NatureNeuro\rebuttal\data\PLC_V1_sfc.hdf5"
	if animal_list is None:
		animal_list = ru.animals.keys()
	for animal in animal_list:
		if session_range is None:
			session_list = ru.animals[animal][1].keys()
		else: 
			session_list = [x for x in ru.animals[animal][1].keys() if int(x[-6:-4]) in session_range]
		for session in session_list:
			##check to see if we have already computed this data
			f_out = h5py.File(save_file,'a')
			try:
				session_exists = f_out[animal][session]
				f_out.close()
			except KeyError: ##go ahead and get the data
				f_out.close()
				##this should be the path to the plexon file
				plxfile = os.path.join(root_dir,animal,session)
				##try to get the list of names for the signals and the targets
				try:
					sig_list1 = ru.animals[animal][1][session][sig_type1][sig1]
				except KeyError:
					sig_list1 = []
				try:
					sig_list2 = ru.animals[animal][1][session][sig_type2][sig2]
				except KeyError:
					sig_list2 = []
				try:
					event = ru.animals[animal][1][session]['events'][target][0]
				except KeyError:
					event = 0
				##if we do have everything we need, continue
				if (len(sig_list1)>0 and len(sig_list2)>0 and event != 0):
					print("working on "+animal+" "+session)
					##open the raw plexon data file
					raw_data = plxread.import_file(plxfile,AD_channels=range(1,97),save_wf=False,
						import_unsorted=False,verbose=False)
					##this will be our list of lfp pairs to send for parallel coherence calculations
					trial_data = []
					target_ts = raw_data[event]*1000.0
					##get the info about the duration of this session
					lfp_id = ru.animals[animal][1][session]['lfp']['V1_lfp'][0]
					duration = int((np.ceil(raw_data[lfp_id+'_ts'].max()*1000)/100)*100)+1
					##now process each signal for this session to get the time-locked traces for each trial:
					for i in range(len(sig_list1)):
						signame1 = sig_list1[i]
						tempdata1 = raw_data[signame1]
						if sig_type1 == 'lfp':
							sigts1 = raw_data[signame1+"_ts"]
							#convert the ad ts to samples, and integers for indexing
							sigts1 = np.ceil((sigts1*1000)).astype(int)
							sigdata1 = np.zeros(sigts1.shape[0]+1000)
							sigdata1[sigts1] = tempdata1
						elif sig_type1 == 'units':
							sigdata1 = tempdata1*1000.0
							sigdata1 = np.histogram(sigdata1,bins=duration,range=(0,duration))
							sigdata1 = sigdata1[0].astype(bool).astype(int)
						traces1 = get_data_window_lfp(sigdata1,target_ts,window[0],window[1])
						for j in range(len(sig_list2)):
							signame2 = sig_list2[j]
							tempdata2 = raw_data[signame2]
							if sig_type2 == 'lfp':
								sigts2 = raw_data[signame2+"_ts"]
								#convert the ad ts to samples, and integers for indexing
								sigts2 = np.ceil((sigts2*1000)).astype(int)
								sigdata2 = np.zeros(sigts2.shape[0]+1000)
								sigdata2[sigts2] = tempdata2
							elif sig_type2 == 'units':
								sigdata2 = tempdata2*1000.0
								sigdata2 = np.histogram(sigdata2,bins=duration,range=(0,duration))
								sigdata2 = sigdata2[0].astype(bool).astype(int)
							traces2 = get_data_window_lfp(sigdata2,target_ts,window[0],window[1])
							if sig_type1 == 'units':
								trial_data.append([traces1,traces2])
							elif sig_type2 == 'units':
								trial_data.append([traces2,traces1])
					pool = mp.Pool(processes=mp.cpu_count())
					async_result = pool.map_async(SFC.mp_sfc,trial_data)
					pool.close()
					pool.join()
					cohgrams = async_result.get()
					f_out = h5py.File(save_file,'a')
					try:
						a_group = f_out[animal]
					except KeyError:
						a_group = f_out.create_group(animal)
					a_group.create_dataset(session, data=np.asarray(cohgrams))
					f_out.close()
	print("Done!")

def get_ensemble_correlations_mouse():
	##define some gobal parameters
	animal_list = ["m11","m13","m15","m17"]
	window = [120000,30000]
	tau = 20
	dt = 1
	session_range = None
	sig_type = "units"
	root_dir = r"L:\data"
	save_file = r"L:\data\NatureNeuro\rebuttal\data\mouse_correlations.hdf5"
	if animal_list is None:
		animal_list = ru.animals.keys()
	for animal in animal_list:
		if session_range is None:
			session_list = ru.animals[animal][1].keys()
		else: 
			session_list = [x for x in ru.animals[animal][1].keys() if int(x[-6:-4]) in session_range]
		for session in session_list:
			##this should be the path to the plexon file
			plxfile = os.path.join(root_dir,animal,session)
			##try to get the list of names for the signals and the targets
			try:
				e1_list = ru.animals[animal][1][session][sig_type]["e1_units"]
			except KeyError:
				e1_list = []
			try:
				e2_list = ru.animals[animal][1][session][sig_type]["e2_units"]
			except KeyError:
				e2_list = []
			try:
				ind_list = ru.animals[animal][1][session][sig_type]["V1_units"]
			except KeyError:
				ind_list = []
			##if we do have everything we need, continue
			if (len(e1_list)>0 and len(e2_list)>0 and len(ind_list)>0):
				print("working on "+animal+" "+session)
				##open the raw plexon data file
				raw_data = plxread.import_file(plxfile,AD_channels=range(100,200),save_wf=False,
					import_unsorted=False,verbose=False)
				##this will be our list of correlation timecourses
				e1_data = []
				e2_data = []
				ind_data = []
				##get the info about the duration of this session
				lfp_id = ru.animals[animal][1][session]['lfp']['V1_lfp'][0]
				duration = int((np.ceil(raw_data[lfp_id+'_ts'].max()*1000)/100)*100)+1
				##now process each set of units for this session
				##starting with E1 units:
				##a list of tuples containing all the pairwise combinations of e1 units
				##(necessary because some sessions have more than 2 ensemble units)
				unit_combinations = list(itertools.combinations(e1_list,2))
				for c in unit_combinations:
					##extract and process the data from the datafile
					unit1 = raw_data[c[0]]*1000.0
					unit1 = np.histogram(unit1,bins=duration,range=(0,duration))
					unit1 = unit1[0].astype(bool).astype(int)
					####
					unit2 = raw_data[c[1]]*1000.0
					unit2 = np.histogram(unit2,bins=duration,range=(0,duration))
					unit2 = unit2[0].astype(bool).astype(int)
					##now do a windowed correlation analysis
					result = ss.window_corr(unit1,unit2,window,tau,dt)
					e1_data.append(result)
				##repeat for e2 units
				unit_combinations = list(itertools.combinations(e2_list,2))
				for c in unit_combinations:
					##extract and process the data from the datafile
					unit1 = raw_data[c[0]]*1000.0
					unit1 = np.histogram(unit1,bins=duration,range=(0,duration))
					unit1 = unit1[0].astype(bool).astype(int)
					####
					unit2 = raw_data[c[1]]*1000.0
					unit2 = np.histogram(unit2,bins=duration,range=(0,duration))
					unit2 = unit2[0].astype(bool).astype(int)
					##now do a windowed correlation analysis
					result = ss.window_corr(unit1,unit2,window,tau,dt)
					e2_data.append(result)
				##repeat for indirect units
				unit_combinations = list(itertools.combinations(ind_list,2))
				for c in unit_combinations:
					##extract and process the data from the datafile
					unit1 = raw_data[c[0]]*1000.0
					unit1 = np.histogram(unit1,bins=duration,range=(0,duration))
					unit1 = unit1[0].astype(bool).astype(int)
					####
					unit2 = raw_data[c[1]]*1000.0
					unit2 = np.histogram(unit2,bins=duration,range=(0,duration))
					unit2 = unit2[0].astype(bool).astype(int)
					##now do a windowed correlation analysis
					result = ss.window_corr(unit1,unit2,window,tau,dt)
					ind_data.append(result)
				f_out = h5py.File(save_file,'a')
				try:
					a_group = f_out[animal]
				except KeyError:
					a_group = f_out.create_group(animal)
				s_group = a_group.create_group(session)
				s_group.create_dataset("E1",data = np.asarray(e1_data))
				s_group.create_dataset("E2",data = np.asarray(e2_data))
				s_group.create_dataset("indirect",data = np.asarray(ind_data))
				f_out.close()
	print("Done!")


def plot_mouse_correlations():
	##here we want to do 2 types of plots. 
	#1) mean corr across sessions
	#2) corr within sessions for jaws and non-jaws days
	##star by opening the data file 
	source_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\mouse_correlations.hdf5"
	f = h5py.File(source_file,'r')
	##create some lists to store the different types of data
	all_e1 = []
	all_e2 = []
	all_indirect = []
	animal_e1 = []
	animal_e2 = []
	animal_indirect = []
	across_sessions_e1 = []
	across_sessions_e2 = []
	across_sessions_indirect = []
	for animal in f.keys():
		e1_means = []
		e2_means = []
		indirect_means = []
		e1_sessions = []
		e2_sessions = []
		indirect_sessions = []
		sessions = f[animal].keys()
		for session in sessions:
			e1_data = np.asarray(f[animal][session]['E1']).squeeze()
			e2_data = np.asarray(f[animal][session]['E2']).squeeze()
			indirect_data = np.asarray(f[animal][session]['indirect']).squeeze()
			##get the mean over the whole session
			e1_means.append(e1_data.mean())
			e2_means.append(e2_data.mean())
			indirect_means.append(indirect_data.mean())
			##get the mean across the session
			if len(e1_data.shape) > 1:
				e1_sessions.append(e1_data.mean(axis=0))
				all_e1.append(e1_data.mean(axis=0))
			else:
				e1_sessions.append(e1_data)
				all_e1.append(e1_data)
				##
			if len(e2_data.shape) > 1:
				e2_sessions.append(e2_data.mean(axis=0))
				all_e2.append(e2_data.mean(axis=0))
			else:
				e2_sessions.append(e2_data)
				all_e2.append(e2_data)
				##
			if len(indirect_data.shape) > 1:
				indirect_sessions.append(indirect_data.mean(axis=0))
				all_indirect.append(indirect_data.mean(axis=0))
			else:
				indirect_sessions.append(indirect_data)
				all_indirect.append(indirect_data)
		##add this data to the master lists
		across_sessions_e1.append(np.asarray(e1_means))
		across_sessions_e2.append(np.asarray(e2_means))
		across_sessions_indirect.append(np.asarray(indirect_means))
		animal_e1.append(np.asarray(equalize_arrs(e1_sessions)).mean(axis=0))
		animal_e2.append(np.asarray(equalize_arrs(e2_sessions)).mean(axis=0))
		animal_indirect.append(np.asarray(equalize_arrs(indirect_sessions)).mean(axis=0))
	f.close()
	##now we need to equalize the lengths of our arrays
	all_e1 = np.asarray(equalize_arrs(all_e1))
	all_e2 = np.asarray(equalize_arrs(all_e2))
	all_indirect = np.asarray(equalize_arrs(all_indirect))
	animal_e1 = np.asarray(equalize_arrs(animal_e1))
	animal_e2 = np.asarray(equalize_arrs(animal_e2))
	animal_indirect = np.asarray(equalize_arrs(animal_indirect))
	across_sessions_e1 = np.asarray(equalize_arrs(across_sessions_e1))
	across_sessions_e2 = np.asarray(equalize_arrs(across_sessions_e2))
	across_sessions_indirect = np.asarray(equalize_arrs(across_sessions_indirect))
	###finally, we can plot these things
	session_data = [all_e1,all_e2,all_indirect]
	animal_data = [animal_e1,animal_e2,animal_indirect]
	across_data = [across_sessions_e1,across_sessions_e2,across_sessions_indirect]
	colors = ['g','b','k']
	labels = ['E1','E2','Indirect']
	##start with the within session data
	x = np.linspace(0,60,all_e1.shape[1])
	fig,ax = plt.subplots(1)
	for i, dataset in enumerate(session_data):
		mean = np.nanmean(dataset,axis=0)
		sem = np.nanstd(dataset,axis=0)/np.sqrt(dataset.shape[0])
		ax.plot(x,mean,linewidth=2,color=colors[i],label=labels[i])
		ax.fill_between(x,mean-sem,mean+sem,color=colors[i],alpha=0.5)
	ax.set_xlabel("Time in session, mins",fontsize=16)
	ax.set_ylabel("Correlation coefficient",fontsize=16)
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax.legend()
	fig.suptitle("Correlations by session")
	##now look at it by animal
	x = np.linspace(0,60,animal_e1.shape[1])
	fig,ax = plt.subplots(1)
	for i, dataset in enumerate(animal_data):
		mean = np.nanmean(dataset,axis=0)
		sem = np.nanstd(dataset,axis=0)/np.sqrt(dataset.shape[0])
		ax.plot(x,mean,linewidth=2,color=colors[i],label=labels[i])
		ax.fill_between(x,mean-sem,mean+sem,color=colors[i],alpha=0.5)
	ax.set_xlabel("Time in session, mins",fontsize=16)
	ax.set_ylabel("Correlation coefficient",fontsize=16)
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax.legend()
	fig.suptitle("Correlations by animal")
	##now do the across session plot
	fig,ax = plt.subplots(1)
	for i, dataset in enumerate(across_data):
		x = np.arange(1,dataset.shape[1]+1)
		mean = np.nanmean(dataset,axis=0)
		sem = np.nanstd(dataset,axis=0)/np.sqrt(dataset.shape[0])
		ax.plot(x,mean,linewidth=2,color=colors[i],label=labels[i])
		ax.fill_between(x,mean-sem,mean+sem,color=colors[i],alpha=0.5)
	ax.set_xlabel("Traning day",fontsize=16)
	ax.set_ylabel("Mean correlation coefficient",fontsize=16)
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax.legend()
	ax.set_title("Correlations across training",fontsize=16)
	ax.plot(np.arange(1,8),np.ones(7)*-0.045,linewidth=4,color='r')
	ax.text(3,-0.04,"Jaws",fontsize=14)
	ax.set_xlim(0,13)

def plot_cohgram_data():
	source_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\mouse_e1_V1_sfc.hdf5"
	f = h5py.File(source_file,'r')
	early_range = np.array([6])
	late_range = np.array([10])
	vmin=0.05
	vmax=0.1
	by_animal_early = []
	by_animal_late = []
	by_session_early = []
	by_session_late = []
	for animal in f.keys():
		all_sessions = []
		if early_range == None:
			sessions = f[animal].keys()
		else:
			sessions = [x for x in f[animal].keys() if int(x[-6:-4]) in early_range]
		for session in sessions:
			data = np.asarray(f[animal][session])
			all_sessions.append(np.nanmean(data,axis=0))
			by_session_early.append(np.nanmean(data,axis=0))
		by_animal_early.append(np.nanmean(np.asarray(all_sessions),axis=0))
	for animal in f.keys():
		all_sessions = []
		if late_range == None:
			sessions = f[animal].keys()
		else:
			sessions = [x for x in f[animal].keys() if int(x[-6:-4]) in late_range]
		for session in sessions:
			data = np.asarray(f[animal][session])
			all_sessions.append(np.nanmean(data,axis=0))
			by_session_late.append(np.nanmean(data,axis=0))
		by_animal_late.append(np.nanmean(np.asarray(all_sessions),axis=0))
	f.close()
	##now plot
	by_session_early = np.asarray(by_session_early)
	by_session_late = np.asarray(by_session_late)
	by_animal_early = np.asarray(by_animal_early)
	by_animal_late = np.asarray(by_animal_late)
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	cax1 = ax1.imshow(np.nanmean(by_animal_early,axis=0).T,aspect='auto',
		origin='lower',extent=(-6,6,0,100),vmin=vmin,vmax=vmax)
	ax1.axvline(x=0,color='white',linestyle='dashed',linewidth=2)
	# cb = plt.colorbar(cax,label='coherence')
	ax1.set_xlabel("Time to rewarded target",fontsize=16)
	ax1.set_ylabel("Frequency, Hz",fontsize=16)
	ax1.set_title("Early training sessions",fontsize=16)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	##for the late session data
	cax2 = ax2.imshow(np.nanmean(by_animal_late,axis=0).T,aspect='auto',
		origin='lower',extent=(-6,6,0,100),vmin=vmin,vmax=vmax)
	ax2.axvline(x=0,color='white',linestyle='dashed',linewidth=2)
	cbaxes = fig.add_axes([0.85, 0.08, 0.08, 0.85]) 
	cb = plt.colorbar(cax2,cax=cbaxes)
	cb.set_label(label='coherence',fontsize=16)
	ax2.set_xlabel("Time to rewarded target",fontsize=16)
	# ax2.set_ylabel("Frequency, Hz",fontsize=16)
	ax2.set_yticks([])
	for tick in ax2.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	ax2.set_title("Late training sessions",fontsize=16)
	fig.suptitle("PrL-V1 field-field coherence",fontsize=16,weight='bold')

##to get the time-to target latencies for Jaws animals
def get_target_latencies():
	##define some gobal parameters
	animal_list = None
	session_range = None
	root_dir = r"D:\Ryan\V1_BMI"
	save_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\all_target_latencies.hdf5"
	if animal_list is None:
		animal_list = ru.animals.keys()
	for animal in animal_list:
		if session_range is None:
			session_list = ru.animals[animal][1].keys()
		else: 
			session_list = [x for x in ru.animals[animal][1].keys() if int(x[-6:-4]) in session_range]
		for session in session_list:
			##this should be the path to the plexon file
			plxfile = os.path.join(root_dir,animal,session)
			##try to get the list of names for the signals and the targets
			try:
				t1_id = ru.animals[animal][1][session]['events']['t1'][0]
			except KeyError:
				t1_id = 0
			try:
				t2_id = ru.animals[animal][1][session]['events']['t2'][0]
			except KeyError:
				t2_id = 0
			try:
				miss_id = ru.animals[animal][1][session]['events']['miss'][0]
			except KeyError:
				miss_id = 0
			##if we do have everything we need, continue
			if (t1_id != 0):
				print("working on "+animal+" "+session)
				##open the raw plexon data file
				raw_data = plxread.import_file(plxfile,AD_channels=range(1,97),save_wf=False,
					import_unsorted=False,verbose=False)
				##get the timestamps of all the events
				try:
					t1_ts = raw_data[t1_id]*1000.0
				except KeyError:
					print("No t1's for this session")
					t1_ts = np.array([])
				try:
					t2_ts = raw_data[t2_id]*1000.0
				except KeyError:
					print("No t2's for this session")
					t2_ts = np.array([])
				try:
					miss_ts = raw_data[miss_id]*1000.0
				except KeyError:
					print("No misses for this file")
					miss_ts = np.array([])
				###now get 2 arrays, one of all the timestamps and one of all the timestamp IDs
				ids = np.concatenate((np.full(t1_ts.size,"t1",dtype='S2'),
									  np.full(t2_ts.size,"t2",dtype='S2'),
									  np.full(miss_ts.size,"miss_ts",dtype='S4')))
				ts = np.concatenate((t1_ts,t2_ts,miss_ts))
				##order all the timestamps and the ts id's
				idx = np.argsort(ts)
				ids = ids[idx]
				ts = ts[idx]
				##now go through and get the time to target for each target type
				t1_times = []
				t2_times = []
				miss_times = []
				t = 0
				while t < ts.size-1:
					start_ts = ts[t]
					##now we want to know what the outcome of this trial was
					if ids[t+1] == 't1': ##case where this was a t1 trial
						trial_duration = ts[t+1]-start_ts
						t1_times.append(trial_duration)
						t += 1
					elif ids[t+1] == 't2': ##case where this was a t2 trial
						trial_duration = ts[t+1]-start_ts
						t2_times.append(trial_duration)
						t += 1
					elif ids[t+1] == 'miss': ##case where this was a miss trial
						trial_duration = ts[t+1]-start_ts
						miss_times.append(trial_duration)
						t += 1
					else:
						print("unknown trial type: "+ids[t+1])
						t+=1
				t1_times = np.asarray(t1_times)
				t2_times = np.asarray(t2_times)
				miss_times = np.asarray(miss_times)
				f_out = h5py.File(save_file,'a')
				try: 
					a_group = f_out[animal]
				except KeyError:
					a_group = f_out.create_group(animal)
				s_group = a_group.create_group(session)
				s_group.create_dataset("t1",data=t1_times)
				s_group.create_dataset("t2",data=t2_times)
				s_group.create_dataset("miss",data=miss_times)
				f_out.close()
	print("Done")

def plot_target_latencies():
	##define global parameters
	target = 't1'
	early_range = [1,2,3,4]
	late_range = [5,6,7]
	animal_list = ['V01','V02','V03','V04','V05','V11','V13','R11','R13','V14']
	source_file = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\all_target_latencies.hdf5"
	f = h5py.File(source_file,'r')
	if animal_list is None:
		animals = f.keys()
	else:
		animals = animal_list
	all_trials_early = []
	all_trials_late = []
	animal_average_early = []
	animal_average_late = []
	for animal in animals:
		animal_data_early = []
		animal_data_late = []
		sessions_early = [x for x in f[animal].keys() if int(x[-6:-4]) in early_range]
		sessions_late = [x for x in f[animal].keys() if int(x[-6:-4]) in late_range]
		for session in sessions_early:
			data = np.asarray(f[animal][session][target])
			all_trials_early.append(data)
			animal_data_early.append(np.nanmean(data))
		for session in sessions_late:
			data = np.asarray(f[animal][session][target])
			all_trials_late.append(data)
			animal_data_late.append(np.nanmean(data))
		animal_average_early.append(np.asarray(np.nanmean(animal_data_early)))
		animal_average_late.append(np.asarray(np.nanmean(animal_data_late)))
	animal_average_early = np.nan_to_num(np.asarray(animal_average_early))/1000.0
	animal_average_late = np.nan_to_num(np.asarray(animal_average_late))/1000.0
	all_trials_early = np.nan_to_num(np.concatenate(all_trials_early))/1000.0
	all_trials_late = np.concatenate(all_trials_late)/1000.0
	f.close()
	##look at by animal first
	early_mean = animal_average_early.mean()
	early_sem = stats.sem(animal_average_early)
	late_mean = animal_average_late.mean()
	late_sem = stats.sem(animal_average_late)
	means = np.array([early_mean,late_mean])
	sems = np.array([early_sem,late_sem])
	fig,ax = plt.subplots(1)
	x = [1,2]
	xerr = [0.1,0.1]
	ax.errorbar(x,means,yerr=sems,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	all_together = np.vstack([animal_average_early,animal_average_late])
	for i in range(all_together.shape[1]):
		plt.plot(x,all_together[:,i],linewidth=2,marker='o',color='k',alpha=0.5)
	ax.set_xticks([1,2])
	ax.set_xticklabels(["Early","Late"])
	ax.set_ylabel("Time to target",fontsize=14)
	if target == 't1':
		ax.set_title("Latencies to rewarded targets",fontsize=14)
	else:
		ax.set_title("Latencies to unrewarded targets",fontsize=14)
	#ax.set_ylim(8,19)
	pval_animals = stats.ttest_rel(animal_average_early,animal_average_late)[1]
	tval_animals = stats.ttest_rel(animal_average_early,animal_average_late)[0]
	print("mean early = "+str(early_mean))
	print("mean late = "+str(late_mean))
	print("pval = "+str(pval_animals))
	print("tval = "+str(tval_animals))
	##now look at all trials pooled
	early_mean = all_trials_early.mean()
	early_sem = stats.sem(all_trials_early)
	late_mean = all_trials_late.mean()
	late_sem = stats.sem(all_trials_late)
	means = np.array([early_mean,late_mean])
	sems = np.array([early_sem,late_sem])
	fig,ax = plt.subplots(1)
	x = [1,2]
	xerr = [0.1,0.1]
	for i in range(all_trials_early.shape[0]):
		plt.plot(1,all_trials_early[i],linewidth=2,marker='o',color='k',alpha=0.2)
	for i in range(all_trials_late.shape[0]):
		plt.plot(2,all_trials_late[i],linewidth=2,marker='o',color='k',alpha=0.2)
	ax.errorbar(x,means,yerr=sems,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	ax.set_xticks([1,2])
	ax.set_xticklabels(["Early","Late"])
	ax.set_ylabel("Time to target",fontsize=14)
	ax.set_title("Latencies to rewarded targets",fontsize=14)
	pval_all = stats.ttest_ind(all_trials_early,all_trials_late)[1]
	tval_all = stats.ttest_ind(all_trials_early,all_trials_late)[0]
	data = [all_trials_early,all_trials_late]
	fig,ax = plt.subplots(1)
	ax.boxplot(data)
	ax.set_ylabel("Time to target (s)",fontsize=14)
	ax.set_xticklabels(["Early","Late"])
	ax.set_title("Latencies to rewarded targets",fontsize=14)
	print("mean early = "+str(early_mean))
	print("mean late = "+str(late_mean))
	print("pval = "+str(pval_all))
	print("tval = "+str(tval_all))

def plot_dms_locked():
	f = h5py.File(r"D:\Ryan\V1_BMI\processed_data\V1_BMI_final\raw_data\dms_spikes_smoothed.hdf5")
	vmin=-2
	vmax=15
	early_data = np.asarray(f['early'])
	late_data = np.asarray(f['late'])
	for i in range(early_data.shape[1]):
		early_data[:,i] = zscore(early_data[:,i])
	for i in range(late_data.shape[1]):
		late_data[:,i] = zscore(late_data[:,i])
	##let's sort by the late session peaks
	late_peaks = np.argmax(abs(late_data),axis=0)
	late_idx = np.argsort(late_peaks)
	early_peaks = np.argmax(abs(early_data),axis=0)
	early_idx = np.argsort(early_peaks)
	late_data = late_data[:,late_idx]
	early_data = early_data[:,early_idx]
	plt.imshow(late_data.T,aspect='auto',vmin=vmin,vmax=vmax,interpolation='none')
	plt.title("late")
	plt.colorbar()
	plt.figure()
	plt.imshow(early_data.T,aspect='auto',vmin=vmin,vmax=vmax,interpolation='none')
	plt.title("early")
	plt.colorbar()
	f.close()
	##each session contains an array, which is 


def get_timelocked_frs():
	unit_types = ['Str_units'] ##the type of units to look at on
	root_dir = "/Volumes/data/V1_BMI"
	animal_list = [x for x in ru.animals.keys() if not x.startswith("m")]
	session_list = None
	window = [10000,4000]
	save_file = "/Volumes/data/V1_BMI/Neuron/data/dms_projection_units_t1"
	f_out = h5py.File(save_file,'a')
	f_out.close()
	for animal in animal_list:
		f_out = h5py.File(save_file,'a')
		try:
			a_group = f_out[animal]
			f_out.close()
		except KeyError:
			a_group = f_out.create_group(animal)
			f_out.close()
			#if session_list is None:
			session_list = ru.animals[animal][1].keys()
			for session in session_list:
				print("Working on "+animal+" "+session)
				filepath = os.path.join(root_dir,animal,session) ##the file path to the session data
				##open the file
				data = plxread.import_file(filepath,AD_channels=range(1,97),save_wf=False,
					import_unsorted=False,verbose=False)
				##what is the duration of this file
				duration = None
				for arr in data.keys():
					if arr.startswith('AD') and arr.endswith('_ts'):
						duration = int(np.ceil((data[arr].max()*1000)/100)*100)+1
						break
				else: print("No A/D timestamp data found!!!")
				##we are going to need the T1 and T2 timestamps fo each file
				try:
					t1_id = ru.animals[animal][1][session]['events']['t1'][0] ##the event name in the plexon file
					t1_ts = data[t1_id]*1000.0
				except KeyError:
					t1_ts = np.array([])
				try:
					t2_id = ru.animals[animal][1][session]['events']['t2'][0] ##the event name in the plexon file
					t2_ts = data[t2_id]*1000.0
				except KeyError:
					t2_ts = np.array([])
				for unit_type in unit_types:
					mean_t1 = []
					mean_t2 = []
					try:
						unit_ids = ru.animals[animal][1][session]['units'][unit_type]
						for unit in unit_ids:
							##get the timestamps for this unit from the datafile
							try:
								unit_ts = data[unit]*1000.0
								##binary transform
								spiketrain = np.histogram(unit_ts,bins=duration,range=(0,duration))
								spiketrain = spiketrain[0].astype(bool).astype(int)
								##now get the mean response for this unit
								if len(t1_ts)>0:
									traces_t1 = get_data_window(spiketrain,t1_ts,
										window[0],window[1])
									if traces_t1 is not None:
										mean_t1.append(traces_t1.mean(axis=1))
								if len(t2_ts)>0:
									traces_t2 = get_data_window(spiketrain,t2_ts,
										window[0],window[1])
									if traces_t2 is not None:
										mean_t2.append(traces_t2.mean(axis=1))
							except KeyError:
								print("No "+unit+" for "+unit_type+" in "+animal+" "+session)
						mean_t1 = np.asarray(mean_t1)
						mean_t2 = np.asarray(mean_t2)
						f_out = h5py.File(save_file,'a')
						a_group = f_out[animal]
						try:
							s_group = a_group[session]
						except KeyError:
							s_group = a_group.create_group(session)
						u_group = s_group.create_group(unit_type)
						if len(t1_ts)>0:
							u_group.create_dataset("t1",data=mean_t1)
						if len(t2_ts)>0:
							u_group.create_dataset("t2",data=mean_t2)
						f_out.close()
					except KeyError:
						print("No "+unit_type+" for "+animal+" "+session)
	print('Done')

def plot_timelocked_frs():
	datafile = r"D:\Ryan\V1_BMI\NatureNeuro\rebuttal\data\rat_spikes_t1_t2.hdf5"
	animal_list = None
	early_range = np.arange(0,4)
	late_range = np.arange(4,50)
	start = -2.5
	stop = 2.5
	abs_z = False
	f = h5py.File(datafile,'r')
	if animal_list is None:
		animal_list = list(f)
	##start with the early data
	e1_t1 = []
	e1_t2 = []
	e2_t1 = []
	e2_t2 = []
	ind_t1 = []
	ind_t2 = []
	for animal in animal_list:
		session_list = [x for x in f[animal].keys() if int(x[-6:-4]) in early_range]
		for session in session_list:
			s_group = f[animal][session]
			##start with e1 units
			try:
				u_group = s_group['e1_units']
				##start first with t1
				try:
					t1_data = np.asarray(u_group['t1']).squeeze()
					if len(t1_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t1_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t1_data[unit,:],[100,50]))
							e1_t1.append(smoothed)
					elif len(t1_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t1_data,[100,50]))
						e1_t1.append(smoothed)
				except KeyError:
					print("No T1 for "+animal+" "+session)
				##now do t2
				try:
					t2_data = np.asarray(u_group['t2']).squeeze()
					if len(t2_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t2_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t2_data[unit,:],[100,50]))
							e1_t2.append(smoothed)
					elif len(t2_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t2_data,[100,50]))
						e1_t2.append(smoothed)
				except KeyError:
					print("No T2 for "+animal+" "+session)
			except KeyError:
				print("No E1 units for "+animal+" "+session)
			##now do e2 units
			try:
				u_group = s_group['e2_units']
				##start first with t1
				try:
					t1_data = np.asarray(u_group['t1']).squeeze()
					if len(t1_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t1_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t1_data[unit,:],[100,50]))
							e2_t1.append(smoothed)
					elif len(t1_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t1_data,[100,50]))
						e2_t1.append(smoothed)
				except KeyError:
					print("No T1 for "+animal+" "+session)
				##now do t2
				try:
					t2_data = np.asarray(u_group['t2']).squeeze()
					if len(t2_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t2_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t2_data[unit,:],[100,50]))
							e2_t2.append(smoothed)
					elif len(t2_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t2_data,[100,50]))
						e2_t2.append(smoothed)
				except KeyError:
					print("No T2 for "+animal+" "+session)
			except KeyError:
				print("No E2 units for "+animal+" "+session)
			##now do indirect units
			try:
				u_group = s_group['V1_units']
				##start first with t1
				try:
					t1_data = np.asarray(u_group['t1']).squeeze()
					if len(t1_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t1_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t1_data[unit,:],[100,50]))
							ind_t1.append(smoothed)
					elif len(t1_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t1_data,[100,50]))
						ind_t1.append(smoothed)
				except KeyError:
					print("No T1 for "+animal+" "+session)
				##now do t2
				try:
					t2_data = np.asarray(u_group['t2']).squeeze()
					if len(t2_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t2_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t2_data[unit,:],[100,50]))
							ind_t2.append(smoothed)
					elif len(t2_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t2_data,[100,50]))
						ind_t2.append(smoothed)
				except KeyError:
					print("No T2 for "+animal+" "+session)
			except KeyError:
				print("No indirect units for "+animal+" "+session)
	f.close()
	##now we have all of the early data
	e1_t1 = np.asarray(e1_t1)
	e1_t2 = np.asarray(e1_t2)
	e2_t1 = np.asarray(e2_t1)
	e2_t2 = np.asarray(e2_t2)
	ind_t1 = np.asarray(ind_t1)
	ind_t2 = np.asarray(ind_t2)
	##take the absolute value, if requested
	if abs_z:
		e1_t1 = abs(e1_t1)
		e1_t2 = abs(e1_t2)
		e2_t1 = abs(e2_t1)
		e2_t2 = abs(e2_t2)
		ind_t1 = abs(ind_t1)
		ind_t2 = abs(ind_t2)
	##now plot this data, for both targets
	fig = plt.figure()
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2,sharey=ax1)
	x = np.linspace(-4,4,e1_t1.shape[1])
	##start with t1
	mean_e1_t1 = np.nanmean(e1_t1,axis=0)
	sem_e1_t1 = np.nanstd(e1_t1,axis=0)/np.sqrt(e1_t1.shape[0])
	mean_e2_t1 = np.nanmean(e2_t1,axis=0)
	sem_e2_t1 = np.nanstd(e2_t1,axis=0)/np.sqrt(e2_t1.shape[0])
	mean_ind_t1 = np.nanmean(ind_t1,axis=0)
	sem_ind_t1 = np.nanstd(ind_t1,axis=0)/np.sqrt(ind_t1.shape[0])
	ax1.plot(x,mean_e1_t1,linewidth=2,color='g',label='E1')
	ax1.fill_between(x,mean_e1_t1-sem_e1_t1,mean_e1_t1+sem_e1_t1,color='g',alpha=0.5)
	ax1.plot(x,mean_e2_t1,linewidth=2,color='b',label='E2')
	ax1.fill_between(x,mean_e2_t1-sem_e2_t1,mean_e2_t1+sem_e2_t1,color='b',alpha=0.5)
	ax1.plot(x,mean_ind_t1,linewidth=2,color='k',label='indirect')
	ax1.fill_between(x,mean_ind_t1-sem_ind_t1,mean_ind_t1+sem_ind_t1,color='k',alpha=0.5)
	ax1.set_title("Rewarded target",fontsize=16)
	ax1.set_ylabel("Zscore",fontsize=16)
	ax1.set_xlabel("Time to target (s)",fontsize=16)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax1.legend()
	##repeat for T2
	mean_e1_t2 = np.nanmean(e1_t2,axis=0)
	sem_e1_t2 = np.nanstd(e1_t2,axis=0)/np.sqrt(e1_t2.shape[0])
	mean_e2_t2 = np.nanmean(e2_t2,axis=0)
	sem_e2_t2 = np.nanstd(e2_t2,axis=0)/np.sqrt(e2_t2.shape[0])
	mean_ind_t2 = np.nanmean(ind_t2,axis=0)
	sem_ind_t2 = np.nanstd(ind_t2,axis=0)/np.sqrt(ind_t2.shape[0])
	ax2.plot(x,mean_e1_t2,linewidth=2,color='g',label='E1')
	ax2.fill_between(x,mean_e1_t2-sem_e1_t2,mean_e1_t2+sem_e1_t2,color='g',alpha=0.5)
	ax2.plot(x,mean_e2_t2,linewidth=2,color='b',label='E2')
	ax2.fill_between(x,mean_e2_t2-sem_e2_t2,mean_e2_t2+sem_e2_t2,color='b',alpha=0.5)
	ax2.plot(x,mean_ind_t2,linewidth=2,color='k',label='indirect')
	ax2.fill_between(x,mean_ind_t2-sem_ind_t2,mean_ind_t2+sem_ind_t2,color='k',alpha=0.5)
	ax2.set_title("Unrewarded target",fontsize=16)
	ax2.set_ylabel("Zscore",fontsize=16)
	ax2.set_xlabel("Time to target (s)",fontsize=16)
	for tick in ax2.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	if abs_z:
		fig.suptitle("Abs. zscore, early sessions",fontsize=16)
	else:
		fig.suptitle("Zscore, early sessions",fontsize=16)
	##finally, get some details about modulation depth
	full_idx = np.linspace(-4,4,e1_t1.shape[1])
	start_idx = np.where(full_idx<start)[0][-1]
	stop_idx = np.where(full_idx>stop)[0][0]
	e1_t1_mod_early = np.nanmax(e1_t1[:,start_idx:stop_idx],axis=1)
	e1_t2_mod_early = np.nanmax(e1_t2[:,start_idx:stop_idx],axis=1)
	e2_t1_mod_early = np.nanmax(e2_t1[:,start_idx:stop_idx],axis=1)
	e2_t2_mod_early = np.nanmax(e2_t2[:,start_idx:stop_idx],axis=1)
	ind_t1_mod_early = np.nanmax(ind_t1[:,start_idx:stop_idx],axis=1)
	ind_t2_mod_early = np.nanmax(ind_t2[:,start_idx:stop_idx],axis=1)
	##########################################
	###########################################
	##########################################
	##########################################
	##OKAY! Now, repeat for late sessions:
	f = h5py.File(datafile,'r')	
	e1_t1 = []
	e1_t2 = []
	e2_t1 = []
	e2_t2 = []
	ind_t1 = []
	ind_t2 = []
	for animal in animal_list:
		session_list = [x for x in f[animal].keys() if int(x[-6:-4]) in late_range]
		for session in session_list:
			s_group = f[animal][session]
			##start with e1 units
			try:
				u_group = s_group['e1_units']
				##start first with t1
				try:
					t1_data = np.asarray(u_group['t1']).squeeze()
					if len(t1_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t1_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t1_data[unit,:],[100,50]))
							e1_t1.append(smoothed)
					elif len(t1_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t1_data,[100,50]))
						e1_t1.append(smoothed)
				except KeyError:
					print("No T1 for "+animal+" "+session)
				##now do t2
				try:
					t2_data = np.asarray(u_group['t2']).squeeze()
					if len(t2_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t2_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t2_data[unit,:],[100,50]))
							e1_t2.append(smoothed)
					elif len(t2_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t2_data,[100,50]))
						e1_t2.append(smoothed)
				except KeyError:
					print("No T2 for "+animal+" "+session)
			except KeyError:
				print("No E1 units for "+animal+" "+session)
			##now do e2 units
			try:
				u_group = s_group['e2_units']
				##start first with t1
				try:
					t1_data = np.asarray(u_group['t1']).squeeze()
					if len(t1_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t1_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t1_data[unit,:],[100,50]))
							e2_t1.append(smoothed)
					elif len(t1_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t1_data,[100,50]))
						e2_t1.append(smoothed)
				except KeyError:
					print("No T1 for "+animal+" "+session)
				##now do t2
				try:
					t2_data = np.asarray(u_group['t2']).squeeze()
					if len(t2_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t2_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t2_data[unit,:],[100,50]))
							e2_t2.append(smoothed)
					elif len(t2_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t2_data,[100,50]))
						e2_t2.append(smoothed)
				except KeyError:
					print("No T2 for "+animal+" "+session)
			except KeyError:
				print("No E2 units for "+animal+" "+session)
			##now do indirect units
			try:
				u_group = s_group['V1_units']
				##start first with t1
				try:
					t1_data = np.asarray(u_group['t1']).squeeze()
					if len(t1_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t1_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t1_data[unit,:],[100,50]))
							ind_t1.append(smoothed)
					elif len(t1_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t1_data,[100,50]))
						ind_t1.append(smoothed)
				except KeyError:
					print("No T1 for "+animal+" "+session)
				##now do t2
				try:
					t2_data = np.asarray(u_group['t2']).squeeze()
					if len(t2_data.shape)>1:
						##process the data from each unit, which is the mean over all trials
						for unit in range(t2_data.shape[0]):
							smoothed = stats.zscore(ss.windowRate(t2_data[unit,:],[100,50]))
							ind_t2.append(smoothed)
					elif len(t2_data) == 1:
						smoothed = stats.zscore(ss.windowRate(t2_data,[100,50]))
						ind_t2.append(smoothed)
				except KeyError:
					print("No T2 for "+animal+" "+session)
			except KeyError:
				print("No indirect units for "+animal+" "+session)
	f.close()
	##now we have all of the early data
	e1_t1 = np.asarray(e1_t1)
	e1_t2 = np.asarray(e1_t2)
	e2_t1 = np.asarray(e2_t1)
	e2_t2 = np.asarray(e2_t2)
	ind_t1 = np.asarray(ind_t1)
	ind_t2 = np.asarray(ind_t2)
	##take the absolute value, if requested
	if abs_z:
		e1_t1 = abs(e1_t1)
		e1_t2 = abs(e1_t2)
		e2_t1 = abs(e2_t1)
		e2_t2 = abs(e2_t2)
		ind_t1 = abs(ind_t1)
		ind_t2 = abs(ind_t2)
	##now plot this data, for both targets
	fig = plt.figure()
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2,sharey=ax1)
	x = np.linspace(-4,4,e1_t1.shape[1])
	##start with t1
	mean_e1_t1 = np.nanmean(e1_t1,axis=0)*2
	sem_e1_t1 = np.nanstd(e1_t1,axis=0)/np.sqrt(e1_t1.shape[0])*2
	mean_e2_t1 = np.nanmean(e2_t1,axis=0)*2
	sem_e2_t1 = np.nanstd(e2_t1,axis=0)/np.sqrt(e2_t1.shape[0])*2
	mean_ind_t1 = np.nanmean(ind_t1,axis=0)*2
	sem_ind_t1 = np.nanstd(ind_t1,axis=0)/np.sqrt(ind_t1.shape[0])*2
	ax1.plot(x,mean_e1_t1,linewidth=2,color='g',label='E1')
	ax1.fill_between(x,mean_e1_t1-sem_e1_t1,mean_e1_t1+sem_e1_t1,color='g',alpha=0.5)
	ax1.plot(x,mean_e2_t1,linewidth=2,color='b',label='E2')
	ax1.fill_between(x,mean_e2_t1-sem_e2_t1,mean_e2_t1+sem_e2_t1,color='b',alpha=0.5)
	ax1.plot(x,mean_ind_t1,linewidth=2,color='k',label='indirect')
	ax1.fill_between(x,mean_ind_t1-sem_ind_t1,mean_ind_t1+sem_ind_t1,color='k',alpha=0.5)
	ax1.set_title("Rewarded target",fontsize=16)
	ax1.set_ylabel("Zscore",fontsize=16)
	ax1.set_xlabel("Time to target (s)",fontsize=16)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax1.legend()
	##repeat for T2
	mean_e1_t2 = np.nanmean(e1_t2,axis=0)*2
	sem_e1_t2 = np.nanstd(e1_t2,axis=0)/np.sqrt(e1_t2.shape[0])*2
	mean_e2_t2 = np.nanmean(e2_t2,axis=0)*2
	sem_e2_t2 = np.nanstd(e2_t2,axis=0)/np.sqrt(e2_t2.shape[0])*2
	mean_ind_t2 = np.nanmean(ind_t2,axis=0)*2
	sem_ind_t2 = np.nanstd(ind_t2,axis=0)/np.sqrt(ind_t2.shape[0])*2
	ax2.plot(x,mean_e1_t2,linewidth=2,color='g',label='E1')
	ax2.fill_between(x,mean_e1_t2-sem_e1_t2,mean_e1_t2+sem_e1_t2,color='g',alpha=0.5)
	ax2.plot(x,mean_e2_t2,linewidth=2,color='b',label='E2')
	ax2.fill_between(x,mean_e2_t2-sem_e2_t2,mean_e2_t2+sem_e2_t2,color='b',alpha=0.5)
	ax2.plot(x,mean_ind_t2,linewidth=2,color='k',label='indirect')
	ax2.fill_between(x,mean_ind_t2-sem_ind_t2,mean_ind_t2+sem_ind_t2,color='k',alpha=0.5)
	ax2.set_title("Unrewarded target",fontsize=16)
	ax2.set_ylabel("Zscore",fontsize=16)
	ax2.set_xlabel("Time to target (s)",fontsize=16)
	for tick in ax2.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	if abs_z:
		fig.suptitle("Absolute modulations locked to targets",fontsize=16)
	else:
		fig.suptitle("Zscore, late sessions",fontsize=16)
	##finally, get some details about modulation depth
	e1_t1_mod_late = np.nanmax(e1_t1[:,start_idx:stop_idx],axis=1)
	e1_t2_mod_late = np.nanmax(e1_t2[:,start_idx:stop_idx],axis=1)
	e2_t1_mod_late = np.nanmax(e2_t1[:,start_idx:stop_idx],axis=1)
	e2_t2_mod_late = np.nanmax(e2_t2[:,start_idx:stop_idx],axis=1)
	ind_t1_mod_late = np.nanmax(ind_t1[:,start_idx:stop_idx],axis=1)
	ind_t2_mod_late = np.nanmax(ind_t2[:,start_idx:stop_idx],axis=1)
	##now plot the change in modulation depths
	fig, ax = plt.subplots(1)
	dir_mod_early = np.concatenate((e1_t1_mod_early,e2_t1_mod_early))
	dir_mod_late = np.concatenate((e1_t1_mod_early,e2_t1_mod_late))
	dir_mean = np.array([np.nanmean(dir_mod_early),np.nanmean(dir_mod_late)])
	dir_sem = np.array([np.nanstd(dir_mod_early)/np.sqrt(dir_mod_early.shape[0]),
		np.nanstd(dir_mod_late)/np.sqrt(dir_mod_late.shape[0])])
	ind_mean = np.array([np.nanmean(ind_t1_mod_early),np.nanmean(ind_t1_mod_late)])
	ind_sem = np.array([np.nanstd(ind_t1_mod_early)/np.sqrt(ind_t1_mod_early.shape[0]),
		np.nanstd(ind_t1_mod_late)/np.sqrt(ind_t1_mod_late.shape[0])])
	x = [1,2]
	xerr = [0.1,0.1]
	ax.errorbar(x,dir_mean,yerr=dir_sem,xerr=xerr,ecolor='g',capthick=2,elinewidth=2,
		linewidth=2,color='g',label='Direct')
	ax.errorbar(x,ind_mean,yerr=ind_sem,xerr=xerr,ecolor='k',capthick=2,elinewidth=2,
		linewidth=2,color='k',label='Indirect')
	ax.set_ylabel("Absolute modulation depth (zscore)",fontsize=16)
	ax.set_title("Modulation depths of direct VS indirect units",fontsize=16)
	ax.legend()
	ax.text(0.5,0.35,"**",fontsize=20,transform=ax.transAxes)
	ax.set_xticks([1,2])
	ax.set_xticklabels(["Early","Late"])
	ax.set_ylim(2,5)
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	###now run some stats on the modulation depths early/late
	tval_e1_t1,pval_e1_t1 = stats.ttest_ind(e1_t1_mod_early,e1_t1_mod_late,nan_policy='omit')
	print("t-test e1 t1 = "+str(tval_e1_t1)+", "+str(pval_e1_t1))
	tval_e1_t2,pval_e1_t2 = stats.ttest_ind(e1_t2_mod_early,e1_t2_mod_late,nan_policy='omit')
	print("t-test e1 t2 = "+str(tval_e1_t2)+", "+str(pval_e1_t2))
	tval_e2_t1,pval_e2_t1 = stats.ttest_ind(e2_t1_mod_early,e2_t1_mod_late,nan_policy='omit')
	print("t-test e2 t1 = "+str(tval_e2_t1)+", "+str(pval_e2_t1))
	tval_e2_t2,pval_e2_t2 = stats.ttest_ind(e2_t2_mod_early,e2_t2_mod_late,nan_policy='omit')
	print("t-test e2 t2 = "+str(tval_e2_t2)+", "+str(pval_e2_t2))
	tval_ind_t1,pval_ind_t1 = stats.ttest_ind(ind_t1_mod_early,ind_t1_mod_late,nan_policy='omit')
	print("t-test ind t1 = "+str(tval_ind_t1)+", "+str(pval_ind_t1))
	tval_ind_t2,pval_ind_t2 = stats.ttest_ind(ind_t2_mod_early,ind_t2_mod_late,nan_policy='omit')
	print("t-test ind t2 = "+str(tval_ind_t2)+", "+str(pval_ind_t2))


########################## NEURON ######################

"""
computes the correlation coefficient of E1 units for each trial
(including all trial types). Returns a dataframe with the CC of each trial
as well as the animal and trial identity.
"""
def trial_correlations():
	unit_type = 'e1_units'
	root_dir = "/Volumes/Untitled/Ryan/V1_BMI"
	animal_list = [x for x in ru.animals.keys() if not x.startswith("m")]
	window = [1000,100] ##trial window
	tau = 20
	dt = 5
	master_t1_ccs = []
	master_t2_ccs = []
	master_miss_ccs = []
	for animal in animal_list:
		animal_t1_ccs = []
		animal_t2_ccs = []
		animal_miss_ccs = []
		session_list = ru.animals[animal][1].keys()
		for session in session_list:
			session_id = session[-11:-4]
			print("Working on "+animal+" "+session_id)
			filepath = os.path.join(root_dir,animal,session) ##the file path to the session data
			##open the file
			data = plxread.import_file(filepath,AD_channels=list(np.arange(1,97)))
			##what is the duration of this file
			duration = None
			for arr in data.keys():
				if arr.startswith('AD') and arr.endswith('_ts'):
					duration = int(np.ceil((data[arr].max()*1000)/100)*100)+1
					break
			else: print("No A/D timestamp data found!!!")
			##we are going to need the T1, T2, and timeout timestamps fo each file
			try:
				t1_id = ru.animals[animal][1][session]['events']['t1'][0] ##the event name in the plexon file
				t1_ts = data[t1_id]*1000.0
			except KeyError:
				t1_ts = np.array([])
			try:
				t2_id = ru.animals[animal][1][session]['events']['t2'][0] ##the event name in the plexon file
				t2_ts = data[t2_id]*1000.0
			except KeyError:
				t2_ts = np.array([])
			try:
				miss_id = ru.animals[animal][1][session]['events']['miss'][0] ##the event name in the plexon file
				miss_ts = data[miss_id]*1000.0
			except KeyError:
				miss_ts = np.array([])
			try:
				unit_ids = ru.animals[animal][1][session]['units'][unit_type]
				##only consider cases where there are 2 units in the ensemble (should be everything)
				if len(unit_ids) == 2:
					spike_data = np.zeros((2,duration))
					t1_ccs = []
					t2_ccs = []
					miss_ccs = []
					for i,unit in enumerate(unit_ids):
						##get the timestamps for this unit from the datafile
						try:
							unit_ts = data[unit]*1000.0
							##binary transform
							spiketrain = np.histogram(unit_ts,bins=duration,range=(0,duration))
							spiketrain = spiketrain[0].astype(bool).astype(int)
							##add to the data
							spike_data[i,:] = spiketrain
						except KeyError:
							print("No "+unit+" for "+unit_type+" in "+animal+" "+session)
					events = [t1_ts,t2_ts,miss_ts]
					ccs = [t1_ccs,t2_ccs,miss_ccs]
					for event_ts,cc_data in zip(events,ccs):
						if len(event_ts)>0:
							##now, for each trial, take a window around the trial
							unit1_traces = get_data_window(spike_data[0,:],event_ts,window[0],window[1])
							unit2_traces = get_data_window(spike_data[1,:],event_ts,window[0],window[1])
							##re-define n_trials in case there we dropped any
							n_trials = unit1_traces.shape[1]
							##now compute the CC and save the data
							for t in range(n_trials):
								p1 = unit1_traces[:,t]
								p2 = unit2_traces[:,t]
								cc_data.append(ss.corr_coeff(p1,p2,tau,dt))
						else:
							cc_data.append(np.nan)
					##now add these data to the global data containers
					animal_t1_ccs.append(np.nanmean(np.asarray(t1_ccs)))
					animal_t2_ccs.append(np.nanmean(np.asarray(t2_ccs)))
					animal_miss_ccs.append(np.nanmean(np.asarray(miss_ccs)))
				else:
					print("Only {} units in this file".format(len(unit_ids)))
			except KeyError:
				print("No "+unit_type+" for "+animal+" "+session_id)
				##add data from this animal to the master
		master_t1_ccs.append(np.asarray(animal_t1_ccs))
		master_t2_ccs.append(np.asarray(animal_t2_ccs))
		master_miss_ccs.append(np.asarray(animal_miss_ccs))
	return master_t1_ccs,master_t2_ccs,master_miss_ccs

def plot_trial_correlations():
	t1,t2,miss = trial_correlations()
	t1_by_session = np.concatenate(t1)
	t2_by_session = np.concatenate(t2)
	miss_by_session = np.concatenate(miss)
	t1_by_animal = []
	t2_by_animal = []
	miss_by_animal = []
	for i in range(len(t1)):
		t1_by_animal.append(np.nanmean(t1[i]))
	t1_by_animal = np.asarray(t1_by_animal)
	for i in range(len(t2)):
		t2_by_animal.append(np.nanmean(t2[i]))
	t2_by_animal = np.asarray(t2_by_animal)
	for i in range(len(miss)):
		miss_by_animal.append(np.nanmean(miss[i]))
	miss_by_animal = np.asarray(miss_by_animal)
	##first do it by animal
	data = np.asarray([t1_by_animal,t2_by_animal,miss_by_animal])
	means = np.array([np.nanmean(t1_by_animal),np.nanmean(t2_by_animal),np.nanmean(miss_by_animal)])
	sem = np.array([np.nanstd(t1_by_animal)/np.sqrt(t1_by_animal.size),
		np.nanstd(t2_by_animal)/np.sqrt(t2_by_animal.size),np.nanstd(miss_by_animal)/np.sqrt(miss_by_animal.size)])
	tval_t1_t2,pval_t1_t2 = stats.ttest_rel(t1_by_animal, t2_by_animal,nan_policy='omit')
	tval_t1_miss,pval_t1_miss = stats.ttest_rel(t1_by_animal, miss_by_animal,nan_policy='omit')
	tval_t2_miss,pval_t2_miss = stats.ttest_rel(t2_by_animal, miss_by_animal,nan_policy='omit')
	idx = np.arange(3)
	width = 1.0
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, width, color = ['r','b','k'], yerr = sem, ecolor = 'k', alpha = 0.5)
	# ax.set_ylim(0,0.03)
	ax.set_xlim(-0.5, 3.5)
	ax.set_xticks(idx+0.5)
	ax.set_xticklabels(("T1", "T2", "Miss"))
	for i in range(data.shape[1]):
		plt.plot((idx), data[:,i], alpha = 0.5, color = 'k', marker = 'o', linestyle='none')
	ax.set_ylabel("Correlation coefficient", fontsize = 14)
	ax.set_xlabel("Target", fontsize = 14)
	# ax.text(0.3, 0.03, "p = " + str(p_val_p_cd), fontsize = 12)
	# ax.text(2.3, 0.03, "p = " + str(p_val_cd_r), fontsize = 12)
	# ax.text(1.3, 0.03, "p = " + str(p_val_p_r), fontsize = 12)
	fig.suptitle("CC by animal", fontsize = 16)
	print("By animal stats")
	print("pval t1-t2 = "+str(pval_t1_t2))
	print("tval t1-t2 = "+str(tval_t1_t2))
	print("pval t1-miss = "+str(pval_t1_miss))
	print("tval t1-miss = "+str(tval_t1_miss))
	print("pval t2-miss = "+str(pval_t2_miss))
	print("tval t2-miss = "+str(tval_t2_miss))
	print("mean t1 = "+str(means[0]))
	print("mean t2 = "+str(means[1]))
	print("mean miss = "+str(means[2]))
	##next do it by session
	data = np.asarray([t1_by_session,t2_by_session,miss_by_session])
	means = np.array([np.nanmean(t1_by_session),np.nanmean(t2_by_session),np.nanmean(miss_by_session)])
	sem = np.array([np.nanstd(t1_by_session)/np.sqrt(t1_by_session.size),
		np.nanstd(t2_by_session)/np.sqrt(t2_by_session.size),np.nanstd(miss_by_session)/np.sqrt(miss_by_session.size)])
	tval_t1_t2,pval_t1_t2 = stats.ttest_rel(t1_by_session, t2_by_session,nan_policy='omit')
	tval_t1_miss,pval_t1_miss = stats.ttest_rel(t1_by_session, miss_by_session,nan_policy='omit')
	tval_t2_miss,pval_t2_miss = stats.ttest_rel(t2_by_session, miss_by_session,nan_policy='omit')
	idx = np.arange(3)
	width = 1.0
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, width, color = ['r','b','k'], yerr = sem, ecolor = 'k', alpha = 0.5)
	# ax.set_ylim(0,0.03)
	ax.set_xlim(-0.5, 3.5)
	ax.set_xticks(idx+0.5)
	ax.set_xticklabels(("T1", "T2", "Miss"))
	for i in range(data.shape[1]):
		plt.plot((idx), data[:,i], alpha = 0.5, color = 'k', marker = 'o', linestyle = 'none')
	ax.set_ylabel("Correlation coefficient", fontsize = 14)
	ax.set_xlabel("Target", fontsize = 14)
	# ax.text(0.3, 0.03, "p = " + str(p_val_p_cd), fontsize = 12)
	# ax.text(2.3, 0.03, "p = " + str(p_val_cd_r), fontsize = 12)
	# ax.text(1.3, 0.03, "p = " + str(p_val_p_r), fontsize = 12)
	fig.suptitle("CC by session", fontsize = 16)
	print("By animal stats")
	print("pval t1-t2 = "+str(pval_t1_t2))
	print("tval t1-t2 = "+str(tval_t1_t2))
	print("pval t1-miss = "+str(pval_t1_miss))
	print("tval t1-miss = "+str(tval_t1_miss))
	print("pval t2-miss = "+str(pval_t2_miss))
	print("tval t2-miss = "+str(tval_t2_miss))
	print("mean t1 = "+str(means[0]))
	print("mean t2 = "+str(means[1]))
	print("mean miss = "+str(means[2]))

def correlations_v_performance():
	##define some gobal parameters
	unit_type = 'e1_units'
	root_dir = "/Volumes/Untitled/Ryan/V1_BMI"
	f_out = h5py.File("/Volumes/Untitled/Ryan/V1_BMI/Neuron/performance_v_correlation_fine.hdf5",'w-')
	animal_list = [x for x in ru.animals.keys() if not x.startswith("m")]
	tau = 10
	dt = 5
	window = [5000,1000]
	session_range = np.arange(0,10)
	all_performance = []
	all_correlation = []
	for animal in animal_list:
		animal_performance = []
		animal_correlation = []
		if session_range is None:
			session_list = ru.animals[animal][1].keys()
		else: 
			session_list = [x for x in ru.animals[animal][1].keys() if int(x[-6:-4]) in session_range]
		for session in session_list:
			session_id = session[-11:-4]
			print("Working on "+animal+" "+session_id)
			filepath = os.path.join(root_dir,animal,session) ##the file path to the session data
			data = plxread.import_file(filepath,AD_channels=list(np.arange(1,97)))
			##what is the duration of this file
			duration = None
			for arr in data.keys():
				if arr.startswith('AD') and arr.endswith('_ts'):
					duration = int(np.ceil((data[arr].max()*1000)/100)*100)+1
					break
			else: print("No A/D timestamp data found!!!")
			##we are going to need the T1, T2, and timeout timestamps fo each file
			try:
				t1_id = ru.animals[animal][1][session]['events']['t1'][0] ##the event name in the plexon file
				t1_ts = data[t1_id]*1000.0
				t1_ts = np.histogram(t1_ts,bins=duration,range=(0,duration))
				t1_ts = t1_ts[0].astype(bool).astype(int)
			except KeyError:
				t1_ts = np.zeros(duration)
			try:
				t2_id = ru.animals[animal][1][session]['events']['t2'][0] ##the event name in the plexon file
				t2_ts = data[t2_id]*1000.0
				t2_ts = np.histogram(t2_ts,bins=duration,range=(0,duration))
				t2_ts = t2_ts[0].astype(bool).astype(int)
			except KeyError:
				t2_ts = np.zeros(duration)
			try:
				miss_id = ru.animals[animal][1][session]['events']['miss'][0] ##the event name in the plexon file
				miss_ts = data[miss_id]*1000.0
				miss_ts = np.histogram(miss_ts,bins=duration,range=(0,duration))
				miss_ts = miss_ts[0].astype(bool).astype(int)
			except KeyError:
				miss_ts = np.zeros(duration)
			try:
				unit_ids = ru.animals[animal][1][session]['units'][unit_type]
				##only consider cases where there are 2 units in the ensemble (should be everything)
				if len(unit_ids) == 2:
					spike_data = np.zeros((2,duration))
					for i,unit in enumerate(unit_ids):
						##get the timestamps for this unit from the datafile
						try:
							unit_ts = data[unit]*1000.0
							##binary transform
							spiketrain = np.histogram(unit_ts,bins=duration,range=(0,duration))
							spiketrain = spiketrain[0].astype(bool).astype(int)
							##add to the data
							spike_data[i,:] = spiketrain
						except KeyError:
							print("No "+unit+" for "+unit_type+" in "+animal+" "+session)
					##now we have the spike data, as well as all of the event timestamps in
					##equal-length binary arrays. Now, we should be able to compute performance
					##AND correlation coefficients with equal windowing
					N = duration
					Nwin = window[0]
					Nstep = window[1]
					winstart = np.arange(0,N-Nwin, Nstep)
					nw = winstart.shape[0]
					performance = np.zeros(nw)
					correlation = np.zeros(nw)
					for n in range(nw):
						idx = np.arange(winstart[n], winstart[n]+Nwin)
						t1_counts = t1_ts[idx].sum()
						t2_counts = t2_ts[idx].sum()
						miss_counts = miss_ts[idx].sum()
						total_counts = t1_counts+t2_counts+miss_counts
						if total_counts>0:
							perc_correct = t1_counts/total_counts
						else:
							perc_correct = 0
						##now compute the correlation coeff for this chunk
						p1 = spike_data[0,idx]
						p2 = spike_data[1,idx]
						cc = ss.corr_coeff(p1,p2,tau,dt)
						##now add to the data arrays
						performance[n] = perc_correct
						correlation[n] = cc
					##okay, now we have these arrays for this session; add them to the animal array
					animal_performance.append(performance)
					animal_correlation.append(correlation)
				else:
					print("Only {} units in this file".format(len(unit_ids)))
			except KeyError:
				print("No "+unit_type+" for "+animal+" "+session_id)
		##now equalize these arrays
		f_out.create_dataset(animal,data=equalize_arrs(animal_performance))
		f_out.create_dataset(animal,data=equalize_arrs(animal_correlation))
	# return all_performance,all_correlation

def parse_correlations():
	f_in = h5py.File("/Volumes/Untitled/Ryan/V1_BMI/Neuron/performance_v_correlation.hdf5",'r')
	animal_list = list(f_in)
	pearson_r = []
	pearson_p = []
	all_r = []
	all_p = []
	for animal in animal_list:
		print(animal)
		avg_r = []
		avg_p = []
		performance = np.asarray(f_in[animal]['performance'])
		correlation = np.asarray(f_in[animal]['correlation'])
		for s in range(performance.shape[0]):
			try:
				# print(s)
				p = stats.zscore(remove_nan(performance[s,:]))
				c = stats.zscore(remove_nan(correlation[s,:]))
				# print(p.shape)
				# print(c.shape)
				avg_r.append(stats.pearsonr(p,c)[0])
				avg_p.append(stats.pearsonr(p,c)[1])
				all_r.append(stats.pearsonr(p,c)[0])
				all_p.append(stats.pearsonr(p,c)[1])
			except ValueError:
				pass
		pearson_r.append(np.asarray(avg_r))
		pearson_p.append(np.asarray(avg_p))
	f_in.close()
	return np.asarray(all_r),np.asarray(all_p)
	



def get_light_change_data2():
	source_file = "/Volumes/Ryan/Data/processed_data/V1_BMI_final/raw_data/R7_thru_V13_all_data.hdf5"
	destination_file = "/Volumes/Untitled/Ryan/V1_BMI/Neuron/light_change_data.hdf5"
	f = h5py.File(source_file, 'r')
	f_out = h5py.File(destination_file, 'a')

	t1_light_to_dark = []
	t2_light_to_dark = []
	miss_light_to_dark = []
	p_correct_light_to_dark = []

	t1_dark_to_light = []
	t2_dark_to_light = []
	miss_dark_to_light = []
	p_correct_dark_to_light = []

	##R8 D1
	##light to dark
	session = f['R8']['BMI_D11.plx']
	t1,t2,miss,p_correct = get_event_data(session)
	t1_light_to_dark.append(t1)
	t2_light_to_dark.append(t2)
	miss_light_to_dark.append(miss)
	p_correct_light_to_dark.append(p_correct)
	##R8 D2
	##light to dark
	session = f['R8']['BMI_D17.plx']
	t1,t2,miss,p_correct = get_event_data(session)
	t1_light_to_dark.append(t1)
	t2_light_to_dark.append(t2)
	miss_light_to_dark.append(miss)
	p_correct_light_to_dark.append(p_correct)

	##V02
	##light to dark
	session1 = f['V02']['BMI_D14.plx']
	session2 = f['V02']['BMI_D15.plx']
	t1_1,t2_1,miss_1,p_correct_1 = get_event_data(session1)
	t1_2,t2_2,miss_2,p_correct_2 = get_event_data(session2)

	t1 = np.concatenate([t1_1[-30:],t1_2[:40]])
	t2 = np.concatenate([t2_1[-30:],t2_2[:40]])
	miss = np.concatenate([miss_1[-30:],miss_2[:40]])
	p_correct = np.concatenate([p_correct_1[-30:],p_correct_2[:40]])

	t1_light_to_dark.append(t1)
	t2_light_to_dark.append(t2)
	miss_light_to_dark.append(miss)
	p_correct_light_to_dark.append(p_correct)

	##V03
	##light to dark
	session1 = f['V03']['BMI_D14.plx']
	session2 = f['V03']['BMI_D15.plx']
	t1_1,t2_1,miss_1,p_correct_1 = get_event_data(session1)
	t1_2,t2_2,miss_2,p_correct_2 = get_event_data(session2)

	t1 = np.concatenate([t1_1[-30:],t1_2[:40]])
	t2 = np.concatenate([t2_1[-30:],t2_2[:40]])
	miss = np.concatenate([miss_1[-30:],miss_2[:40]])
	p_correct = np.concatenate([p_correct_1[-30:],p_correct_2[:40]])

	t1_light_to_dark.append(t1)
	t2_light_to_dark.append(t2)
	miss_light_to_dark.append(miss)
	p_correct_light_to_dark.append(p_correct)

	##V04
	##light to dark
	session1 = f['V04']['BMI_D14.plx']
	session2 = f['V04']['BMI_D15.plx']
	t1_1,t2_1,miss_1,p_correct_1 = get_event_data(session1)
	t1_2,t2_2,miss_2,p_correct_2 = get_event_data(session2)

	t1 = np.concatenate([t1_1[-30:],t1_2[:40]])
	t2 = np.concatenate([t2_1[-30:],t2_2[:40]])
	miss = np.concatenate([miss_1[-30:],miss_2[:40]])
	p_correct = np.concatenate([p_correct_1[-30:],p_correct_2[:40]])

	t1_light_to_dark.append(t1)
	t2_light_to_dark.append(t2)
	miss_light_to_dark.append(miss)
	p_correct_light_to_dark.append(p_correct)


	##V05
	##light to dark
	session1 = f['V05']['BMI_D14.plx']
	session2 = f['V05']['BMI_D15.plx']
	t1_1,t2_1,miss_1,p_correct_1 = get_event_data(session1)
	t1_2,t2_2,miss_2,p_correct_2 = get_event_data(session2)

	t1 = np.concatenate([t1_1[-30:],t1_2[:40]])
	t2 = np.concatenate([t2_1[-30:],t2_2[:40]])
	miss = np.concatenate([miss_1[-30:],miss_2[:40]])
	p_correct = np.concatenate([p_correct_1[-30:],p_correct_2[:40]])

	t1_light_to_dark.append(t1)
	t2_light_to_dark.append(t2)
	miss_light_to_dark.append(miss)
	p_correct_light_to_dark.append(p_correct)

	##R8
	##dark to light
	session1 = f['R8']['BMI_D11.plx']
	session2 = f['R8']['BMI_D12.plx']
	t1_1,t2_1,miss_1,p_correct_1 = get_event_data(session1)
	t1_2,t2_2,miss_2,p_correct_2 = get_event_data(session2)

	t1 = np.concatenate([t1_1[-30:],t1_2[:40]])
	t2 = np.concatenate([t2_1[-30:],t2_2[:40]])
	miss = np.concatenate([miss_1[-30:],miss_2[:40]])
	p_correct = np.concatenate([p_correct_1[-30:],p_correct_2[:40]])

	t1_dark_to_light.append(t1)
	t2_dark_to_light.append(t2)
	miss_dark_to_light.append(miss)
	p_correct_dark_to_light.append(p_correct)

	##V02
	##dark to light
	session = f['V02']['BMI_D14.plx']
	t1,t2,miss,p_correct = get_event_data(session)
	t1_dark_to_light.append(t1)
	t2_dark_to_light.append(t2)
	miss_dark_to_light.append(miss)
	p_correct_dark_to_light.append(p_correct)

	##V03
	##dark to light
	session = f['V03']['BMI_D14.plx']
	t1,t2,miss,p_correct = get_event_data(session)
	t1_dark_to_light.append(t1)
	t2_dark_to_light.append(t2)
	miss_dark_to_light.append(miss)
	p_correct_dark_to_light.append(p_correct)

	##V04
	##dark to light
	session = f['V04']['BMI_D14.plx']
	t1,t2,miss,p_correct = get_event_data(session)
	t1_dark_to_light.append(t1)
	t2_dark_to_light.append(t2)
	miss_dark_to_light.append(miss)
	p_correct_dark_to_light.append(p_correct)

	##V05
	##dark to light
	session = f['V05']['BMI_D14.plx']
	t1,t2,miss,p_correct = get_event_data(session)
	t1_dark_to_light.append(t1)
	t2_dark_to_light.append(t2)
	miss_dark_to_light.append(miss)
	p_correct_dark_to_light.append(p_correct)

	f.close()

	##now equalize all the array lenghts
	t1_dark_to_light = equalize_arrs(t1_dark_to_light)
	t2_dark_to_light = equalize_arrs(t2_dark_to_light)
	miss_dark_to_light = equalize_arrs(miss_dark_to_light)
	p_correct_dark_to_light = equalize_arrs(p_correct_dark_to_light)

	t1_light_to_dark = equalize_arrs(t1_light_to_dark)
	t2_light_to_dark = equalize_arrs(t2_light_to_dark)
	miss_light_to_dark = equalize_arrs(miss_light_to_dark)
	p_correct_light_to_dark = equalize_arrs(p_correct_light_to_dark)

	f_out.create_dataset('t1_light_to_dark',data = t1_light_to_dark)
	f_out.create_dataset('t2_light_to_dark',data = t2_light_to_dark)
	f_out.create_dataset('miss_light_to_dark',data = miss_light_to_dark)
	f_out.create_dataset('p_correct_light_to_dark',data = p_correct_light_to_dark)
	f_out.create_dataset('t1_dark_to_light',data = t1_dark_to_light)
	f_out.create_dataset('t2_dark_to_light',data = t2_dark_to_light)
	f_out.create_dataset('miss_dark_to_light',data = miss_dark_to_light)
	f_out.create_dataset('p_correct_dark_to_light',data = p_correct_dark_to_light)

	f_out.close()
	print("Done!")

def plot_light_change2():
	f = h5py.File("/Volumes/Untitled/Ryan/V1_BMI/Neuron/light_change_data.hdf5",'r')
	light_dark = np.asarray(f['p_correct_light_to_dark'])
	dark_light = np.asarray(f['p_correct_dark_to_light'])
	# for i in range(light_dark.shape[0]):
	# 	light_dark[i,:] = ss.gauss_convolve(light_dark[i,:],0.3)
	# for i in range(dark_light.shape[0]):
	# 	dark_light[i,:] = ss.gauss_convolve(dark_light[i,:],0.3)
	f.close()
	duration = max(light_dark.shape[1],dark_light.shape[1])
	x_axis = np.arange(duration)
	mean_dl = np.nanmean(dark_light,axis=0)
	std_dl = np.nanstd(dark_light,axis=0)
	sem_dl = std_dl/np.sqrt(dark_light.shape[0])
	mean_ld = np.nanmean(light_dark,axis=0)
	std_ld = np.nanstd(light_dark,axis=0)
	sem_ld = std_ld/np.sqrt(light_dark.shape[0])

	fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True)
	ax1.plot(x_axis[0:30],mean_dl[0:30],linewidth = 2,color = 'k')
	ax1.plot(x_axis[30:],mean_dl[30:],linewidth=2,color='yellow')
	ax2.plot(x_axis[0:30],mean_ld[0:30],linewidth = 2,color = 'yellow')
	ax2.plot(x_axis[30:mean_ld.size],mean_ld[30:],linewidth=2,color='k')	
	ax1.fill_between(x_axis[0:30],mean_dl[0:30]-sem_dl[0:30],mean_dl[0:30]+sem_dl[0:30], 
		alpha=0.5,facecolor='k')
	ax1.fill_between(x_axis[30:],mean_dl[30:]-sem_dl[30:],mean_dl[30:]+sem_dl[30:], 
		alpha=0.5,facecolor='yellow')
	ax2.fill_between(x_axis[30:mean_ld.size],mean_ld[30:]-sem_ld[30:],mean_ld[30:]+sem_ld[30:], 
		alpha=0.5,facecolor='k')
	ax2.fill_between(x_axis[0:30],mean_ld[0:30]-sem_ld[0:30],mean_ld[0:30]+sem_ld[0:30], 
		alpha=0.5,facecolor='yellow')
	for tick in ax1.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax1.set_xlabel("Time in session, mins", fontsize = 16)
	ax1.set_ylabel("Percent rewarded", fontsize = 16)
	ax1.set_ylim((0,1))
	ax2.set_xlabel("Time in session, mins", fontsize = 16)
	# ax2.set_ylabel("Percent rewarded", fontsize = 16)
	ax2.set_ylim((0,1))
	ax1.set_xlim((0,60))
	ax2.set_xlim((0,60))
	ax1.set_title("Train dark/test light", fontsize = 18)
	ax2.set_title("Train light/test dark", fontsize=18)
	##now do the comarisons
	f = h5py.File("/Volumes/Untitled/Ryan/V1_BMI/Neuron/light_change_data.hdf5",'r')
	light_dark = np.asarray(f['p_correct_light_to_dark'])
	dark_light = np.asarray(f['p_correct_dark_to_light'])
	f.close()
	##first for dark->light
	train = light_dark[:,0:15].mean(axis=1)
	test_early = light_dark[:,30:45].mean(axis=1)
	test_late = light_dark[:,45:60].mean(axis=1)
	data = np.asarray([train,test_early,test_late]).T
	means = np.array([np.nanmean(train), np.nanmean(test_early), np.nanmean(test_late)])
	sem = np.array([np.nanstd(train), np.nanstd(test_early), np.nanstd(test_late)])/np.sqrt(train.shape[0])
	t_train_early,p_train_early = stats.ttest_rel(train, test_early)
	t_train_late,p_train_late = stats.ttest_rel(train, test_late)
	t_early_late,p_early_late = stats.ttest_rel(test_early, test_late)
	idx = np.arange(3)
	width = 1.0
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, width, color = ['yellow','k','k'], yerr = sem, ecolor = 'k', alpha = 0.5)
	ax.set_ylim(0,0.9)
	ax.set_xlim(-0.5, 3.5)
	ax.set_xticks(idx+0.5)
	ax.set_xticklabels(("Train", "Test early", "Test late"))
	for i in range(train.shape[0]):
		plt.plot(idx, data[i,:], alpha = 0.5, color = 'k', marker = 'o', linewidth = 2)
	ax.set_ylabel("Percent correct", fontsize = 14)
	ax.set_xlabel("Condition", fontsize = 14)
	ax.text(0.3, 0.81, "p = " + str(p_train_early), fontsize = 12)
	ax.text(2.3, 0.81, "p = " + str(p_train_late), fontsize = 12)
	ax.text(1.3, 0.85, "p = " + str(p_early_late), fontsize = 12)
	fig.suptitle("Performance during light change", fontsize = 16)
	print("pval train-early = "+str(p_train_early))
	print("tval train-early = "+str(t_train_early))
	print("pval train-late = "+str(p_train_late))
	print("tval train-late = "+str(t_train_late))
	print("pval early-late = "+str(p_early_late))
	print("tval early-late = "+str(t_early_late))
	print("mean train = "+str(means[0]))
	print("mean early = "+str(means[1]))
	print("mean late = "+str(means[2]))

	train = dark_light[:,0:15].mean(axis=1)
	test_early = dark_light[:,30:45].mean(axis=1)
	test_late = dark_light[:,45:60].mean(axis=1)
	data = np.asarray([train,test_early,test_late]).T
	means = np.array([np.nanmean(train), np.nanmean(test_early), np.nanmean(test_late)])
	sem = np.array([np.nanstd(train), np.nanstd(test_early), np.nanstd(test_late)])/np.sqrt(train.shape[0])
	t_train_early,p_train_early = stats.ttest_rel(train, test_early)
	t_train_late,p_train_late = stats.ttest_rel(train, test_late)
	t_early_late,p_early_late = stats.ttest_rel(test_early, test_late)
	idx = np.arange(3)
	width = 1.0
	fig, ax = plt.subplots()
	bars = ax.bar(idx, means, width, color = ['k','yellow','yellow'], yerr = sem, ecolor = 'k', alpha = 0.5)
	ax.set_ylim(0,0.9)
	ax.set_xlim(-0.5, 3.5)
	ax.set_xticks(idx+0.5)
	ax.set_xticklabels(("Train", "Test early", "Test late"))
	for i in range(train.shape[0]):
		plt.plot(idx, data[i,:], alpha = 0.5, color = 'k', marker = 'o', linewidth = 2)
	ax.set_ylabel("Percent correct", fontsize = 14)
	ax.set_xlabel("Condition", fontsize = 14)
	ax.text(0.3, 0.81, "p = " + str(p_train_early), fontsize = 12)
	ax.text(2.3, 0.81, "p = " + str(p_train_late), fontsize = 12)
	ax.text(1.3, 0.85, "p = " + str(p_early_late), fontsize = 12)
	fig.suptitle("Performance during light change", fontsize = 16)
	print("pval train-early = "+str(p_train_early))
	print("tval train-early = "+str(t_train_early))
	print("pval train-late = "+str(p_train_late))
	print("tval train-late = "+str(t_train_late))
	print("pval early-late = "+str(p_early_late))
	print("tval early-late = "+str(t_early_late))
	print("mean train = "+str(means[0]))
	print("mean early = "+str(means[1]))
	print("mean late = "+str(means[2]))

def get_stim_v_nostim_frs():
	root_dir = "/Volumes/Untitled/Ryan/V1_BMI"
	animal_list = [x for x in ru.animals.keys() if x.startswith("m")]
	save_file = "/Volumes/Untitled/Ryan/V1_BMI/Neuron/fr_v_stim.hdf5"
	window = [6000,100]
	session_range = np.arange(1,8)
	f_out = h5py.File(save_file,'a')
	f_out.close()
	for animal in animal_list:
		animal_means_stim = []
		animal_means_off = []
		f_out = h5py.File(save_file,'a')
		try:
			a_group = f_out[animal]
		except KeyError:
			a_group = f_out.create_group(animal)
			#if session_list is None:
			session_list = [x for x in list(ru.animals[animal][1]) if int(x[-6:-4]) in session_range]
			for session in session_list:
				session_group = a_group.create_group(session)
				stim_rates = []
				off_rates = []
				print("Working on "+animal+" "+session)
				filepath = os.path.join(root_dir,animal,session) ##the file path to the volitional part
				metapath = os.path.join(root_dir,animal,session[:-4]+".mat")
				##start with the non-manipulation file
				data = plxread.import_file(filepath,save_wf=False,
					import_unsorted=False,verbose=False)
				##we are going to need the T1 and T2 timestamps fo each file
				t1_id = ru.animals[animal][1][session]['events']['t1'][0] ##the event name in the plexon file
				t2_id = ru.animals[animal][1][session]['events']['t2'][0]
				miss_id = ru.animals[animal][1][session]['events']['miss'][0]
				try:
					t1_ts = data[t1_id]*1000
				except KeyError:
					t1_ts =np.array([])
				try:
					t2_ts = data[t2_id]*1000
				except KeyError:
					t2_ts = np.array([])
				try:
					miss_ts = data[miss_id]*1000
				except KeyError:
					miss_ts = np.array([])
				trial_ts = np.concatenate([t1_ts,t2_ts,miss_ts])
				stim_idx,off_idx = parse_stim_data(metapath)
				stim_idx = [x for x in stim_idx if x < len(trial_ts)]
				off_idx = [x for x in off_idx if x < len(trial_ts)]
				stim_trials = trial_ts[stim_idx]
				off_trials = trial_ts[off_idx]
				unit_list = ru.animals[animal][1][session]['units']['V1_units']
				if len(trial_ts)>0:
					for unit in unit_list:
						try:
							spike_ts = data[unit]*1000.0
							duration = (spike_ts.max()/1000.0)+2
							spike_data = ss.event_times_to_binary(spike_ts,duration)
							stim_spikes = get_data_window(spike_data,stim_trials,window[0],window[1])
							off_spikes = get_data_window(spike_data,off_trials,window[0],window[1])
							if stim_spikes is not None and off_spikes is not None:
								n_seconds_stim = len(stim_trials)*(window[0]+window[1])/1000.0
								n_seconds_off = len(off_trials)*(window[0]+window[1])/1000.0
								stim_rates.append(stim_spikes.sum()/n_seconds_stim)
								off_rates.append(off_spikes.sum()/n_seconds_off)
						except KeyError:
							pass
				session_group.create_dataset("stim_rates",data=np.asarray(stim_rates))
				session_group.create_dataset("off_rates",data=np.asarray(off_rates))
				animal_means_stim.append(np.mean(stim_rates))
				animal_means_off.append(np.mean(off_rates))
			a_group.create_dataset('stim_means',data=np.asarray(animal_means_stim))
			a_group.create_dataset('off_means',data=np.asarray(animal_means_off))
	f_out.close()
	print("Done")

def plot_stim_frs():
	f = h5py.File("/Volumes/Untitled/Ryan/V1_BMI/Neuron/fr_v_stim.hdf5",'r')
	##probably just want to compare all units
	stim_units = []
	off_units = []
	animal_list = list(f)
	for animal in animal_list:
		sessions = [x for x in list(f[animal]) if x.startswith("BMI")]
		for session in sessions:
			stim_rates = np.asarray(f[animal][session]['stim_rates'])
			off_rates = np.asarray(f[animal][session]['off_rates'])
			stim_units.append(stim_rates)
			off_units.append(off_rates)
	f.close()
	stim_units = np.concatenate(stim_units)/8.0
	off_units = np.concatenate(off_units)/8.0
	stim_mean = np.mean(stim_units)
	off_mean = np.mean(off_units)
	stim_sem = stats.sem(stim_units)
	off_sem = stats.sem(off_units)
	means = np.array([stim_mean,off_mean])
	sems = np.array([stim_sem,off_sem])
	t_val,p_val = stats.ttest_rel(stim_units,off_units)

	fig, ax2 = plt.subplots(1)
	xr = np.array([0,1])
	err_x = np.array([0,1])
	yerr = sems
	xerr = np.ones(2)*0.1
	width=0.5
	# for i in range(rew.shape[1]):
	# 	ax2.plot(xr,rew[:,i],color='k',linewidth=2,marker='o')
	bars = ax2.bar(xr, means, width, color = ['red','k'], yerr = sems, ecolor = 'k', alpha = 0.5)
	# ax2.errorbar(err_x,means,yerr=yerr,xerr=xerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),['LED on','LED off'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.3,1.3)
	ax2.set_ylim(0,12)
	ax2.set_title("V1 firing rates, LED on V LED off")
	print("pval = "+str(p_val))
	print("tval = "+str(t_val))
	print("mean LED on = "+str(means[0]))
	print("mean LED off = "+str(means[1]))

def parse_dms_locked2():
	datafile = r"C:\Users\Ryan\Documents\data\t1_triggered.hdf5"
	savefile = r"D:\Ryan\V1_BMI\Neuron\dms_t1_locked2.hdf5"
	f = h5py.File(datafile,'r')
	f_out = h5py.File(savefile,'w-')
	early_session_range = np.arange(0,4)
	late_session_range = np.arange(7,15)
	early_sessions = [x for x in list(f) if int(x[-2:]) in early_session_range]
	late_sessions = [x for x in list(f) if int(x[-2:]) in late_session_range]
	early_spikes = []
	late_spikes = []
	for s in early_sessions:
		try: 
			spike_data = np.asarray(f[s]['Str_units']) ##shape trials x time x units
			for u in range(spike_data.shape[2]):
				unit_data = spike_data[:,:,u] ##shape trials x bins
				smoothed = smooth_spikes(unit_data,smooth_method='both',smooth_width=[80,40]) #still trials x bins
				avg = np.nanmean(smoothed,axis=0) ##avg all trials, now shape is (bins,)
				early_spikes.append(zscore(avg))				
		except KeyError:
			pass
	for s in late_sessions:
		try: 
			spike_data = np.asarray(f[s]['Str_units']) ##shape trials x time x units
			for u in range(spike_data.shape[2]):
				unit_data = spike_data[:,:,u] ##shape trials x bins
				smoothed = smooth_spikes(unit_data,smooth_method='both',smooth_width=[80,40]) #still trials x bins
				avg = np.nanmean(smoothed,axis=0) ##avg all trials, now shape is (bins,)
				late_spikes.append(zscore(avg))				
		except KeyError:
			pass
	f.close()
	group = f_out.create_group("across_days")
	group.create_dataset("early",data=np.asarray(early_spikes))
	group.create_dataset("late",data=np.asarray(late_spikes))
	##now repeat, but look at WithIN session differences
	datafile = r"C:\Users\Ryan\Documents\data\t1_triggered.hdf5"
	f = h5py.File(datafile,'r')
	session_range = np.arange(0,12)
	sessions = [x for x in list(f) if int(x[-2:]) in session_range]
	early_spikes = []
	late_spikes = []
	for s in sessions:
		try: 
			spike_data = np.asarray(f[s]['Str_units']) ##shape trials x time x units
			for u in range(spike_data.shape[2]):
				unit_data = spike_data[:,:,u] ##shape trials x bins
				smoothed = smooth_spikes(unit_data,smooth_method='both',smooth_width=[80,40]) #still trials x bins
				##Now get the early trials (first 25%)
				total_trials = smoothed.shape[0]
				chunk_size = int(np.ceil(total_trials*0.25))
				early_trials_avg = np.nanmean(smoothed[0:chunk_size,:],axis=0)
				late_trials_avg = np.nanmean(smoothed[-chunk_size:,:],axis=0)
				early_spikes.append(zscore(early_trials_avg))
				late_spikes.append(zscore(late_trials_avg))				
		except KeyError:
			pass
	f.close()
	group = f_out.create_group("within_session_all")
	group.create_dataset("early",data=np.asarray(early_spikes))
	group.create_dataset("late",data=np.asarray(late_spikes))
	early = np.asarray(early_spikes)
	late = np.asarray(late_spikes)
	early_mean = np.nanmean(early,axis=0)
	early_sem = np.nanstd(early,axis=0)/np.sqrt(early.shape[0])
	late_mean = np.nanmean(late,axis=0)
	late_sem = np.nanstd(late,axis=0)/np.sqrt(late.shape[0])
	fig,ax = plt.subplots(1)
	x = np.linspace(-6,2,early.shape[1])
	ax.plot(x,early_mean,color='cyan',linewidth=2,label='early sessions')
	ax.plot(x,late_mean,color='blue',linewidth=2,label='late sessions')
	ax.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='cyan',alpha=0.5)
	ax.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='blue',alpha=0.5)
	ax.set_ylabel("z-score",fontsize=14)
	ax.set_xlabel("Time to target, s",fontsize=14)
	ax.set_title("DMS modulation within sessions, all sessions")
	##now do within session early
	datafile = r"C:\Users\Ryan\Documents\data\t1_triggered.hdf5"
	f = h5py.File(datafile,'r')
	session_range = np.arange(0,4)
	sessions = [x for x in list(f) if int(x[-2:]) in session_range]
	early_spikes = []
	late_spikes = []
	for s in sessions:
		try: 
			spike_data = np.asarray(f[s]['Str_units']) ##shape trials x time x units
			for u in range(spike_data.shape[2]):
				unit_data = spike_data[:,:,u] ##shape trials x bins
				smoothed = smooth_spikes(unit_data,smooth_method='both',smooth_width=[80,40]) #still trials x bins
				##Now get the early trials (first 25%)
				total_trials = smoothed.shape[0]
				chunk_size = int(np.ceil(total_trials*0.25))
				early_trials_avg = np.nanmean(smoothed[0:chunk_size,:],axis=0)
				late_trials_avg = np.nanmean(smoothed[-chunk_size:,:],axis=0)
				early_spikes.append(zscore(early_trials_avg))
				late_spikes.append(zscore(late_trials_avg))				
		except KeyError:
			pass
	f.close()
	group = f_out.create_group("within_session_early")
	group.create_dataset("early",data=np.asarray(early_spikes))
	group.create_dataset("late",data=np.asarray(late_spikes))
	early = np.asarray(early_spikes)
	late = np.asarray(late_spikes)
	early_mean = np.nanmean(early,axis=0)
	early_sem = np.nanstd(early,axis=0)/np.sqrt(early.shape[0])
	late_mean = np.nanmean(late,axis=0)
	late_sem = np.nanstd(late,axis=0)/np.sqrt(late.shape[0])
	fig,ax = plt.subplots(1)
	x = np.linspace(-6,2,early.shape[1])
	ax.plot(x,early_mean,color='cyan',linewidth=2,label='early sessions')
	ax.plot(x,late_mean,color='blue',linewidth=2,label='late sessions')
	ax.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='cyan',alpha=0.5)
	ax.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='blue',alpha=0.5)
	ax.set_ylabel("z-score",fontsize=14)
	ax.set_xlabel("Time to target, s",fontsize=14)
	ax.set_title("DMS modulation within sessions, early sessions")
	##Now within session late
	datafile = r"C:\Users\Ryan\Documents\data\t1_triggered.hdf5"
	f = h5py.File(datafile,'r')
	session_range = np.arange(7,12)
	sessions = [x for x in list(f) if int(x[-2:]) in session_range]
	early_spikes = []
	late_spikes = []
	for s in sessions:
		try: 
			spike_data = np.asarray(f[s]['Str_units']) ##shape trials x time x units
			for u in range(spike_data.shape[2]):
				unit_data = spike_data[:,:,u] ##shape trials x bins
				smoothed = smooth_spikes(unit_data,smooth_method='both',smooth_width=[80,40]) #still trials x bins
				##Now get the early trials (first 25%)
				total_trials = smoothed.shape[0]
				chunk_size = int(np.ceil(total_trials*0.25))
				early_trials_avg = np.nanmean(smoothed[0:chunk_size,:],axis=0)
				late_trials_avg = np.nanmean(smoothed[-chunk_size:,:],axis=0)
				early_spikes.append(zscore(early_trials_avg))
				late_spikes.append(zscore(late_trials_avg))				
		except KeyError:
			pass
	f.close()
	group = f_out.create_group("within_session_late")
	group.create_dataset("early",data=np.asarray(early_spikes))
	group.create_dataset("late",data=np.asarray(late_spikes))
	early = np.asarray(early_spikes)
	late = np.asarray(late_spikes)
	early_mean = np.nanmean(early,axis=0)
	early_sem = np.nanstd(early,axis=0)/np.sqrt(early.shape[0])
	late_mean = np.nanmean(late,axis=0)
	late_sem = np.nanstd(late,axis=0)/np.sqrt(late.shape[0])
	fig,ax = plt.subplots(1)
	x = np.linspace(-6,2,early.shape[1])
	ax.plot(x,early_mean,color='cyan',linewidth=2,label='early sessions')
	ax.plot(x,late_mean,color='blue',linewidth=2,label='late sessions')
	ax.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='cyan',alpha=0.5)
	ax.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='blue',alpha=0.5)
	ax.set_ylabel("z-score",fontsize=14)
	ax.set_xlabel("Time to target, s",fontsize=14)
	ax.set_title("DMS modulation within sessions, late sessions")
	f_out.close()

def plot_dms_locked2():
	datafile = r"D:\Ryan\V1_BMI\Neuron\DMS_T1_locked.hdf5"
	f = h5py.File(datafile,'r')
	group = f['across_days']
	early = np.asarray(group['early'])
	late = np.asarray(group['late'])
	early_mean = np.nanmean(early,axis=0)
	early_sem = np.nanstd(early,axis=0)/np.sqrt(early.shape[0])
	late_mean = np.nanmean(late,axis=0)
	late_sem = np.nanstd(late,axis=0)/np.sqrt(late.shape[0])
	mod_depth_early = np.abs(np.nanmax(early[:,125:175],axis=1)-np.nanmin(early[:,125:175],axis=1))
	mod_depth_late = np.abs(np.nanmax(late[:,125:175],axis=1)-np.nanmin(late[:,125:175],axis=1))
	fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1)
	x = np.linspace(-6,2,early.shape[1])
	ax1.plot(x,early_mean,color='cyan',linewidth=2,label='early sessions')
	ax1.plot(x,late_mean,color='blue',linewidth=2,label='late sessions')
	ax1.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='cyan',alpha=0.5)
	ax1.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='blue',alpha=0.5)
	ax1.set_ylabel("z-score",fontsize=14)
	ax1.set_xlabel("Time to target, s",fontsize=14)
	# ax1.set_title("Across days",fontsize=14)
	##now plot the modulation depth
	means = np.array([np.nanmean(mod_depth_early),np.nanmean(mod_depth_late)])
	serr = np.array([np.nanstd(mod_depth_early)/np.sqrt(mod_depth_early.shape[0]),
		np.nanstd(mod_depth_late)/np.sqrt(mod_depth_late.shape[0])])
	##now do the plotting
	x2 = np.array([0,1])
	err_x = np.array([0.2,0.2])
	# for i in range(all_mod.shape[1]):
	# 	ax2.plot(x2,all_mod[:,i],color='k',linewidth=2,marker='o')
	# ax2.errorbar(x2,means,yerr=serr,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	bars = ax2.bar(x2,means,0.5,color = ['cyan','blue'],yerr=serr,ecolor='k',alpha=1)
	plt.xticks(np.arange(0,2),['Early','Late'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.5,1.5)
	# ax2.set_ylim(0,5.5)
	ax2.set_ylabel("Modulation depth",fontsize=14)
	ax2.set_title("Modulation depth",fontsize=14)
	fig.suptitle("Across days",fontsize=14)
	tval,pval = stats.ttest_ind(mod_depth_early,mod_depth_late,nan_policy='omit')
	ax2.text(0,3.4,"p={}".format(pval))
	ax2.text(0,3.6,"t={}".format(tval))
	plt.tight_layout()
	##repeat for within sesssions
	group = f['within_session_all']
	early = np.asarray(group['early'])
	late = np.asarray(group['late'])
	early_mean = np.nanmean(early,axis=0)
	early_sem = np.nanstd(early,axis=0)/np.sqrt(early.shape[0])
	late_mean = np.nanmean(late,axis=0)
	late_sem = np.nanstd(late,axis=0)/np.sqrt(late.shape[0])
	mod_depth_early = np.abs(np.nanmax(early[:,125:175],axis=1)-np.nanmin(early[:,125:175],axis=1))
	mod_depth_late = np.abs(np.nanmax(late[:,125:175],axis=1)-np.nanmin(late[:,125:175],axis=1))
	fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1)
	x = np.linspace(-6,2,early.shape[1])
	ax1.plot(x,early_mean,color='cyan',linewidth=2,label='early sessions')
	ax1.plot(x,late_mean,color='blue',linewidth=2,label='late sessions')
	ax1.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='cyan',alpha=0.5)
	ax1.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='blue',alpha=0.5)
	ax1.set_ylabel("z-score",fontsize=14)
	ax1.set_xlabel("Time to target, s",fontsize=14)
	# ax1.set_title("Within sessions",fontsize=14)
	##now plot the modulation depth
	means = np.array([np.nanmean(mod_depth_early),np.nanmean(mod_depth_late)])
	serr = np.array([np.nanstd(mod_depth_early)/np.sqrt(mod_depth_early.shape[0]),
		np.nanstd(mod_depth_late)/np.sqrt(mod_depth_late.shape[0])])
	##now do the plotting
	x2 = np.array([0,1])
	err_x = np.array([0.2,0.2])
	# for i in range(all_mod.shape[1]):
	# 	ax2.plot(x2,all_mod[:,i],color='k',linewidth=2,marker='o')
	# ax2.errorbar(x2,means,yerr=serr,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	bars = ax2.bar(x2,means,0.5,color = ['cyan','blue'],yerr=serr,ecolor='k',alpha=1)
	plt.xticks(np.arange(0,2),['Early','Late'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.5,1.5)
	# ax2.set_ylim(0,5.5)
	ax2.set_ylabel("Modulation depth",fontsize=14)
	ax2.set_title("Modulation depth",fontsize=14)
	fig.suptitle("Within sessions",fontsize=14)
	tval,pval = stats.ttest_rel(mod_depth_early,mod_depth_late,nan_policy='omit')
	ax2.text(0,3.4,"p={}".format(pval))
	ax2.text(0,3.6,"t={}".format(tval))
	plt.tight_layout()
	##repeat for within session early
	group = f['within_session_early']
	early = np.asarray(group['late'])
	late = np.asarray(group['early'])
	early_mean = np.nanmean(early,axis=0)
	early_sem = np.nanstd(early,axis=0)/np.sqrt(early.shape[0])
	late_mean = np.nanmean(late,axis=0)
	late_sem = np.nanstd(late,axis=0)/np.sqrt(late.shape[0])
	mod_depth_early = np.abs(np.nanmax(early[:,135:152],axis=1)-np.nanmin(early[:,125:175],axis=1))
	mod_depth_late = np.abs(np.nanmax(late[:,135:152],axis=1)-np.nanmin(late[:,125:175],axis=1))
	fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1)
	x = np.linspace(-6,2,early.shape[1])
	ax1.plot(x,early_mean,color='cyan',linewidth=2,label='early sessions')
	ax1.plot(x,late_mean,color='blue',linewidth=2,label='late sessions')
	ax1.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='cyan',alpha=0.5)
	ax1.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='blue',alpha=0.5)
	ax1.set_ylabel("z-score",fontsize=14)
	ax1.set_xlabel("Time to target, s",fontsize=14)
	# ax1.set_title("Within sessions",fontsize=14)
	##now plot the modulation depth
	means = np.array([np.nanmean(mod_depth_early),np.nanmean(mod_depth_late)])
	serr = np.array([np.nanstd(mod_depth_early)/np.sqrt(mod_depth_early.shape[0]),
		np.nanstd(mod_depth_late)/np.sqrt(mod_depth_late.shape[0])])
	##now do the plotting
	x2 = np.array([0,1])
	err_x = np.array([0.2,0.2])
	# for i in range(all_mod.shape[1]):
	# 	ax2.plot(x2,all_mod[:,i],color='k',linewidth=2,marker='o')
	# ax2.errorbar(x2,means,yerr=serr,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	bars = ax2.bar(x2,means,0.5,color = ['cyan','blue'],yerr=serr,ecolor='k',alpha=1)
	plt.xticks(np.arange(0,2),['Early','Late'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.5,1.5)
	# ax2.set_ylim(0,5.5)
	ax2.set_ylabel("Modulation depth",fontsize=14)
	ax2.set_title("Modulation depth",fontsize=14)
	fig.suptitle("Within early sessions",fontsize=14)
	tval,pval = stats.ttest_rel(mod_depth_early,mod_depth_late,nan_policy='omit')
	ax2.text(0,2.4,"p={}".format(pval))
	ax2.text(0,2.6,"t={}".format(tval))
	plt.tight_layout()
	##repeat for within session late
	group = f['within_session_late']
	early = np.asarray(group['early'])
	late = np.asarray(group['late'])
	early_mean = np.nanmean(early,axis=0)
	early_sem = np.nanstd(early,axis=0)/np.sqrt(early.shape[0])
	late_mean = np.nanmean(late,axis=0)
	late_sem = np.nanstd(late,axis=0)/np.sqrt(late.shape[0])
	mod_depth_early = np.abs(np.nanmax(early[:,125:175],axis=1)-np.nanmin(early[:,125:175],axis=1))
	mod_depth_late = np.abs(np.nanmax(late[:,125:175],axis=1)-np.nanmin(late[:,125:175],axis=1))
	fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1)
	x = np.linspace(-6,2,early.shape[1])
	ax1.plot(x,early_mean,color='cyan',linewidth=2,label='early sessions')
	ax1.plot(x,late_mean,color='blue',linewidth=2,label='late sessions')
	ax1.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='cyan',alpha=0.5)
	ax1.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='blue',alpha=0.5)
	ax1.set_ylabel("z-score",fontsize=14)
	ax1.set_xlabel("Time to target, s",fontsize=14)
	# ax1.set_title("Within sessions",fontsize=14)
	##now plot the modulation depth
	means = np.array([np.nanmean(mod_depth_early),np.nanmean(mod_depth_late)])
	serr = np.array([np.nanstd(mod_depth_early)/np.sqrt(mod_depth_early.shape[0]),
		np.nanstd(mod_depth_late)/np.sqrt(mod_depth_late.shape[0])])
	##now do the plotting
	x2 = np.array([0,1])
	err_x = np.array([0.2,0.2])
	# for i in range(all_mod.shape[1]):
	# 	ax2.plot(x2,all_mod[:,i],color='k',linewidth=2,marker='o')
	# ax2.errorbar(x2,means,yerr=serr,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	bars = ax2.bar(x2,means,0.5,color = ['cyan','blue'],yerr=serr,ecolor='k',alpha=1)
	plt.xticks(np.arange(0,2),['Early','Late'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.5,1.5)
	# ax2.set_ylim(0,5.5)
	ax2.set_ylabel("Modulation depth",fontsize=14)
	ax2.set_title("Modulation depth",fontsize=14)
	fig.suptitle("Within late sessions",fontsize=14)
	tval,pval = stats.ttest_rel(mod_depth_early,mod_depth_late,nan_policy='omit')
	ax2.text(0,3.4,"p={}".format(pval))
	ax2.text(0,3.6,"t={}".format(tval))
	plt.tight_layout()
	f.close()

def get_ensemble_correlations_new(dt=1):
	##define some gobal parameters
	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13", "R7", "R8"]
	##range of sessions that are valid for this particular analysis (ie no CD testing)
	ranges = [[0,8], [0,13], [0,11], [0,11], [0,11], [0,11], [0,11], [0,8], [0,7], [5,8], [5,8]] 
	##window to smooth over
	early_win = [0,5*60*1000]
	late_win = [-15*60*1000:-10*60*1000]
	sig_type = "units"
	root_dir = "/Volumes/data/V1_BMI"
	save_file = "/Volumes/data/V1_BMI/Neuron/data/correlations_1ms.hdf5"
	if animal_list is None:
		animal_list = ru.animals.keys()
	for r,animal in enumerate(animal_list):
		session_list = [x for x in ru.animals[animal][1].keys() if int(x[-6:-4]) in range(ranges[r][0],ranges[r][1])]
		for session in session_list:
			f_out = h5py.File(save_file,'a')
			try:
				a_group = f_out[animal]
			except KeyError:
				a_group = f_out.create_group(animal)
			try:
				s_group = a_group[session]
				f_out.close()
			except KeyError:
				f_out.close()
				##this should be the path to the plexon file
				plxfile = os.path.join(root_dir,animal,session)
				##try to get the list of names for the signals and the targets
				try:
					e1_list = ru.animals[animal][1][session][sig_type]["e1_units"]
				except KeyError:
					e1_list = []
				try:
					e2_list = ru.animals[animal][1][session][sig_type]["e2_units"]
				except KeyError:
					e2_list = []
				try:
					ind_list = ru.animals[animal][1][session][sig_type]["V1_units"]
				except KeyError:
					ind_list = []
				##if we do have everything we need, continue
				if (len(e1_list)>0 and len(e2_list)>0 and len(ind_list)>0):
					print("working on "+animal+" "+session)
					##open the raw plexon data file
					raw_data = plxread.import_file(plxfile,AD_channels=range(0,96),save_wf=False,
						import_unsorted=False,verbose=False)
					##this will be our list of correlation timecourses
					e1_data_early = []
					e1_data_late = []
					e2_data_early = []
					e2_data_late = []
					ind_data_early = []
					ind_data_late = []
					##get the info about the duration of this session
					lfp_id = ru.animals[animal][1][session]['lfp']['V1_lfp'][0]
					duration = int((np.ceil(raw_data[lfp_id+'_ts'].max()*1000)/100)*100)+1
					##now process each set of units for this session
					##starting with E1 units:
					##a list of tuples containing all the pairwise combinations of e1 units
					##(necessary because some sessions have more than 2 ensemble units)
					unit_combinations = list(itertools.combinations(e1_list,2))
					for c in unit_combinations:
						##extract and process the data from the datafile
						unit1 = raw_data[c[0]]*1000.0
						unit1 = np.histogram(unit1,bins=duration,range=(0,duration))
						unit1 = unit1[0].astype(bool).astype(int)
						####
						unit2 = raw_data[c[1]]*1000.0
						unit2 = np.histogram(unit2,bins=duration,range=(0,duration))
						unit2 = unit2[0].astype(bool).astype(int)
						##now do a correlation analysis on the early and late chunks
						result_early = ss.corr_coeff2(unit1[early_win[0]:early_win[1]],
							unit2[early_win[0]:early_win[1]],dt)
						result_late = ss.corr_coeff2(unit1[late_win[0]:late_win[1]],
							unit2[late_win[0]:late_win[1]])
						e1_data_early.append(result_early)
						e1_data_late.append(result_late)
					##repeat for e2 units
					unit_combinations = list(itertools.combinations(e2_list,2))
					for c in unit_combinations:
						##extract and process the data from the datafile
						unit1 = raw_data[c[0]]*1000.0
						unit1 = np.histogram(unit1,bins=duration,range=(0,duration))
						unit1 = unit1[0].astype(bool).astype(int)
						####
						unit2 = raw_data[c[1]]*1000.0
						unit2 = np.histogram(unit2,bins=duration,range=(0,duration))
						unit2 = unit2[0].astype(bool).astype(int)
						##now do a correlation analysis on the early and late chunks
						result_early = ss.corr_coeff2(unit1[early_win[0]:early_win[1]],
							unit2[early_win[0]:early_win[1]],dt)
						result_late = ss.corr_coeff2(unit1[late_win[0]:late_win[1]],
							unit2[late_win[0]:late_win[1]])
						e2_data_early.append(result_early)
						e2_data_late.append(result_late)
					##repeat for indirect units
					unit_combinations = list(itertools.combinations(ind_list,2))
					for c in unit_combinations:
						##extract and process the data from the datafile
						unit1 = raw_data[c[0]]*1000.0
						unit1 = np.histogram(unit1,bins=duration,range=(0,duration))
						unit1 = unit1[0].astype(bool).astype(int)
						####
						unit2 = raw_data[c[1]]*1000.0
						unit2 = np.histogram(unit2,bins=duration,range=(0,duration))
						unit2 = unit2[0].astype(bool).astype(int)
						##now do a correlation analysis on the early and late chunks
						result_early = ss.corr_coeff2(unit1[early_win[0]:early_win[1]],
							unit2[early_win[0]:early_win[1]],dt)
						result_late = ss.corr_coeff2(unit1[late_win[0]:late_win[1]],
							unit2[late_win[0]:late_win[1]])
						ind_data_early.append(result_early)
						ind_data_late.append(result_late)
					f_out = h5py.File(save_file,'a')
					a_group = f_out[animal]	
					s_group = a_group.create_group(session)
					s_group.create_dataset("E1_early",data = np.asarray(e1_data_early))
					s_group.create_dataset("E1_late",data = np.asarray(e1_data_late))
					s_group.create_dataset("E2_early",data = np.asarray(e2_data_early))
					s_group.create_dataset("E2_late",data = np.asarray(e2_data_late))
					s_group.create_dataset("indirect_early",data = np.asarray(ind_data_early))
					s_group.create_dataset("indirect_late",data = np.asarray(ind_data_late))
					f_out.close()
	print("Done!")


def plot_correlations_new():
	##here we want to do 2 types of plots. 
	#1) mean corr across sessions
	#2) corr within sessions for jaws and non-jaws days
	##star by opening the data file 
	source_file = "/Volumes/data/V1_BMI/Neuron/data/correlations_10ms.hdf5"
	dt = "10 ms bins"
	f = h5py.File(source_file,'r')
	##create some lists to store the different types of data
	all_e1 = []
	all_e2 = []
	all_indirect = []
	animal_e1 = []
	animal_e2 = []
	animal_indirect = []
	across_sessions_e1 = []
	across_sessions_e2 = []
	across_sessions_indirect = []
	for animal in f.keys():
		e1_means = []
		e2_means = []
		indirect_means = []
		e1_sessions = []
		e2_sessions = []
		indirect_sessions = []
		sessions = f[animal].keys()
		for session in sessions:
			e1_data = np.asarray(f[animal][session]['E1']).squeeze()
			e2_data = np.asarray(f[animal][session]['E2']).squeeze()
			indirect_data = np.asarray(f[animal][session]['indirect']).squeeze()
			##get the mean over the whole session
			e1_means.append(e1_data.mean())
			e2_means.append(e2_data.mean())
			indirect_means.append(indirect_data.mean())
			##get the mean across the session
			if len(e1_data.shape) > 1:
				e1_sessions.append(e1_data.mean(axis=0))
				all_e1.append(e1_data.mean(axis=0))
			else:
				e1_sessions.append(e1_data)
				all_e1.append(e1_data)
				##
			if len(e2_data.shape) > 1:
				e2_sessions.append(e2_data.mean(axis=0))
				all_e2.append(e2_data.mean(axis=0))
			else:
				e2_sessions.append(e2_data)
				all_e2.append(e2_data)
				##
			if len(indirect_data.shape) > 1:
				indirect_sessions.append(indirect_data.mean(axis=0))
				all_indirect.append(indirect_data.mean(axis=0))
			else:
				indirect_sessions.append(indirect_data)
				all_indirect.append(indirect_data)
		##add this data to the master lists
		across_sessions_e1.append(np.asarray(e1_means))
		across_sessions_e2.append(np.asarray(e2_means))
		across_sessions_indirect.append(np.asarray(indirect_means))
		animal_e1.append(np.asarray(equalize_arrs(e1_sessions)).mean(axis=0))
		animal_e2.append(np.asarray(equalize_arrs(e2_sessions)).mean(axis=0))
		animal_indirect.append(np.asarray(equalize_arrs(indirect_sessions)).mean(axis=0))
	f.close()
	##now we need to equalize the lengths of our arrays
	all_e1 = np.asarray(equalize_arrs(all_e1))
	all_e2 = np.asarray(equalize_arrs(all_e2))
	all_indirect = np.asarray(equalize_arrs(all_indirect))
	animal_e1 = np.asarray(equalize_arrs(animal_e1))
	animal_e2 = np.asarray(equalize_arrs(animal_e2))
	animal_indirect = np.asarray(equalize_arrs(animal_indirect))
	across_sessions_e1 = np.asarray(equalize_arrs(across_sessions_e1))
	across_sessions_e2 = np.asarray(equalize_arrs(across_sessions_e2))
	across_sessions_indirect = np.asarray(equalize_arrs(across_sessions_indirect))
	###finally, we can plot these things
	session_data = [all_e1,all_e2,all_indirect]
	animal_data = [animal_e1,animal_e2,animal_indirect]
	across_data = [across_sessions_e1,across_sessions_e2,across_sessions_indirect]
	colors = ['g','b','k']
	labels = ['E1','E2','Indirect']
	##start with the within session data
	x = np.linspace(0,60,all_e1.shape[1])
	fig,ax = plt.subplots(1)
	for i, dataset in enumerate(session_data):
		mean = np.nanmean(dataset,axis=0)
		sem = np.nanstd(dataset,axis=0)/np.sqrt(dataset.shape[0])
		ax.plot(x,mean,linewidth=2,color=colors[i],label=labels[i])
		ax.fill_between(x,mean-sem,mean+sem,color=colors[i],alpha=0.5)
	ax.set_xlabel("Time in session, mins",fontsize=16)
	ax.set_ylabel("Correlation coefficient",fontsize=16)
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax.legend()
	fig.suptitle("Correlations by session, "+dt)
	##now look at it by animal
	x = np.linspace(0,60,animal_e1.shape[1])
	fig,ax = plt.subplots(1)
	for i, dataset in enumerate(animal_data):
		mean = np.nanmean(dataset,axis=0)
		sem = np.nanstd(dataset,axis=0)/np.sqrt(dataset.shape[0])
		ax.plot(x,mean,linewidth=2,color=colors[i],label=labels[i])
		ax.fill_between(x,mean-sem,mean+sem,color=colors[i],alpha=0.5)
	ax.set_xlabel("Time in session, mins",fontsize=16)
	ax.set_ylabel("Correlation coefficient",fontsize=16)
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax.legend()
	fig.suptitle("Correlations by animal, "+dt)
	##now do the across session plot
	fig,ax = plt.subplots(1)
	for i, dataset in enumerate(across_data):
		x = np.arange(1,dataset.shape[1]+1)
		mean = np.nanmean(dataset,axis=0)
		sem = np.nanstd(dataset,axis=0)/np.sqrt(dataset.shape[0])
		ax.plot(x,mean,linewidth=2,color=colors[i],label=labels[i])
		ax.fill_between(x,mean-sem,mean+sem,color=colors[i],alpha=0.5)
	ax.set_xlabel("Traning day",fontsize=16)
	ax.set_ylabel("Mean correlation coefficient",fontsize=16)
	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(16)
	ax.legend()
	ax.set_title("Correlations across training, "+dt,fontsize=16)
	ax.plot(np.arange(1,8),np.ones(7)*-0.045,linewidth=4,color='r')
	ax.text(3,-0.04,"Jaws",fontsize=14)
	ax.set_xlim(0,13)

## a function to plot the performance of DLS animals
def plot_DLS_performance():
	##open the matlab file from Aaron
	datafile = "/Volumes/data/V1_BMI/Neuron/data/DLS_performance_all.mat"
	datamat = spio.loadmat(datafile)
	##convert this in to a recognizeable array
	data = np.zeros([4,7])
	##the names of the animals' data in this matrix
	vectors = [x for x in list(datamat) if x.startswith('a')]
	##add to data array
	for i,v in enumerate(vectors):
		data[i,:] = datamat[v][0,:]
	##start with across days
	mean = data.mean(axis=0)
	sem = stats.sem(data,axis=0)
	x_axis = np.arange(1,8)
	fig, ax = plt.subplots(1)
	ax.errorbar(x_axis,mean,yerr=sem,linewidth=2,color='orange')
	ax.set_xlabel("Training day",fontsize=14)
	ax.set_ylabel("Percentage correct",fontsize=14)
	ax.set_title("DLS inhibition")
	ax.set_xlim(0,9)
	for animal in range(data.shape[0]):
		plt.plot(x_axis,data[animal,:],color='orange',
			linewidth=1,alpha=0.5)
	##now we do the line plot version
	early = data[:,0:3].mean(axis=1)
	late = data[:,4:7].mean(axis=1)
	means = np.array([early.mean(),late.mean()])
	sems = np.array([early.std()/2,late.std()/2])
	labels = ["DLS early","DLS late"]
	fig, ax2 = plt.subplots(1)
	DLS = np.vstack((early,late))
	xd = np.array([0,1])
	for i in range(early.shape[0]):
		ax2.plot(xd,DLS[:,i],color='orange',linewidth=2,marker='o')
	err_x = np.arange(0,2)
	yerr=sems
	xerr=np.ones(2)*0.25
	ax2.errorbar(err_x,means,yerr=yerr,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	plt.xticks(np.arange(0,2),labels)
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.35,1.35)
	ax2.set_ylim(0.0,1)
	ax2.set_ylabel("percentage correct", fontsize = 14)
	ax2.set_xlabel("Training period", fontsize = 14)
	##get the stats
	t_earlyLate,p_earlyLate = stats.ttest_rel(early,late)	
	##now print the stats
	print("mean early = "+str(means[0]))
	print("mean late = "+str(means[1]))
	print("P-val early vs late= "+str(p_earlyLate))
	print("T-val early vs late= "+str(t_earlyLate))

##totally redo-ing the DMS analysis because they might be interneurons. UGH!
def fetch_DMS_triggered():
	unit_type = 'Str_units' ##the type of units to look at on
	##data location
	root_dir = "/Volumes/data/V1_BMI"
	##define some gobal parameters
	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13"]
	##range of sessions that are valid for this particular analysis (ie no CD testing)
	ranges = [[0,8], [0,13], [0,11], [0,11], [0,11], [0,11], [0,11], [0,8], [0,7]] 
	##size of analysis window, in ms
	window = [10000,4000]
	##range of spike rates to allow in analysis
	cutoff_hz = [10,100]
	save_file = "/Volumes/data/V1_BMI/Neuron/Neuron_V2/data/triggered_dms_interneurons.hdf5"
	##create blank file
	f_out = h5py.File(save_file,'a')
	f_out.close()
	for r,animal in enumerate(animal_list):
		session_list = [x for x in ru.animals[animal][1].keys() if int(x[-6:-4]) in range(ranges[r][0],ranges[r][1])]
		for session in session_list:
			##reopen the save file
			f_out = h5py.File(save_file,'a')
			#see if a group exists for this animal
			try:
				a_group = f_out[animal]
			##if not, create it
			except KeyError:
				a_group = f_out.create_group(animal)
			##see if a session group exists for this animal
			try:
				##if so, move on to the next session
				s_group = a_group[session]
				f_out.close()
			except KeyError:
				##if not, close the file while we do the analysis
				f_out.close()
				print("Working on "+animal+" "+session)
				filepath = os.path.join(root_dir,animal,session) ##the file path to the session data
				##open the file
				data = plxread.import_file(filepath,AD_channels=range(1,97),save_wf=False,
					import_unsorted=False,verbose=False)
				##what is the duration of this file
				duration = None
				for arr in data.keys():
					if arr.startswith('AD') and arr.endswith('_ts'):
						duration = int(np.ceil((data[arr].max()*1000)/100)*100)+1
						break
				else: print("No A/D timestamp data found!!!")
				##we are going to need the T1 timestamps fo each file
				try:
					t1_id = ru.animals[animal][1][session]['events']['t1'][0] ##the event name in the plexon file
					t1_ts = data[t1_id]*1000.0
				except KeyError:
					t1_ts = np.array([])
				try:
					t2_id = ru.animals[animal][1][session]['events']['t2'][0] ##the event name in the plexon file
					t2_ts = data[t2_id]*1000.0
				except KeyError:
					t2_ts = np.array([])
				##now try to get data for the appropriate unit type
				try:
					unit_ids = ru.animals[animal][1][session]['units'][unit_type]
					##create a container for the unit data
					dataset_t1 = []
					dataset_t2 = []
					n_in_range = 0 ##keep track of how many units are within the rate requirements
					for unit in unit_ids:
						##get the timestamps for this unit from the datafile
						unit_ts = data[unit]*1000.0
						##binary transform
						spiketrain = np.histogram(unit_ts,bins=duration,range=(0,duration))
						spiketrain = spiketrain[0].astype(bool).astype(int)
						##perform the rate check
						if (get_spike_rate(spiketrain) > cutoff_hz[0] and get_spike_rate(spiketrain) < cutoff_hz[1]):
							n_in_range += 1
							##now get the triggered data for this unit
							if len(t1_ts)>0:
								traces_t1 = get_data_window(spiketrain,t1_ts,
									window[0],window[1])
								if traces_t1 is not None:
									dataset_t1.append(traces_t1)
							if len(t2_ts)>0:
								traces_t2 = get_data_window(spiketrain,t2_ts,
									window[0],window[1])
								if traces_t2 is not None:
									dataset_t2.append(traces_t2)
					##now turn the dataset into an array, with first axis = n_units
					dataset_t1 = np.asarray(dataset_t1)
					dataset_t2 = np.asarray(dataset_t2)
					##if there is actually data here, add it to the save file
					if n_in_range > 0:
						##reopen the save file, now that it is safe (we have the data; plxread won't cause a crash)
						f_out = h5py.File(save_file,'a')
						##we already made sure an animal group exists, so get a handle
						a_group = f_out[animal]
						#we also know that a session group does NOT exist, so create it
						s_group = a_group.create_group(session)
						##now save the datasets to the file
						if len(dataset_t1>0):
							s_group.create_dataset('t1',data=dataset_t1)
						if len(dataset_t2)>0:
							s_group.create_dataset("t2",data=dataset_t2)
						f_out.close()
					else:
						print("No units in range for this session")
				except KeyError:
					print("No "+unit_type+" for "+animal+" "+session)
	print('Done')

## a function to re-plot the triggered DMS spike rates, separated by putative unit type
def parse_dms_triggered_new():
	##file path of projection units
	f_projection = "/Volumes/data/V1_BMI/Neuron/Neuron_V2/data/triggered_dms_projection_neurons.hdf5"
	##file path of interneurons
	f_interneuron = "/Volumes/data/V1_BMI/Neuron/Neuron_V2/data/triggered_dms_interneurons.hdf5"
	##the file to save the data in
	save_file = "/Volumes/data/V1_BMI/Neuron/Neuron_V2/data/processed_dms_interneurons_raw.hdf5"
	##data ranges to use for early and late
	early_session_range = np.arange(0,4)
	late_session_range = np.arange(7,15)
	##selection of data type to use
	datafile = f_interneuron
	##open the data file 
	f = h5py.File(datafile,'r')
	##create the save file
	f_out = h5py.File(save_file,'a')
	f_out.close()
	##get the list of animals for this data file
	animal_list = list(f)
	for animal in animal_list:
		##first check if there is any data for this animal
		session_list = list(f[animal])
		if len(session_list) > 0:
			##split sessions into early and late
			early_list = [x for x in session_list if int(x[-6:-4]) in early_session_range]
			late_list = [x for x in session_list if int(x[-6:-4]) in late_session_range]
			##if there is data for both early and late, open the save file and create a data group
			if (len(early_list)>0 and len(late_list)>0):
				f_out = h5py.File(save_file,'a')
				a_group = f_out.create_group(animal)
				##now run through data from each session period
				##container for processed mean traces for all units recorded in the early period
				early_traces = []
				late_traces = []
				for session in early_list:
					##get the dataset; shape is units x bins(ms) x trials
					data = np.asarray(f[animal][session]['t1'])
					##now we want to get the mean smoothed trace for each unit.
					for unit in range(data.shape[0]):
						unit_data = data[unit,:,:] ##shape bins x trials
						##now we do the smoothing; need to invert here to be compatible
						smoothed = smooth_spikes(unit_data.T,smooth_method='both',smooth_width=[80,40]).T
						##data  here is still bins x trials (but now many fewer bins)
						#now take the mean across trials for this unit and add to the dataset
						early_traces.append(smoothed.mean(axis=1))
				##now convert to an array
				early_traces = np.asarray(early_traces)
				for session in late_list:
					##get the dataset; shape is units x bins(ms) x trials
					data = np.asarray(f[animal][session]['t1'])
					##now we want to get the mean smoothed trace for each unit.
					for unit in range(data.shape[0]):
						unit_data = data[unit,:,:] ##shape bins x trials
						##now we do the smoothing; need to invert here to be compatible
						smoothed = smooth_spikes(unit_data.T,smooth_method='both',smooth_width=[80,40]).T
						##data  here is still bins x trials (but now many fewer bins)
						#now take the mean across trials for this unit and add to the dataset
						late_traces.append(smoothed.mean(axis=1))
				##now convert to an array
				late_traces = np.asarray(late_traces)
				##now save these data
				a_group.create_dataset("early_traces",data=early_traces)
				a_group.create_dataset("late_traces",data=late_traces)
				f_out.close()
	print("Done!")

##plot the data we gathered in the previous 2 functions
def plot_dms_locked_2():
	datafile = "/Volumes/data/V1_BMI/Neuron/Neuron_V2/data/processed_dms_projection_neurons_zscore.hdf5"
	unit_type = "DMS projection neurons"
	f = h5py.File(datafile,'r')
	animal_list = list(f)
	##containers for mean data for each animal
	early = []
	late = []
	for animal in animal_list:
		# early.append(np.asarray(f[animal]['early_traces']))
		# late.append(np.asarray(f[animal]['late_traces']))
		early.append(np.asarray(f[animal]['early_traces']).mean(axis=0))
		late.append(np.asarray(f[animal]['late_traces']).mean(axis=0))
	f.close()
	early = np.asarray(early)
	late = np.asarray(late)
	# early = np.vstack(early)
	# late = np.vstack(late)
	early_mean = np.nanmean(early,axis=0)
	early_sem = np.nanstd(early,axis=0)/np.sqrt(early.shape[0])
	late_mean = np.nanmean(late,axis=0)
	late_sem = np.nanstd(late,axis=0)/np.sqrt(late.shape[0])
	mod_depth_early = np.abs(np.nanmax(early[:,215:255],axis=1)-np.nanmin(early[:,255:],axis=1))
	mod_depth_late = np.abs(np.nanmax(late[:,215:255],axis=1)-np.nanmin(late[:,255:],axis=1))
	fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1)
	x = np.linspace(-10,4,early.shape[1])
	ax1.plot(x,early_mean,color='orange',linewidth=2,label='early sessions')
	ax1.plot(x,late_mean,color='red',linewidth=2,label='late sessions')
	ax1.fill_between(x,early_mean-early_sem,early_mean+early_sem,color='orange',alpha=0.5)
	ax1.fill_between(x,late_mean-late_sem,late_mean+late_sem,color='red',alpha=0.5)
	ax1.set_ylabel("Spike rate, Z-score",fontsize=14)
	ax1.set_xlabel("Time to target, s",fontsize=14)
	##now plot the modulation depth
	means = np.array([np.nanmean(mod_depth_early),np.nanmean(mod_depth_late)])
	serr = np.array([np.nanstd(mod_depth_early)/np.sqrt(mod_depth_early.shape[0]),
		np.nanstd(mod_depth_late)/np.sqrt(mod_depth_late.shape[0])])
	##now do the plotting
	x2 = np.array([0,1])
	err_x = np.array([0.2,0.2])
	# for i in range(all_mod.shape[1]):
	# 	ax2.plot(x2,all_mod[:,i],color='k',linewidth=2,marker='o')
	# ax2.errorbar(x2,means,yerr=serr,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	bars = ax2.bar(x2,means,0.5,color = ['orange','red'],yerr=serr,ecolor='k',alpha=1)
	plt.xticks(np.arange(0,2),['Early','Late'])
	for ticklabel in ax2.get_xticklabels():
		ticklabel.set_fontsize(14)
	for ticklabel in ax2.get_yticklabels():
		ticklabel.set_fontsize(14)
	ax2.set_xlim(-0.5,1.5)
	# ax2.set_ylim(0,5.5)
	ax2.set_ylabel("Modulation depth",fontsize=14)
	ax2.set_title("Modulation depth",fontsize=14)
	fig.suptitle(unit_type,fontsize=14)
	tval,pval = stats.ttest_ind(mod_depth_early,mod_depth_late,nan_policy='omit')
	ax2.text(0.6,means[0]/2.0,"p={}".format(pval))
	ax2.text(0.6,means[0]/2.0+0.1,"t={}".format(tval))
	plt.tight_layout()

##re-do the linear regression, but now place controls on the range of spike rates to use in the 
##analysis from the DMS unit side of things. This allows separation of interneurons and projection neurons
def linear_regression_dms():
	direct_types = ['e1_units','e2_units']
	unit_type = 'Str_units' ##the type of units to look at on
	##data location
	root_dir = "/Volumes/data/V1_BMI"
	##define some gobal parameters
	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13"]
	##range of sessions that are valid for this particular analysis (ie no CD testing)
	ranges = [[0,8], [0,13], [0,11], [0,11], [0,11], [0,11], [0,11], [0,8], [0,7]] 
	window = [400,0]
	bin_width = 50
	##make some dictionaries to store the results
	results = {}
	##we should be able to run regression for each session as a whole.
	##first, we need to get two arrays: X; the data matrix of spike data
	##in dimensions trials x units x bins, and then y; the binary matrix
	## of target 1 and target 2 values.
	cutoff_hz = [0,10]
	save_file = "/Volumes/data/V1_BMI/Neuron/Neuron_V2/data/linear_regression_dms_projection_neurons.hdf5"
	##create blank file
	f_out = h5py.File(save_file,'a')
	f_out.close()
	for r,animal in enumerate(animal_list):
		session_list = [x for x in ru.animals[animal][1].keys() if int(x[-6:-4]) in range(ranges[r][0],ranges[r][1])]
		for session in session_list:
			##check to see if data for this session already exists
			try:
				f_out = h5py.File(save_file,'a')
				session_exists = f_out[animal][session]
				print(animal+" "+session+" data exists; moving to next file")
				f_out.close()
			except KeyError:
				f_out.close()
				print("analyzing data from "+animal+" "+session)
				try:
					t1_id = ru.animals[animal][1][session]['events']['t1'][0]
				except KeyError:
					t1_id = None
				try:
					t2_id = ru.animals[animal][1][session]['events']['t2'][0]
				except KeyError:
					t2_id = None
				##continue only if there are 2 targets for this file
				if t2_id is not None and t1_id is not None:
					##see if the correct unit types are present in this file
					unit_list = []
					try:
						unit_ids = ru.animals[animal][1][session]['units'][unit_type]
						for ui in unit_ids:
							unit_list.append(ui)
					except KeyError:
						print("No "+ut+" present in this file")
					direct_list = []
					for ut in direct_types:
						try:
							unit_ids = ru.animals[animal][1][session]['units'][ut]
							for ui in unit_ids:
								direct_list.append(ui)
						except KeyError:
							print("No "+ut+" present in this file")
					##continue only if there are units to analyze
					if len(unit_list)>0 and len(direct_list)>0:
						plxfile = os.path.join(root_dir,animal,session)
						##load the plx file
						raw_data = plxread.import_file(plxfile,AD_channels=range(1,97),save_wf=False,
								import_unsorted=False,verbose=False)
						duration = None
						for arr in raw_data.keys():
							if arr.startswith('AD') and arr.endswith('_ts'):
								duration = int(np.ceil((raw_data[arr].max()*1000)/100)*100)+1
								break
						else: print("No A/D timestamp data found!!!")
						##make sure that this file had at least 20 trials of each type
						try:
							t1_ts = raw_data[t1_id]*1000.0
						except KeyError:
							t1_ts = np.array([])
						try: 
							t2_ts = raw_data[t2_id]*1000.0
						except KeyError:
							t2_ts = np.array([])
						if (t1_ts.size >= 1) and (t2_ts.size >= 1):
						##now load the data arrays for each direct unit
							y = []
							for unit in direct_list:
								try:
									spiketrain = raw_data[unit]*1000.0 #timestamps in ms
									##convert to binary array
									spiketrain = np.histogram(spiketrain,bins=duration,range=(0,duration))
									spiketrain = spiketrain[0].astype(bool).astype(int)
									spike_windows_t1 = get_data_window(spiketrain,t1_ts,window[0],window[1])
									spike_windows_t2 = get_data_window(spiketrain,t2_ts,window[0],window[1]) ##size bins x trials
									y.append(np.concatenate((spike_windows_t1,spike_windows_t2),axis=1)) ##concatenate trials
								except KeyError:
									print("Can't find "+unit+" in this file")
							##repeat for the other units
							X = []
							for unit in unit_list:
								try:
									spiketrain = raw_data[unit]*1000.0 #timestamps in ms
									##convert to binary array
									spiketrain = np.histogram(spiketrain,bins=duration,range=(0,duration))
									spiketrain = spiketrain[0].astype(bool).astype(int)
									##here is where we perform the spike rate check
									if (get_spike_rate(spiketrain) > cutoff_hz[0] and get_spike_rate(spiketrain) < cutoff_hz[1]):
										spike_windows_t1 = get_data_window(spiketrain,t1_ts,window[0],window[1])
										spike_windows_t2 = get_data_window(spiketrain,t2_ts,window[0],window[1]) ##size bins x trials
										X.append(np.concatenate((spike_windows_t1,spike_windows_t2),axis=1)) ##concatenate trials
								except KeyError:
									print("Can't find "+unit+" in this file")
							##make sure we still have more than 0 units
							if len(X)>0 and len(y)>0:
								X = np.asarray(X) ##shape should now be units x bins x trials
								y = np.asarray(y)
								##add the number of total units to the data array
								total_units = X.shape[0]
								##sum the spike counts over the interval, and transpose into trials x units
								X = X.sum(axis=1).T
								y = y.sum(axis=1).T
								##FINALLY, we can run the linear regression
								R2s,pvals = lin2.permutation_test_multi(X,y,n_iter_cv=5,n_iter_p=500)
								##now save these data
								f_out = h5py.File(save_file,'a')
								try:
									a_group = f_out[animal]
								except KeyError:
									a_group = f_out.create_group(animal)
								s_group = a_group.create_group(session)
								s_group.create_dataset('pvals',data=pvals)
								s_group.create_dataset('R2s',data=R2s)
								f_out.close()
	print("Done")

##function to plot the results from the above function
def plot_lin_regression():
	datafile = "/Volumes/data/V1_BMI/Neuron/Neuron_V2/data/linear_regression_dms_projection_neurons.hdf5"
	f = h5py.File(datafile,'r')
	animal_list = f.keys()
	##store the means of all the animals
	totals = []
	sig_vals = []
	var_explained = []
	for a in animal_list:
		tot = []
		r2 = []
		sig = []
		for s in f[a].keys():
			r2.append(np.asarray(f[a][s]['R2s']).mean())
			sig.append((np.asarray(f[a][s]['pvals'])<0.05).sum())
			tot.append(np.asarray(f[a][s]['R2s']).size)
		totals.append(np.asarray(tot))
		sig_vals.append(np.asarray(sig))
		var_explained.append(np.asarray(r2))
	f.close()
	totals = equalize_arrs(totals)
	sig_vals = equalize_arrs(sig_vals)
	var_explained = equalize_arrs(var_explained)
	x_axis = np.arange(1,totals.shape[1]+1)
	sig_perc = sig_vals/totals
	##now do the plots
	fig,(ax1,ax2,ax3) = plt.subplots(3,sharex=True)
	mean_total = np.nanmean(totals,axis=0)
	serr_total = np.nanstd(totals,axis=0)/np.sqrt(totals.shape[0])
	mean_var = np.nanmean(var_explained,axis=0)
	serr_var = np.nanstd(var_explained,axis=0)/np.sqrt(var_explained.shape[0])
	mean_sig = np.nanmean(sig_perc,axis=0)
	serr_sig = np.nanstd(sig_perc,axis=0)/np.sqrt(sig_perc.shape[0])
	ax1.set_ylabel("R2",fontsize=14)
	ax2.set_ylabel("Number\n of units",fontsize=14)
	ax3.set_ylabel("Percent\n significant",fontsize=14)
	ax3.set_xlabel("Training day",fontsize=14)
	ax1.set_title("Variance of direct units explained by DMS units",fontsize=14)
	ax2.set_title ("Total number of DMS units",fontsize=14)
	ax3.set_title("Percent direct units with significant prediction by indirect units",fontsize=14)
	ax1.errorbar(x_axis,mean_var,yerr=serr_var,color='k',linewidth=2)
	ax2.errorbar(x_axis,mean_total,yerr=serr_total,color='k',linewidth=2)
	ax3.errorbar(x_axis,mean_sig,yerr=serr_sig,color='k',linewidth=2)
	for tick in ax1.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for i in range(totals.shape[0]):
		ax1.plot(x_axis,var_explained[i,:],alpha=0.5,color='k')
		ax2.plot(x_axis,totals[i,:],alpha=0.5,color='k')
		ax3.plot(x_axis,sig_perc[i,:],alpha=0.5,color='k')
	##now let's get the summary for all days
	x = np.array([1,2])
	err_x = np.array([0.1])
	labels = ['mean R2','perc sig']
	ps_summ = np.nanmean(sig_perc,axis=1)
	r2_summ = np.nanmean(var_explained,axis=1)
	ps_mean = ps_summ.mean()
	ps_err = stats.sem(ps_summ)
	r2_mean = r2_summ.mean()
	r2_err = stats.sem(r2_summ)
	fig,ax = plt.subplots(1)
	ax2 = ax.twinx()
	ax2.errorbar(x[0],ps_mean,yerr=ps_err,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	ax.errorbar(x[1],r2_mean,yerr=r2_err,xerr=err_x,fmt='none',ecolor='k',capthick=2,elinewidth=2)
	for i in range(ps_summ.shape[0]):
		ax2.plot(x[0],ps_summ[i],marker='o',color='k')
	for i in range(r2_summ.shape[0]):
		ax.plot(x[1],r2_summ[i],marker='o',color='k')
	ax.set_xticks([1,2])
	ax.set_xticklabels(labels)
	ax.set_ylabel("R2",fontsize=14)
	ax2.set_ylabel("Percent significant",fontsize=14)
	ax.set_title("Direct unit activity predicted by indirect units",fontsize=14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	# ax1.set_xlim(0,12)
	#ax1.set_ylim(-1,0.4)
	# ax2.set_xlim(0,12)
	# ax3.set_xlim(0,12)






########################HELPERS~########################





"""
Another helper function to bin spike matrices.
Inputs should be in shape trials x units x bins
"""
def bin_matrix(X,bin_size):
	n_bins = int(np.floor(X.shape[2]/bin_size))
	binned_data = np.zeros((X.shape[0],X.shape[1],n_bins))
	##go unit by unit, and trial by trial
	for u in range(X.shape[1]):
		for t in range(X.shape[0]):
			binned_data[t,u,:] = bin_spikes(X[t,u,:],bin_size)
	return binned_data


"""
A helper function to bin arrays already in binary format
Inputs:
	data:1-d binary spike train
	bin_width: with of bins to use
Returns:
	1-d binary spike train with spike counts in each bin
"""
def bin_spikes(data,bin_width):
	bin_vals = []
	idx = 0
	while idx < data.size:
		bin_vals.append(data[idx:idx+bin_width].sum())
		idx += bin_width
	return np.asarray(bin_vals)

"""
A function to convolve data with a gaussian kernel of width sigma.
Inputs:
	array: the data array to convolve. Will work for multi-D arrays;
		shape of data should be samples x trials
	sigma: the width of the kernel, in samples
"""
def gauss_convolve(array, sigma):
	##remove singleton dimesions and make sure values are floats
	array = array.squeeze().astype(float)
	##allocate memory for result
	result = np.zeros(array.shape)
	##if the array is 2-D, handle each trial separately
	try:
		for trial in range(array.shape[1]):
			result[:,trial] = gaussian_filter(array[:,trial],sigma=sigma,order=0,
				mode="constant",cval = 0.0)
	##if it's 1-D:
	except IndexError:
		if array.shape[0] == array.size:
			result = gaussian_filter(array,sigma=sigma,order=0,mode="mirror")
		else:
			print("Check your array input to gaussian filter")
	return result

"""
A helper function to bin arrays already in binary format
Inputs:
	data:1-d binary spike train
	bin_width: with of bins to use
Returns:
	1-d binary spike train with spike counts in each bin
"""
def bin_spikes(data,bin_width):
	n_bins = int(data.size/bin_width)
	bin_vals = np.zeros(n_bins)
	for i in range(n_bins):
		bin_vals[i] = data[i*bin_width:(i+1)*bin_width].sum()
	return bin_vals

"""
a function to equalize the length of different-length arrays
by adding np.nans
Inputs:
	-list of arrays (1-d) of different shapes
Returns:
	2-d array of even size
"""
def equalize_arrs(arrlist):
	longest = 0
	for i in range(len(arrlist)):
		if arrlist[i].shape[0] > longest:
			longest = arrlist[i].shape[0]
	result = np.zeros((len(arrlist),longest))
	result[:] = np.nan
	for i in range(len(arrlist)):
		result[i,0:arrlist[i].shape[0]] = arrlist[i]
	return result

def get_data_window_lfp(lfp, centers, pre_win, post_win):
	verbose = True
	centers = np.squeeze(np.asarray(centers)).astype(np.int64)
	data = np.squeeze(lfp)
	N = data.size
	removed = 0
	try:
		for j, center in enumerate(centers):
			if center <= pre_win or center + post_win >= N:
				centers[j] = centers[j-1]
				removed +=1
				if verbose:
					print("Index too close to start or end to take a full window. Deleting event at "+str(center))
		if removed >= centers.size:
			traces = None
			print("No traces for this file")
		else:
			traces = np.zeros((pre_win+post_win, len(centers)))
			##the actual windowing functionality:
			for n, idx in enumerate(centers):
				trace = data[idx-pre_win:idx+post_win]
				traces[0:trace.shape[0],n] = trace
	except TypeError:
		traces = None
		print("No traces for this file")
	##error checking
	if traces != None:
		if traces.shape[0] != pre_win+post_win:
			traces = None
	return traces

def get_data_window(trace, centers, pre_win, post_win):
	if len(centers)>1:
		centers = np.squeeze(np.asarray(centers)).astype(np.int64)
	else:
		centers = np.asarray(centers).astype(np.int64)
	data = np.squeeze(trace)
	N = data.size
	##do some padding
	pad = np.zeros(pre_win+post_win)
	data = np.concatenate((pad,data,pad))
	##offset the centers by the same amount
	centers = centers+(pre_win+post_win)
	##now, make sure our centers are all within the range of our data length
	to_remove = []
	for j, center in enumerate(centers):
		if center >= data.size:
			to_remove.append(j)
			print("Event index exceeds data length. Deleting event at "+str(center))
	clean_centers = np.delete(centers,to_remove)
	if clean_centers.size == 0:
		traces = None
		print("No traces for this file")
	else:
		traces = np.zeros((pre_win+post_win, len(clean_centers)))
		##the actual windowing functionality:
		for n, idx in enumerate(clean_centers):
			traces[:,n] = data[idx-pre_win:idx+post_win]
	return traces

"""
Returns counts of events for one session, binned by minutes in the session
"""
def get_event_data(root_loc):
	e1_units = [x for x in list(root_loc['e1_units']) if not x.endswith('wf')]
	duration = int(np.ceil(root_loc['e1_units'][e1_units[0]].shape[1]/1000.0/60.0))
	t1_ts = np.asarray(root_loc['event_arrays']['t1'])/1000.0/60.0
	try:
		t2_ts = np.asarray(root_loc['event_arrays']['t2'])/1000.0/60.0
	except KeyError:
		t2_ts = np.array([])
	miss_ts = np.asarray(root_loc['event_arrays']['miss'])/1000.0/60.0
	t1 = np.histogram(t1_ts,bins=duration,range=(0,duration))[0]
	t2 = np.histogram(t2_ts,bins=duration,range=(0,duration))[0]
	miss = np.histogram(miss_ts,bins=duration,range=(0,duration))[0]
	p_correct = np.nan_to_num(t1/(t1+t2+miss))
	return t1,t2,miss,p_correct

"""
A function to remove NaNs from the end of an array
"""
def remove_nan(array):
	##find the first index of nan
	allnans = np.argwhere(np.isnan(array))
	if len(allnans)>0:
		firstnan = allnans.min()
		array = array[:firstnan]
	return array

"""
gets indices of stim trials or non-stim trials from 
a matlab file
"""
def parse_stim_data(matfile):
	data = spio.loadmat(matfile)
	stream = data['stim_trial_stream'].squeeze()
	sel_idx = np.where(np.logical_or(stream==0, stream==1))[0]
	stim_order = stream[sel_idx]
	stim_idx = np.where(stim_order==1)[0]
	off_idx = np.where(stim_order==0)[0]
	return stim_idx,off_idx

"""
A helper function to do spike smoothing. 
Inputs: 
	X: 2-d data array in the shape units x timebins (assuming 1 ms bins here),
		and value of each bin is a spike count)
	smooth_method: type of smoothing to use; choose 'bins', 'gauss', 'both', or 'none'
	smooth_width: size of the bins or gaussian kernel in ms. If both, input should be a list
		with index 0 being the gaussian width and index 1 being the bin width
	Returns: 
		X: smoothed version of X
"""
def smooth_spikes(X,smooth_method,smooth_width):
	if smooth_method == 'bins':
		Xbins = []
		for a in range(X.shape[0]):
			Xbins.append(bin_spikes(X[a,:],smooth_width))
		X = np.asarray(Xbins)
	elif smooth_method == 'gauss':
		for a in range(X.shape[0]):
			X[a,:] = gauss_convolve(X[a,:],smooth_width)
	elif smooth_method == 'both':
		##first smooth with a kernel 
		for a in range(X.shape[0]):
			X[a,:] = gauss_convolve(X[a,:],smooth_width[0])
		##now bin the data
		Xbins = []
		for a in range(X.shape[0]):
			Xbins.append(bin_spikes(X[a,:],smooth_width[1]))
		X = np.asarray(Xbins)
	elif smooth_method == 'none':
		pass
	else:
		raise KeyError("Unrecognized bin method")
		X = None
	return X

"""
A helper function to compute the spike rate of a binary spike train in Hz
(assumes input is binned into 1ms bins)
"""
def get_spike_rate(spiketrain):
	##total ms length of spiketrain
	duration_ms = spiketrain.size
	##total number of seconds for the recording
	duration_s = duration_ms/1000.0
	##now the total number of spikes per second
	spikerate = spiketrain.sum()/duration_s
	return spikerate




