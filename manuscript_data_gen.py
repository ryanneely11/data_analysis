import h5py 
#import DataSet3 as ds
import numpy as np
#import matplotlib.pyplot as plt
import os.path
import SpikeStats2 as ss
from scipy import stats
#import pandas as pd
#import seaborn as sns
import multiprocessing as mp
	
def get_performance_data():
	##functions to generate data for the manuscript
	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13", "R7", "R8"]
	##range of sessions that are valid for this particular analysis (ie no CD testing)
	ranges = [[0,8], [0,13], [0,11], [0,11], [0,11], [0,11], [0,11], [0,8], [0,7], [5,8], [5,8]] 
	##don't have a full set of session for my first two animals (R8 and R7), so exclude them from some analyses
	animal_list2 = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13"]
	##the main source file
	source_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5"
	##file to save the data
	save_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_learning_event_data_chosen.hdf5"
	f_out = h5py.File(save_file, 'w-')

	for i in range(len(animal_list)):
		print "Select " + animal_list[i]
		if animal_list[i] not in f_out.keys():	
			f_out.create_group(animal_list[i])
		else:
			print "Already created a group for " + animal_list[i]
		try:
			t1 = ds.load_event_arrays(source_file, "t1", session_range = ranges[i])
			f_out[animal_list[i]].create_dataset("t1", data = t1)
		except KeyError:
			print "No T1's in this file."
		try:
			t2 = ds.load_event_arrays(source_file, "t2", session_range = ranges[i])
			f_out[animal_list[i]].create_dataset("t2", data = t2)
		except KeyError:
			print "No T2's in this file."
		try:
			miss = ds.load_event_arrays(source_file, "miss", session_range = ranges[i])
			f_out[animal_list[i]].create_dataset("miss", data = miss)
		except KeyError: 
			print "No Misses in this file."

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
	print "Complete!"

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

	early = all_across[:,0:3].ravel()
	late = all_across[:, 6:9].ravel()
	mask = np.isnan(late)
	early = np.delete(early, np.where(mask))
	late = np.delete(late, np.where(mask))
	means = [early.mean(), late.mean()]
	stds = [early.std(), late.std()]
	sems = stds/np.sqrt(late.size)
	pval = stats.ttest_rel(early, late)[1]
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
			print "No t1's."
		try:
			t2 = np.asarray(f[animal]['t2'])
			arrays.append(t2)
		except KeyError:
			print "No t2's."
			t2 = np.zeros(t1.shape)
			arrays.append(t2)
		try:
			miss = np.asarray(f[animal]['miss'])
			arrays.append(miss)
		except KeyError:
			print "No Misses."
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
	print "Done!"
	return np.asarray(by_session), np.asarray(by_animal)


def plot_cd_data():
	f = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_CD_data.hdf5", 'r')
	data = np.asarray(f['by_animal'])
	f.close()
	means = np.array([data[:,0].mean(), data[:,1].mean(), data[:,2].mean()])
	sem = np.array([data[:,0].std(), data[:,1].std(), data[:,2].std()])/np.sqrt(data.shape[0])
	p_val_p_cd = stats.ttest_rel(data[:,0], data[:,1])[1]
	p_val_cd_r = stats.ttest_rel(data[:,1], data[:,2])[1]
	p_val_p_r = stats.ttest_rel(data[:,0], data[:,2])[1]
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
	print "Complete!"

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
	pval = stats.ttest_ind(t1, t2)[1]
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
	print "Done!"

def get_light_data():
	source_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5"
	destination_file = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_light_data.hdf5"
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
	print "Done!"
	
def plot_cr_data():
	f = h5py.File(r"Z:\Data\processed_data\V1_BMI_final\raw_data\R7_thru_V13_CR_data.hdf5", 'r')
	data = np.asarray(f['chunks_by_animal'])

	means = np.array([data[:,0].mean(), data[:,1].mean(), data[:,2].mean()])
	sems = np.array([data[:,0].std(), data[:,1].mean(), data[:,2].std()])/np.sqrt(data.shape[0])
	p_val = stats.ttest_rel(data[:,0], data[:,2])[1]
	p_val_cr = stats.ttest_rel(data[:,0], data[:,1])[1]
	p_val_r = stats.ttest_rel(data[:,1], data[:,2])[1]
	idx = np.array([0.3, 0.6, 0.9])
	width = 0.3
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
#	ax.axvspan(15, 28, alpha = 0.5, color = 'lightblue')
#	ax.axvspan(60, 75, alpha = 0.5, color = 'royalblue')
#	ax.axvspan(32, 48, alpha = 0.5, color = 'orange')
	plt.vlines(28, 0, 1, linestyle = 'dashed')	
	ax.set_xlabel("Time, mins", fontsize = 16)
	ax.set_ylabel("Percent of events", fontsize = 16)
	fig.suptitle("Performance during CR", fontsize = 18)
	ax.legend()


def plot_light_data():
	f = h5py.File(r"Z:\Data\processed_data\V1_BMI_final\raw_data\R7_thru_V13_light_data.hdf5", 'r')
	data = np.asarray(f['chunks_by_session'])

	means = np.array([data[:,0].mean(), data[:,1].mean()])
	sems = np.array([data[:,0].std(), data[:,1].std()])/np.sqrt(data.shape[0])
	p_val = stats.ttest_rel(data[:,0], data[:,1])[1]
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

##specifically retreives data for LATE in sessions
def get_triggered_spike_rates():
	try:
		ds.save_multi_group_triggered_data(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 
			r"J:\Ryan\processed_data\V1_BMI_final\raw_data\e1_t1_spikes_late.hdf5", "t1", ["e1_units", "spikes"], [6000,6000],
			chunk = [0,10])
	except IOError:
		print "File exists; skipping"
	try:
		ds.save_multi_group_triggered_data(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 
			r"J:\Ryan\processed_data\V1_BMI_final\raw_data\e2_t1_spikes_late.hdf5", "t1", ["e2_units", "spikes"], [6000,6000],
			chunk = [0,10])
	except IOError:
		print "File exists; skipping"
	try:
		ds.save_multi_group_triggered_data(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 
			r"J:\Ryan\processed_data\V1_BMI_final\raw_data\V1_t1_spikes_late.hdf5", "t1", ["V1_units", "spikes"], [6000,6000],
			chunk = [0,10])
	except IOError:
		print "File exists; skipping"
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
	print "Complete!"


def plot_triggered_spike_rates():
	f = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_triggered_data_early.hdf5", 'r')
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

def get_mod_depth():
	pass

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
	all_e1 = np.asarray(all_e1)
	all_e2 = np.asarray(all_e2)
	all_ind = np.asarray(all_ind)

	e1_early = all_e1[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	e1_late = all_e1[:,45*60*10:50*60*10].mean(axis = 1)/(5.0*60)

	e2_early = all_e2[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	e2_late = all_e2[:,50*60*10:55*60*10].mean(axis = 1)/(5.0*60)

	ind_early = all_ind[:,0:5*60*10].mean(axis = 1)/(5.0*60)
	ind_late = all_ind[:,50*60*10:55*60*10].mean(axis = 1)/(5.0*60)

	f_out = h5py.File(r"C:\Users\Ryan\Documents\data\R7_thru_V13_spike_rates.hdf5",'w-')
	
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

def plot_fr_data():
	f = h5py.File(r"Z:\Data\processed_data\V1_BMI_final\raw_data\R7_thru_V13_spike_rates.hdf5", 'r')

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
	path_in = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5"
	path_out = r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_ensemble_correlations.hdf5"

	f_out = h5py.File(path_out, 'a')

	animal_list = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13"]
	within_e1_by_animal = []
	within_e2_by_animal = []
	between_e1_e2_by_animal = []

	longest = 0
	for animal in animal_list:
		print "Select "+animal
		within_e1, within_e2, between_e1_e2 = ds.ensemble_correlations(path_in)
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
			print "longest = " + str(longest)
	
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
	print "complete!"

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
			print "No t1's."
		try:
			t2 = np.asarray(f[animal]['t2'])
			arrays.append(t2)
		except KeyError:
			print "No t2's."
			t2 = np.zeros(t1.shape)
			arrays.append(t2)
		try:
			miss = np.asarray(f[animal]['miss'])
			arrays.append(miss)
		except KeyError:
			print "No Misses."
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
			print "No t1's."
		try:
			t2 = np.asarray(f[animal]['t2'])
			arrays.append(t2)
		except KeyError:
			print "No t2's."
			t2 = np.zeros(t1.shape)
			arrays.append(t2)
		try:
			miss = np.asarray(f[animal]['miss'])
			arrays.append(miss)
		except KeyError:
			print "No Misses."
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


def plot_light_change_sessions():
	f = h5py.File(r"Z:\Data\processed_data\V1_BMI_final\raw_data\R7_thru_V13_light_data.hdf5",'r')
	data = np.asarray(f['scaled_p_correct_all'][0:2])
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

def save_V1_ds_ff_cohgram_data():	
	f = h5py.File(r"C:\Users\Ryan\Documents\data\t1_triggered.hdf5",'r')
	sessions = f.keys()
	V1_data = []
	DMS_data = []
	session_names = []
	for s in sessions:
	    try:
	        v1 = None
	        dms = None
	        name = None
	        v1 = np.asarray(f[s]['V1_lfp'][:,3000:,:])
	        dms = np.asarray(f[s]['Str_lfp'][:,3000:,:])
	        name = s
	    except KeyError:
	        pass
	    if (v1 != None and dms != None):
	        V1_data.append(v1)
	        DMS_data.append(dms)
	        session_names.append(s)
	f.close()
	##let's put all this on disc since it's gonna be a lot of data...
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_v1_dms_lfp_t1.hdf5",'w-')
	for i, name in enumerate(session_names):
	    gp=g.create_group(name)
	    gp.create_dataset("v1", data=V1_data[i])
	    gp.create_dataset("dms",data=DMS_data[i])
	g.close()
	DMS_data = None; V1_data = None
	g = h5py.File(r"C:\Users\Ryan\Documents\data\paired_v1_dms_lfp_t1.hdf5",'r')
	results_file = h5py.File(r"C:\Users\Ryan\Documents\data\v1_dms_cohgrams_t12.hdf5",'w-')
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
	    async_result = pool.map_async(ss.mp_cohgrams_sf,data)
	    pool.close()
	    pool.join()
	    cohgrams = async_result.get()
	    results_file.create_dataset(session,data = np.asarray(cohgrams))
	g.close()
	results_file.close()


# def get_cursor_states():
# 	f_in = r"C:\Users\Ryan\Documents\data\R7_thru_V13_all_data.hdf5"
# 	#f_out = h5py.File("C:\Users\Ryan\Documents\data\cursor_states.hdf5", 'w-')
# 	##get the raw. binned e1-e2 values for every session
# 	cvals = ds.get_cursor_vals(f_in, binsize=200, session_range = [1,11])
# 	##make the list into a symmetrical array for easier handling
# 	longest = 0
# 	for i in range(len(cvals)):
# 		if cvals[i].shape[0] > longest:
# 			longest = cvals[i].shape[0]
# 	for n in range(len(cvals)):
# 		if cvals[n].shape[0] < longest:
# 			add1 = np.empty((longest-cvals[n].shape[0],))
# 			add1[:] = np.nan
# 			cvals[n] = np.hstack((cvals[n], add1))
# 	cvals = np.asarray(cvals)
# 	##need to normalize each session by its mean value as well as it's range.
# 	for i in range(cvals.shape[0]):
# 		##subtract the mean:
# 		cvals[i,:] = cvals[i,:]-np.nanmean(cvals[i,:])
# 		##get the max and min values
# 		mx = np.nanmax(cvals[i,:])
# 		mn = np.nanmin(cvals[i,:])
# 		##now, we are going to compute the percent distance to the max/min values for each bin
# 		for v in range(cvals[i,:].shape[0]):
# 			if cvals[i,v] > 0:
# 				cvals[i,v] = cvals[i,v]/mx
# 			elif cvals[i,v] < 0:
# 				cvals[i,v] = -1*(cvals[i,v]/mn)
# 	##now, every session in cvals is on the same scale (0 to 1), where 1 is the rewarded tone,
# 	##0 is the unrewarded (well, not exactly but it's close)
# 	##let's use the first 5 min as early, and mins 35-40 as late
# 	early_cvals = (cvals[:,0:10*60*5]).flatten() #binsize is 200, so 5 bins/sec, 60 sec/min
# 	late_cvals = (cvals[:,5*60*30:5*60*40]).flatten()
# 	n, bins, patches = plt.hist(early_cvals, 50, facecolor = 'red', alpha = 0.4)
# 	n, bins, patches = plt.hist(late_cvals[~np.isnan(late_cvals)], 50, facecolor = 'blue', alpha = 0.4)
# 	f_out.create_dataset("early_cvals", data = early_cvals)
# 	f_out.create_dataset("late_cvals", data = late_cvals)
# 	f_out.create_dataset("raw_cvals", data = cvals)
# 	f_out.create_dataset("norm_cvals", data = cvals)

# def get_cursor_states():
# 	f_in = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_all_data.hdf5", 'r')
# 	f_out = h5py.File(r"J:\Ryan\processed_data\V1_BMI_final\raw_data\R7_thru_V13_ensemble_state_data.hdf5", 'w-')
# 	session_range = np.arange(1,11)
# 	animals = ["R13", "R11", "V01", "V02", "V03", "V04", "V05", "V11", "V13"]
# 	mean_cursor_states = []
# 	sup_longest = 0
# 	for animal in animals:
# 		if "/"+animal in f_out:
# 			pass
# 		else:
# 			f_out.create_group(animal)
# 		longest = 0
# 		e1_arrays = []
# 		e2_arrays = []
# 		sessions = [x for x in f_in[animal].keys() if int(x[5:7]) in session_range]
# 		for session in sessions:
# 			e1_keys = [item for item in f_in[animal][session]['e1_units'].keys() if not item.endswith('_wf')]
# 			e2_keys = [item for item in f_in[animal][session]['e2_units'].keys() if not item.endswith('_wf')]
# 			duration = f_in[animal][session]['e1_units'][e1_keys[0]].shape[1]
# 			e1_arr = np.zeros((duration, len(e1_keys)))
# 			e2_arr = np.zeros((duration, len(e2_keys)))
# 			for key in range(len(e1_keys)):
# 				e1_arr[:, key] = np.asarray(f_in[animal][session]['e1_units'][e1_keys[key]])[0,:]
# 			e1_arr = e1_arr.sum(axis = 1)
# 			for name in range(len(e2_keys)):
# 				e2_arr[:, name] = np.asarray(f_in[animal][session]['e2_units'][e2_keys[name]])[0,:]
# 			e2_arr = e2_arr.sum(axis = 1)
# 			e1_arrays.append(e1_arr)
# 			e2_arrays.append(e2_arr)
# 			if duration > longest:
# 				longest = duration
# 		for n in range(len(e1_arrays)):
# 			if e1_arrays[n].shape[0] < longest:
# 				add1 = np.empty((longest-e1_arrays[n].shape[0],))
# 				add2 = np.empty((longest-e1_arrays[n].shape[0],))
# 				add1[:] = np.nan
# 				add2[:] = np.nan
# 				e1_arrays[n] = np.hstack((e1_arrays[n], add1))
# 				e2_arrays[n] = np.hstack((e2_arrays[n], add2))
# 		e1_arrays = np.asarray(e1_arrays)
# 		e2_arrays = np.asarray(e2_arrays)
# 		if "/"+animal+"/e1_ensembles" in f_out:
# 			del(f_out["/"+animal+"/e1_ensembles"])
# 		if "/"+animal+"/e2_ensembles" in f_out:
# 			del(f_out["/"+animal+"/e2_ensembles"])
# 		if "/"+animal+"/mean_cursor_states" in f_out:
# 			del(f_out["/"+animal+"/mean_cursor_states"])
# 		f_out[animal].create_dataset("e1_ensembles", data = e1_arrays)
# 		f_out[animal].create_dataset("e2_ensembles", data = e2_arrays)
# 		f_out[animal].create_dataset("mean_cursor_states", data = (e1_arrays-e2_arrays).mean(axis = 0))
# 		mean_cursor_states.append((e1_arrays-e2_arrays).mean(axis = 0))
# 		if sup_longest < longest:
# 			sup_longest = longest
# 	for n in range(len(mean_cursor_states)):
# 		if mean_cursor_states[n].shape[0] < sup_longest:
# 			add = np.empty((sup_longest-mean_cursor_states[n].shape[0]))
# 			add[:] = np.nan
# 			mean_cursor_states[n] = np.hstack((mean_cursor_states[n], add))
# 	mean_cursor_states = np.asarray(mean_cursor_states)
# 	if "/across_animals_mean_states" in f_out:
# 		del(f_out["/across_animals_mean_states"])
# 	f_out.create_dataset("/across_animals_mean_states", data = mean_cursor_states)
# 	idx = np.arange(0,duration-300, 100)
# 	binned = np.zeros((mean_cursor_states.shape[0], idx.shape[0]))
# 	for i in idx:
# 		binned[:,i/100] = mean_cursor_states[:,i:i+100].sum(axis = 1)
# 	if "/across_animals_binned_states" in f_out:
# 		del(f_out["/across_animals_binned_states"])
# 	f_out.create_dataset("/across_animals_binned_states", data = mean_cursor_states)
# 	f_in.close()
# 	f_out.close()






