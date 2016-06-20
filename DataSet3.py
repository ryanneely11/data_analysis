##this program is for loading, saving, and parsing a user-selected number of
##recording sessions, and as a way to organize data in a way that facilitates
##analysis by pre-processing plexon files.

_author__ = 'Ryan'

import GuiStuff as gs
import os
import numpy as np
import plxread
import h5py
import RatUnits4 as ru
#from progressbar import *
import gc
import SpikeStats2 as ss
import spectrum as spec
import collections
import multiprocessing as mp
import data_to_gdf as dg
from scipy.stats.mstats import zscore
import matplotlib.pyplot as plt

"""This first function takes raw plexon data files, organizes the data,
and saves it in hdf5 format for fast loading and indexing. 
file_path is the output data file path. Integral to the use of this
function is the RatUnits file (currently RatUnits4), which should contain
a dictionary of metadata for every recording session.
the chunk sessions option will ask you what portion of each session you want to take.
"""
def save_session_data(file_path, verbose = True, chunk_sessions = False):
	##open a new HDF5 file to store all the data
	fout = h5py.File(file_path, 'w-')
	## retrieve a list of animals in the RatUnits metadata directory
	##and ask user which to include
	animals = gs.multichoice(ru.animals.keys(), title = 'Select Animals to Analyze')
	##create a dictionary consisting of lists of sessions to load for 
	##each animal, with animal names as the keys
	sessions_dict = {}
	##close file to save memory
	for an in animals:
		##get a list of sessions from the user
		sessions_dict[an] =  gs.multichoice(ru.animals[an][1].keys(), title = "Sessions for "+an)
		##create a file group for this animal
	for animal in sessions_dict:
		if verbose: print "Current animal is "+ animal
		animal_group = fout.create_group(animal)
		##run through each selected session, organizing and saving the data
		for current_session in sessions_dict[animal]:
			if verbose: print "Current file is "+ current_session
			if chunk_sessions:
				chunk_start =int(raw_input("Enter the start time (mins) for "+animal+" "+current_session))*1000*60
				chunk_end =int(raw_input("Enter the end time (mins) for "+animal+" "+current_session))*1000*60
			##load the file into memory after generating the full file path
			session_data = plxread.import_file(os.path.join(ru.animals[animal][0],current_session), 
				AD_channels = range(1,97), save_wf = True, import_unsorted = False, verbose = False)
			##dont abort everything if file doesn't exist
			if session_data == None:
				print "File " + current_session +" does not exist! Skipping..."
				break
			##create a sub-group in the file for this session
			session_group = animal_group.create_group(current_session)
			##get a list of the events in the file
			event_list = [item for item in session_data.keys() if item.startswith('Event')]		 		
			##check to be sure that the events in metadata are included in the actual file
			to_remove = []
			events_dict = ru.animals[animal][1][current_session]['events']
			for e_item in events_dict:
				if events_dict[e_item][0] not in event_list:
					print "*****"+events_dict[e_item][0]+\
					" is not in "+animal+" "+current_session+"*****"
					to_remove.append(e_item)
			if len(to_remove) != 0:
				for item in to_remove:
					del events_dict[item]
			##add random event to events dict
			events_dict['rand'] = ['EventRand']
			##get a list of the units in the file
			units_list = [item for item in session_data.keys() if item.endswith(('a','b','c','d','e')) 
				and item.startswith('sig')]
			##get a list of unit groups for this file
			unit_groups = ru.animals[animal][1][current_session]['units'].keys()
			##use the last timestamp of an A/D signal to set the recording duration.
			##still have not figured out a better way to do this but I see no reason why
			##it shouldn't work
			##scan through the data dict and find the first A/D ts array
			duration = None
			for arr in session_data.keys():
				if arr.startswith('AD') and arr.endswith('_ts'):
					duration = int(np.ceil((session_data[arr].max()*1000)/100)*100)+1
					break
			if duration != None:
				if verbose: print "Duration of "+current_session+" is "+str(duration)
			else: print "No A/D timestamp data found!!!"
			##generate some random events based on #of t1 hits and the recording duration
			##these will still be in seconds to match the event TS data
			rand_event =  np.random.random_integers(0,duration/1000, session_data[events_dict['t1'][0]].size)
			##add random events to the data file
			session_data['EventRand'] = rand_event
			##start creating groups and datasets for all of the data
			event_group = session_group.create_group("event_arrays")
			##add all of the event arrays as timestamp arrays after converting timestamps to ms
			for event_type in events_dict:
				event_ts_data = session_data[events_dict[event_type][0]]*1000.0
				if chunk_sessions:
					idx =np.where(np.logical_and(event_ts_data >= chunk_start, event_ts_data <= chunk_end))[0]
					event_ts_data = event_ts_data[idx] - chunk_start
				event_group.create_dataset(event_type, data = event_ts_data)
				if verbose: print "Successfully added event timestamps for event " + event_type
			##based on metadata information, split spiketrains and lfp signals into the appropriate groups/datasets
			for unit_group in unit_groups:
				if verbose: print "Working through units from " + unit_group
				##get a list of the members of this group for this file
				members = ru.animals[animal][1][current_session]['units'][unit_group]
				##check to make sure each member unit is actually sorted in the file
				to_remove = []
				for unit in members:
					if unit not in units_list:
						print unit +" is not in this file! Deleting... \n"
						to_remove.append(unit)
				if len(to_remove) != 0:
					for item in to_remove:
						members.remove(item)
				##create a unit group group for this unit group
				unit_group_group = session_group.create_group(unit_group)
				##run through each unit member, and add binary format spike trains and 
				##LFP signals from the same channel
				for current_unit in members:
					wfs = None
					##check to see if this unit has waveforms associated with it
					if current_unit+"_wf" in session_data.keys():
						##see if the wf meets the SNR requirements
						wfs = session_data[current_unit+"_wf"]
						SNR = ss.calc_unit_snr(wfs)
					if wfs == None or SNR >2.0:
						##get the timestamp array and convert to ms
						spiketrain = session_data[current_unit] * 1000
						##check to see that this unit meets the spike rate requirements
						total_seconds = (duration/1000.0)
						spike_rate = spiketrain.size/total_seconds
						if spike_rate > 0.5 and spike_rate < 50:
							##convert the spiketrain to a binary array
							spiketrain = np.histogram(spiketrain, bins = duration, range =(0,duration))
		            		##ensure that spiketrain is truly binary
							spiketrain = spiketrain[0].astype(bool).astype(int)
							if chunk_sessions:
								spiketrain = spiketrain[chunk_start:chunk_end]
							##get the raw lfp data and t/s from the current channel
							chan = current_unit[3:6]
							raw_ad = None
							for arr in session_data.keys():
								if arr.startswith('AD') and arr.endswith(chan):
									raw_ad = session_data[arr]
									break
							if raw_ad is not None:
								if verbose: print "Matching A/D signal found for channel "+chan+" (" + unit_group + ")"
							else: print "No matching A/D signal found for channel "+chan+" (" + unit_group + ")"+"!\n"
							##get the LFP timestamp signal from the same channel
							ad_ts = None
							for arr in session_data.keys():
								if arr.startswith('AD') and arr.endswith('_ts') and arr[2:5] == chan:
									ad_ts = session_data[arr]
									break
							if ad_ts is not None:
								if verbose: print "Matching A/D timestamps found for channel "+chan+" (" + unit_group+ ")"
							else: print "No matching A/D timestamps found for channel "+chan+" (" + unit_group +")"+"!\n"
							#convert the ad ts to samples, and integers for indexing
							ad_ts = np.ceil((ad_ts*1000)).astype(int)
							##account for any gaps caused by pausing the plexon session ****IMPORTANT STEP****	
							## The LFP signal may have fewer points than "duration" if 
							##the session was paused, so we need to account for this
							full_ad = np.zeros(duration)
							full_ad[ad_ts] = raw_ad
							##add stacked spiketrain and LFP data to the file
							if chunk_sessions:
								full_ad = full_ad[chunk_start:chunk_end]
							try:
								unit_group_group.create_dataset(current_unit, data = np.vstack((spiketrain, full_ad)))
								if verbose: print "Successfully added data for " + current_unit
							except:
								print "Did not add data for " + current_unit
							##get the matching waveform signal
							wf_sigs = None
							for arr in session_data.keys():
								if arr[:7] == current_unit and arr.endswith('_wf'):
									wf_sigs = session_data[arr]
									break
							if wf_sigs is not None:
								if verbose: print "Matching waveforms found for channel "+chan
								##append wfs to the file
								try:
									unit_group_group.create_dataset(current_unit+"_wf", data = wf_sigs)
									if verbose: print "Successfully added wf data for unit " + current_unit
								except: print "Did not add wfs for " + current_unit
							else: print "No matching waveforms found for channel "+chan+"!"		
						else:
							print "Channel "+chan+" (" + unit_group+ ")" + "does not meet spike rate requirements. Removing..."
					else: 
						print "Channel "+chan+" (" + unit_group+ ")" + "does not meet SNR requirements. Removing..."
	fout.close()
	gc.collect()
	print "Data save complete!!"

"""
This function converts data from one session to a gdf file for use with the Kirkland et all
Matlab program for gravity analysis.
Inputs are:
f_in: processed data file to take data from
f_out: file path to save gdf to 
event_ids: list of strings corresponding to event ids to include
group_ids: list of strings corresponding to group ids to include
"""
def session_to_gdf(f_in, f_out, event_ids, group_ids):
	##open the input file
	f = h5py.File(f_in, 'r')
	##ask user what animal to take data from
	animal = gs.onechoice(f.keys(), title = 'Select Animal to Analyze')
	##look up the sessions for tha selected animal that are stored in the file,
	##and ask the user which one to take data from
	session = gs.onechoice(f[animal].keys(), title = "Select the session to take data from")
	##get the event arrays
	event_arrs = {}
	for trigger in event_ids:
		try:
			event_array = np.asarray(f[animal][session]["event_arrays"][trigger]).squeeze()
			event_arrs[trigger] = event_array			
		except KeyError:
			print "The event trigger you specified is not in the file."
	##get the spike arrays
	unit_arrs = {}
	for group in group_ids:
		try:
			unit_group = f[animal][session][group]
		except KeyError:
			print "The units group you specified is not in the file."
		##the unit group also includes wf data, so ignore that
		unit_list = [unit for unit in unit_group.keys() if not unit.endswith("_wf")]
		##add the data to the dictionary
		for unit in unit_list:
			unit_arrs[animal+"_"+session+"_"+unit] = np.asarray(unit_group[unit])[0]
	##master list of arrays
	all_arrs = []
	##format the arrays and add to master list
	for key in event_arrs:
		all_arrs.append(dg.event_to_gdf(event_arrs[key], key))
	for key in unit_arrs:
		all_arrs.append(dg.spike_to_gdf(unit_arrs[key], key))
	##get the full concatenated, sorted data
	sorted_data = dg.sort_by_timestamp(np.hstack(all_arrs))
	##export to a gdf file
	dg.save_as_gdf(f_out, sorted_data)
	print "GDF file saved!"


"""
This funciton looks through an HDF5 repository of saved session data, and pulls out event arrays
of a defined type for a defined range of sessions. 
Inputs are:
-f_in: full path of file to load (created by save_session_data)
-event_type: string format, event type to load arrays for (ie "t1", "rand")
-session_range: a range of sessions to load data for; in a tuple format, ie (4,9) is 
	sessions for files named "BMI_D04.plx" through "BMI_D08.plx". "None" loads all files.
-Binary: if true, converts event timestamp arrays to binary format
"""
def load_event_arrays(f_in, event_type, session_range = None, binary = True):
	##open the file
	myFile = h5py.File(f_in, 'r')
	##create an empty list to store event arrays
	all_arrays = []
	## retrieve a list of animal names saved in the data, and ask user what to include
	animal_list = gs.multichoice(myFile.keys(), title = 'Select Animals to Analyze')
	##for each animal group, get the event arrays for the appropriate sessions
	for animal in animal_list:
		##get a handle to the current animal group
		animal_group = myFile[animal]
		##get a list of sessions saved in the file for this animal that are in the specified range
		sessions_list = []
		##if no range is specified, include all sessions
		if session_range is None:
			sessions_list = animal_group.keys()
		##otherwise, only add sessions that are in the correct range
		else:
			for session in animal_group.keys():
				if int(session[5:7]) in range(session_range[0], session_range[1]):
					sessions_list.append(session)
		##now that you have the list of sessions, grab the appropriate event array data for that 
		##session
		for current_session in sessions_list:
			print "current session is " + current_session
			all_arrays.append(np.asarray(animal_group[current_session]["event_arrays"][event_type]))
	##if binary arrays are desired, create them
	if binary:
		##figure out what the longest array is so you can create an empty array
		max_len = 0
		for a in all_arrays:
			if a.max() > max_len:
				max_len = a.max()
		##convert arrays to binary
		for i in range(len(all_arrays)):
			##convert the ts array to a binary array
			all_arrays[i] = np.histogram(all_arrays[i], bins = max_len, range =(0,max_len))[0].astype(bool).astype(int)
		##make it all into a nice numpy array
		all_arrays = np.asarray(all_arrays)
	return all_arrays

"""
This funciton looks through an HDF5 repository of saved session data, and pulls out event arrays
of a defined type for a defined range of sessions. 
Inputs are:
-f_in: full path of file to load (created by save_session_data)
-event_type: string format, event type to load arrays for (ie "t1", "rand")
-session_range: a range of sessions to load data for; in a tuple format, ie (4,9) is 
	sessions for files named "BMI_D04.plx" through "BMI_D08.plx". "None" loads all files.
-Binary: if true, converts event timestamp arrays to binary format
"""
def load_event_arrays2(f_in, event_type, animal = None, session_list = None, binary = True):
	##open the file
	myFile = h5py.File(f_in, 'r')
	##create an empty list to store event arrays
	all_arrays = []
	if animal is None:
	## retrieve a list of animal names saved in the data, and ask user what to include
		animal = gs.onechoice(myFile.keys(), title = 'Select Animal to Analyze')
	##for each animal group, get the event arrays for the appropriate sessions
	##get a handle to the current animal group
	animal_group = myFile[animal]
	##if no range is specified, include all sessions
	if session_list is None:
		session_list = animal_group.keys()
	##now that you have the list of sessions, grab the appropriate event array data for that 
	##session
	for current_session in session_list:
		print "current session is " + current_session
		try:
			all_arrays.append(np.asarray(animal_group[current_session]["event_arrays"][event_type]))
		except KeyError:
			print animal+" "+current_session+" has no "+event_type
			all_arrays.append(np.array([np.random.randint(1000*60*30)]))
	##if binary arrays are desired, create them
	if binary:
		##figure out what the longest array is so you can create an empty array
		max_len = 0
		for a in all_arrays:
			if a.max() > max_len:
				max_len = a.max()
		##convert arrays to binary
		for i in range(len(all_arrays)):
			##convert the ts array to a binary array
			all_arrays[i] = np.histogram(all_arrays[i], bins = max_len, range =(0,max_len))[0].astype(bool).astype(int)
		##make it all into a nice numpy array
		all_arrays = np.asarray(all_arrays)
	return all_arrays


"""
This function grabs the full arrays (not target-locked segments) of "signal" data from
a user-specified set of sessions (again taken from a saved session data file). 
-f_in: file path to an hdf5 file saved by the save_session_data function
-f_out: file path to save the data in
-signal: in the format [brain_region, signal type] (ie ["V1_units", "lfp"])
""" 
def save_full_session_data(f_in, f_out, signal):
##open the file
	f = h5py.File(f_in, 'r')
	##initialize the output file and its datasets
	g = h5py.File(f_out, 'w-')
	## retrieve a list of animals from the file
	##and ask user which to include
	animals = gs.multichoice(f.keys(), title = 'Select Animals to Analyze')
	##create a dictionary consisting of lists of sessions to load for 
	##each animal, with animal names as the keys
	sessions_dict = collections.OrderedDict()
	for an in animals:
		##get a list of sessions from the user
		chosen_sessions =  gs.multichoice(f[an].keys(), title = "Sessions for "+an)
		sessions_dict[an] = chosen_sessions

	g.close()

	for animal in sessions_dict:
		print "Current animal is "+ animal
		for session in sessions_dict[animal]:
			g = h5py.File(f_out, 'r+')
			print "Current Session is " + session
			##get the datasets for this session for each unit
			unit_list = [unit for unit in f[animal][session][signal[0]].keys() if not unit.endswith("_wf")]
			if len(unit_list) > 0:
				##get the relevant data based on input
				if signal[1] == "spikes":
					sig_index = 0
				elif signal[1] == "lfp":
					sig_index = 1
				##how long is the length of the session in samples?
				sig_length = f[animal][session][signal[0]][unit_list[0]][sig_index].size
				for unit in unit_list:
					g.create_dataset(animal+"_"+session+"_"+unit, data =  np.asarray(f[animal][session][signal[0]][unit][sig_index]))

			g.close()
			gc.collect()
	print "Complete!!"


"""
This function is designed to use saved data from the above function 
(save_full_session_data). The output is an hdf5 file of normalized spectragrams
of each full session lfp or spike trace. 
Args:
f_in: hdf5 file where data is stored (from above function's output)
f_out: file path to save output data
sigType: lfp or spikes?
norm: normalize the spectragrams or not?
other params are same as ss.lfpSpecGram

"""
def save_full_specgrams(f_in, f_out, sigType, norm = True, window = [0.5, 0.05], 
	Fs = 1000.0, fpass = [0,100], err = None):
	f = h5py.File(f_in, 'r')
	array_list = f.keys()
	f.close()
	##create a list of arguments for the mp pool
	arglist = []
	for array in array_list:
		arglist.append([f_in, array, window, Fs, fpass, err, sigType, norm])
	##create a worker pool to process all of the specgrams
	pool = mp.Pool(processes = mp.cpu_count())
	async_result = pool.map_async(get_specgrams, arglist)
	pool.close()
	pool.join()
	print "All processes have returned; parsing results"
	results_list = async_result.get()
	timebase = window[1]*1000.0
	##save the data
	g = h5py.File(f_out, 'w-')
	for result in results_list:
		g.create_dataset(result[0], data = result[1])
	g.create_dataset("timebase_ms", data = timebase)
	g.create_dataset("frequency", data = fpass)
	g.close()
	print "Complete!"

##a sub-function used above for multiprocessing
def get_specgrams(args):
	#parse the argument list
	f_in = args[0]
	array = args[1]
	window = args[2]
	Fs = args[3]
	fpass = args[4]
	err = args[5]
	sigType = args[6]
	norm = args[7]
	##open the file
	f = h5py.File(f_in, 'r')
	##run specgram calculation
	S, t, fr, Serr = ss.lfpSpecGram(np.asarray(f[array]), window, Fs = Fs, fpass = fpass, err = err, sigType = sigType)
	if norm:
		for i in range(S.shape[1]):
				S[:,i] = S[:,i]/S[:,i].mean()
	return (array,S)

"""
This function is designed to use saved data from the above function 
(save_full_session_data). The output is an hdf5 file of (zscored s-f cohgrams
of each full session lfp and/or spike traces. 
Args:
f_in1: hdf5 file where data is stored (from above function's output)- spike data
f_in2: hdf5 file where data is stored (from above function's output)- lfp data
sigtypes: list in the format ['spike','lfp'] referring to the input file dtypes
f_out: file path to save output data
norm: zscore the cohgrams or not?
other params are same as ss.lfpSpecGram

"""
def save_full_cohgrams(f_in1, f_in2, f_out, norm = True, window = [0.5, 0.05], 
	Fs = 1000.0, fpass = [0,100], err = None):
	f1 = h5py.File(f_in1, 'r')
	f2 = h5py.File(f_in2, 'r')
	spike_list = f1.keys()
	lfp_list = f2.keys()
	f1.close()
	f2.close()
	##create a list of arguments for the mp pool
	arglist = []
	for spike in spike_list:
		for lfp in lfp_list:
			arglist.append([f_in1, f_in2, spike, lfp, window, Fs, fpass, err, norm])
	##create a worker pool to process all of the specgrams
	pool = mp.Pool(processes = mp.cpu_count())
	async_result = pool.map_async(get_cohgrams, arglist)
	pool.close()
	pool.join()
	print "All processes have returned; parsing results"
	results_list = async_result.get()
	timebase = window[1]*1000.0
	##save the data
	g = h5py.File(f_out, 'w-')
	for result in results_list:
		g.create_dataset(result[0], data = result[1])
	g.create_dataset("timebase_ms", data = timebase)
	g.create_dataset("frequency", data = fpass)
	g.close()
	print "Complete!"

##a sub-function used above for multiprocessing
def get_cohgrams(args):
	#parse the argument list
	f_in1 = args[0]
	f_in2 = args[1]
	spikeName = args[2]
	lfpName = args[3]
	window = args[4]
	Fs = args[5]
	fpass = args[6]
	err = args[7]
	norm = args[8]
	##open the file
	f1 = h5py.File(f_in1, 'r')
	f2 = h5py.File(f_in2, 'r')
	##run specgram calculation
	C, phi, S12, S1, S2, t, f, zerosp, confc, phistd, Cerr = ss.spike_field_cohgram(np.asarray(f1[spikeName]),
		np.asarray(f2[lfpName]), window, Fs, fpass, err)
	if norm:
		C = zscore(C, axis = None)
	return (array,C)

"""
This is another in a chain of functions: it's designed to work with saved data from 
the above function (save_full_specgrams). It takes in saved specgrams and then 
takes time-locked windows around a user specified target.
Args: 
f_in: data file path in
master_file: file path pf the master session data file used to make the f_in file
	(need in order to get event arrays)
f_out: data file path out
window: time before and after trigger to take data in the format [ms_pre, ms_post]
"""
def get_time_locked_specgrams(f_in, master_file, f_out, target, window):
	##open/create files
	f = h5py.File(f_in, 'r')
	master = h5py.File(master_file, 'r')
	g = h5py.File(f_out, 'w-')
	full_session_list = [key for key in f.keys() if key != 'frequency' and key != 'timebase_ms']
	##calibrate time window according to the spectra timebase
	pre_win = (window[0]/np.asarray(f['timebase_ms'])).astype(int)
	post_win = (window[1]/np.asarray(f['timebase_ms'])).astype(int)
	## a function to pull out the animal and session names from the strings contained in
	##the f_in file (assumes format "R11_BMI_D04.plx_sig019a")
	def get_animal_and_session(fullStringIn):
		animal_name = fullStringIn[0:3]
		session_name = fullStringIn[4:15]
		return animal_name, session_name
	##do some metadata indexing in order to get the number of segments
	##that you'll end up with (in order to allocate memory)
	num_freqs = np.asarray(f[f.keys()[0]]).shape[1]
	total_traces = 0
	block_start = [0]
	for string in full_session_list:
		an, s = get_animal_and_session(string)
		total_traces += master[an][s]['event_arrays'][target].size	
		block_start.append(total_traces)
	##create a dataset of the appropriate size (dimensions are gonna be freq x time x trials)
	g.create_dataset("all_arrays", (num_freqs,pre_win+post_win, total_traces), dtype = 'float')
	##now actually add the data to the datafile
	for idx, name in enumerate(full_session_list):
		g = h5py.File(f_out, 'r+')
		##get the animal and session name
		an, s = get_animal_and_session(name)
		##load the corresponding event arrays, and convert according to the window used
		##to create the spectragrans
		print "Calculating time-locked specgrams for " + an + " " + s
		centers = np.asarray(master[an][s]['event_arrays'][target])/np.asarray(f['timebase_ms'])
		print "Saving data to file..."
		g['all_arrays'][:,:,block_start[idx]:block_start[idx+1]] = get_data_window(centers, 
			pre_win, post_win, np.asarray(f[name]).T, verbose = True)
		g.close()
		gc.collect()
	print "Complete!"

"""
This function loads spike and lfp signals for all units in a group
taken from a user-specified data window around user-specified event timestamps.
Args:
-f_in: file path to an hdf5 file saved by the save_session_data function
-trigger: name of the event to use as the trigger (ie "t1")
-event_group: name of the event group to take data from ie "V1_units"
- window in the format "[-5000, 5000]" which would take 5 secs of data before and after
	an event timestamp

Outputs:
-numpy arrays of triggered data, one for spikes and one for lfps. The 
	dimensions of these arrays are # units in group x length of window x number of triggers/events_dict
"""

def load_single_group_triggered_data(f_in, trigger, units, window, animal = None, session = None, chunk = None):
	##open the file
	f = h5py.File(f_in, 'r')
	##ask user what animal to take data from
	if animal is None:
		animal = gs.onechoice(f.keys(), title = 'Select Animal to Analyze')
	##look up the sessions for tha selected animal that are stored in the file,
	##and ask the user which one to take data from
	if session is None:
		session = gs.onechoice(f[animal].keys(), title = "Select the session to take data from")
	##get the event array
	try:
		event_array = np.asarray(f[animal][session]["event_arrays"][trigger]).squeeze()
		if chunk is not None:
			event_array = np.asarray([x for x in event_array if x > chunk[0]*1000*60 and x < chunk[1]*1000*60])				
	except KeyError:
		print "The event trigger you specified is not in the file."
	##get the handle to the group containing unit data for the specified group
	try:
		unit_group = f[animal][session][units]
	except KeyError:
		print "The units group you specified is not in the file."
	##the unit group also includes wf data, so ignore that
	unit_list = [unit for unit in unit_group.keys() if not unit.endswith("_wf")]
	##allocate memory for the resulting data arrays
	spikes = np.zeros((len(unit_list), window[0] + window[1], event_array.size))
	lfps = np.zeros((len(unit_list), window[0] + window[1], event_array.size))
	##fill arrays with trriggered data. Recall that the file you loaded saves LFP and spike
	##arrays as a stack, so here we will separate out each signal into separate arrays.
	for n, unit in enumerate(unit_list):
		print "working on unit " + unit
		traces = get_data_window(event_array, window[0], window[1], np.asarray(unit_group[unit]))
		##if the get_data_window function fails, just fill the array space with a copy of the
		##previous trace. Do the same thing if there were no spikes in the window
		if traces.sum() == 0.0:
			spikes[n,:,:] = spikes[n-1,:,:]
			lfps[n,:,:] = lfps[n-1,:,:]
		##if there are no spikes, 
		else:
			##add the spikes to 'spikes'
			spikes[n,:,:] = traces[0,:,:]
			##add the lfps to 'lfps'
			lfps[n,:,:] = traces[1,:,:]
	##return the data
	return spikes, lfps, unit_list

"""
This is a hefty function. Basically, it will save 4 datasets in 
hdf5 format. 
The first two outputs will be pairwise data from signal1 and signal2, 
locked to trigger1. The second two outputs will be pairwise data from 
signal1 and signal2 locked to trigger2. If "equate spikes" is True,
the thin_spikes algorithm will be used to thin spikes between the two trigger
conditions (if the data is spike data).
Inputs:
	-f_in: hdf5 file to take the data from
	-f_out: path to save the hdf5 file containing the data
	-trigger1: id of trigger1, ie "t1"
	-trigger2: id of trigger2
	-signal1: list containing id of signal1 and the sig type, ie ["e1_units","spikes"]
	-signal2: same as above for signal2
	-window: data window to take, ie [4000,4000]
	-equate_spikes: whether or not to thin spikes between conditions
		corresponding to individial unit data
	-sigma: value used in thinning spikes
""" 
def save_pairwise_triggered_data(f_in, f_out, trigger1, trigger2, signal1, signal2, 
	window, equate_spikes = False, sigma = 10):
	##open the file
	f = h5py.File(f_in, 'r')
	##initialize the output file and its datasets
	g = h5py.File(f_out, 'w-')
	## retrieve a list of animals from the file
	##and ask user which to include
	animals = gs.multichoice(f.keys(), title = 'Select Animals to Analyze')
	##create a dictionary consisting of lists of sessions to load for 
	##each animal, with animal names as the keys
	sessions_dict = collections.OrderedDict()
	##make note of the size of each block you will be recieving for each session
	total_t1_traces = 0
	total_t2_traces = 0
	t1_block_start = [0]
	t2_block_start = [0]
	for an in animals:
		##get a list of sessions from the user
		chosen_sessions =  gs.multichoice(f[an].keys(), title = "Sessions for "+an)
		sessions_dict[an] = chosen_sessions
		for s in chosen_sessions:
			num_u1 = len([unit for unit in f[an][s][signal1[0]].keys() if not unit.endswith("_wf")])
			num_u2 = len([sig for sig in f[an][s][signal2[0]].keys() if not sig.endswith("_wf")])
			num_t1 = f[an][s]['event_arrays'][trigger1].shape[0]
			num_t2 = f[an][s]['event_arrays'][trigger2].shape[0]
			total_t1_traces += (num_u1*num_u2)*num_t1
			total_t2_traces += (num_u1*num_u2)*num_t2
			t1_block_start.append(total_t1_traces)
			t2_block_start.append(total_t2_traces)

	g.create_dataset(signal1[0]+"_"+trigger1+"_"+signal1[1], (window[0]+window[1], total_t1_traces), dtype = 'float64')
	g.create_dataset(signal2[0]+"_"+trigger1+"_"+signal2[1], (window[0]+window[1], total_t1_traces), dtype = 'float64')
	g.create_dataset(signal1[0]+"_"+trigger2+"_"+signal1[1], (window[0]+window[1], total_t2_traces), dtype = 'float64')
	g.create_dataset(signal2[0]+"_"+trigger2+"_"+signal2[1], (window[0]+window[1], total_t2_traces), dtype = 'float64')
	##close the file
	f.close()
	g.close()
	s = 0
	for animal in sessions_dict:
		print "Current animal is "+ animal
		for session in sessions_dict[animal]:
			g = h5py.File(f_out, 'r+')
			print "Current Session is " + session
			##get the datasets for this session
			t1_spikes1, t1_lfps1, names = load_single_group_triggered_data(f_in, trigger1, 
				signal1[0], window, animal = animal, session = session)
			print "shape of t1_spikes1 is " + str(t1_spikes1.shape[0]) + ", " +str(t1_spikes1.shape[1]) + ", " + str(t1_spikes1.shape[2])

			t1_spikes2, t1_lfps2, names = load_single_group_triggered_data(f_in, trigger1, 
				signal2[0], window, animal = animal, session = session)

			t2_spikes1, t2_lfps1, names = load_single_group_triggered_data(f_in, trigger2, 
				signal1[0], window, animal = animal, session = session)

			t2_spikes2, t2_lfps2, names = load_single_group_triggered_data(f_in, trigger2, 
				signal2[0], window, animal = animal, session = session)

			names = 0
			
			##get the relevant data based on input, and clear memory 
			#associated with unneeded data
			if signal1[1] == "spikes":
				t1_data1 = t1_spikes1
				t2_data1 = t2_spikes1
				t1_lfps1 = 0
				t2_lfps1 = 0
			elif signal1[1] == "lfp":
				t1_data1 = t1_lfps1
				t2_data1 = t2_lfps1
				t1_spikes1 = 0
				t2_spikes1 = 0
			if signal2[1] == "spikes":
				t1_data2 = t1_spikes2
				t2_data2 = t2_spikes2
				t1_lfps2 = 0
				t2_lfps2 = 0
			elif signal2[1] == "lfp":
				t1_data2 = t1_lfps2
				t2_data2 = t2_lfps2
				t1_spikes2 = 0
				t2_spikes2 = 0
			print "shape of t1_data1 is " + str(t1_data1.shape[0]) + ", " +str(t1_data1.shape[1]) + ", " + str(t1_data1.shape[2])
			
			#allocate memory for paried data
			p_t1_data1 = np.zeros((window[0]+window[1], t1_block_start[s+1]-t1_block_start[s]))
			print "shape of p_t1_data1 container is " + str(p_t1_data1.shape[0]) + ", " + str(p_t1_data1.shape[1])
			p_t1_data2 = np.zeros((window[0]+window[1], t1_block_start[s+1]-t1_block_start[s]))
			print "shape of p_t1_data2 container is " + str(p_t1_data2.shape[0]) + ", " + str(p_t1_data2.shape[1])
			p_t2_data1 = np.zeros((window[0]+window[1], t2_block_start[s+1]-t2_block_start[s]))
			print "shape of p_t2_data1 container is " + str(p_t2_data1.shape[0]) + ", " + str(p_t2_data1.shape[1])
			p_t2_data2 = np.zeros((window[0]+window[1], t2_block_start[s+1]-t2_block_start[s]))
			print "shape of p_t2_data2 container is " + str(p_t2_data2.shape[0]) + ", " + str(p_t2_data2.shape[1])

			##put the paired data into the containers
			##the number of events in this session
			e = t1_data1.shape[2]
			for i in range(t1_data1.shape[0]):
				for j in range(t1_data2.shape[0]):
					##keep track of how many datasets have been added
					n = i*t1_data2.shape[0]+j
					#add the data to the array
					p_t1_data1[:,n*e:(n+1)*e] = t1_data1[i,:,:]
					p_t1_data2[:,n*e:(n+1)*e] = t1_data2[j,:,:]

			#repeat for target2
			e = t2_data1.shape[2]
			for i in range(t2_data1.shape[0]):
				for j in range(t2_data2.shape[0]):
					##keep track of how many datasets have been added
					n = i*t2_data2.shape[0]+j
					#add the data to the array
					p_t2_data1[:,n*e:(n+1)*e] = t2_data1[i,:,:]
					p_t2_data2[:,n*e:(n+1)*e] = t2_data2[j,:,:]

			#if requested, run spike thinning on any data pairs that are spike data
			if equate_spikes:
				if signal1[1] == "spikes":
					print "thinning spike set 1"
					ss.thin_spikes(p_t1_data1, p_t2_data1, sigma)
				if signal2[1] == "spikes":
					print "thinning spike set 2"
					ss.thin_spikes(p_t1_data2, p_t2_data2, sigma)

			##add data to the file
			g[signal1[0]+"_"+trigger1+"_"+signal1[1]][:,t1_block_start[s]:t1_block_start[s+1]] = p_t1_data1
			g[signal2[0]+"_"+trigger1+"_"+signal2[1]][:,t1_block_start[s]:t1_block_start[s+1]] = p_t1_data2
			g[signal1[0]+"_"+trigger2+"_"+signal1[1]][:,t2_block_start[s]:t2_block_start[s+1]] = p_t2_data1
			g[signal2[0]+"_"+trigger2+"_"+signal2[1]][:,t2_block_start[s]:t2_block_start[s+1]] = p_t2_data2

			g.close()
			gc.collect()
			s+=1
	print "Complete!!"

def save_multi_group_triggered_data(f_in, f_out, trigger, signal, window, chunk = None):
	##open the file
	f = h5py.File(f_in, 'r')
	##initialize the output file and its datasets
	g = h5py.File(f_out, 'w-')
	## retrieve a list of animals from the file
	##and ask user which to include
	animals = gs.multichoice(f.keys(), title = 'Select Animals to Analyze')
	##create a dictionary consisting of lists of sessions to load for 
	##each animal, with animal names as the keys
	sessions_dict = collections.OrderedDict()
	##make note of the size of each block you will be recieving for each session
	total_traces = 0
	block_start = [0]
	for an in animals:
		##get a list of sessions from the user
		chosen_sessions =  gs.multichoice(f[an].keys(), title = "Sessions for "+an)
		sessions_dict[an] = chosen_sessions
		for s in chosen_sessions:
			num_u1 = len([unit for unit in f[an][s][signal[0]].keys() if not unit.endswith("_wf")])
			if chunk is None:
				num_t1 = f[an][s]['event_arrays'][trigger].shape[0]
			else:
				num_t1 = len([x for x in np.asarray(f[an][s]['event_arrays'][trigger]) if x > chunk[0]*1000*60 and x < chunk[1]*1000*60])
			total_traces += num_u1*num_t1
			block_start.append(total_traces)

	g.create_dataset(signal[0]+"_"+trigger+"_"+signal[1], (window[0]+window[1], total_traces), dtype = 'float64')

	##close the file
	f.close()
	g.close()
	s = 0
	for animal in sessions_dict:
		print "Current animal is "+ animal
		for session in sessions_dict[animal]:
			g = h5py.File(f_out, 'r+')
			print "Current Session is " + session
			##get the datasets for this session
			spikes, lfps, names = load_single_group_triggered_data(f_in, trigger, 
				signal[0], window, animal = animal, session = session, chunk = chunk)

			names = 0
			
			##get the relevant data based on input, and clear memory 
			#associated with unneeded load_
			if signal[1] == "spikes":
				if len(spikes.shape) < 3 or spikes.shape[0] == 1:
					print "Warning: only one spike channel detected"
					data1 = np.squeeze(spikes)
				elif spikes.shape[2] == 1:
					print "Only one event detected."
					data1 = np.squeeze(spikes).T
				else:
					data1 = np.hstack(np.squeeze(np.split(spikes, spikes.shape[0], 0)))

			elif signal[1] == "lfp":
				if len(lfps.shape) < 3 or lfps.shape[0] == 1:
					print "Warning: only one spike channel detected"
					data1 = np.squeeze(lfps)
				else:
					data1 = np.hstack(np.squeeze(np.split(lfps, lfps.shape[0], 0)))

			##add data to the file
			g[signal[0]+"_"+trigger+"_"+signal[1]][:,block_start[s]:block_start[s+1]] = data1

			g.close()
			gc.collect()
			s+=1
	print "Complete!!"

	

"""
This function returns the waveforms corresponding to a particular unit group in 
a given session for a given animal. 
Inputs:
	- f_in: file path of the HDF5 file to open
	-group: name of the unit group to load
Outputs:
	-a list of wf data arrays
"""
def get_wfs(f_in, group, animal = None, session = None):
	##open the file
	f = h5py.File(f_in, 'r')
	##ask user what animal to take data from
	if animal == None:
		animal = gs.onechoice(f.keys(), title = 'Select Animal to Analyze')
	##look up the sessions for tha selected animal that are stored in the file,
	##and ask the user which one to take data from
	if session == None:
		session = gs.onechoice(f[animal].keys(), title = "Select the session to take data from")
	##get the handle to the group containing unit data for the specified group
	try:
		unit_group = f[animal][session][group]
	except KeyError:
		print "The units group you specified is not in the file."
	##the unit group also includes unit data, so ignore that
	wf_list = [wf for wf in unit_group.keys() if wf.endswith("_wf")]
	##allocate memory for the resulting data array
	wfs_arr = []
	##fill the array with the data
	for n, item in enumerate(wf_list):
		wfs_arr.append(np.asarray(unit_group[item]))
	##return the data
	return wfs_arr


"""
A somewhat convoluted function to get the contingency degredation
values given a dataset of sessions that include the CD days. It's a little
tricky to write this function, as sometimes the CD and reinstatement sessions happen
on the same days, and sometimes on different days. Therefore the function
asks about every data point. 
Input:
    -the path to the dataset to use for analysis.
Output: 3 lists, where the length of the list is the number of CD sessions, 
        and the separate lists represent:
        0: the RATE (per min) of hits for each CD session pre-degredation, 
        1: the RATE (per min) of hits for each session during the degredation, and 
        2: the RATE of hits for each session post-degredation. 
    Note that these above values are paired, 
    because they only have meaning relative to each other.
"""
def CD_data(f_in, target):
    ##create lists to store the counts. Won't be a ton of data so don't need
    ##to worry about allocating space
	pre_counts = []
	peri_counts = []
	post_counts = []
	##open the file
	f = h5py.File(f_in, 'r')
	## retrieve a list of animals in the RatUnits metadata directory
	##and ask user which to include
	animals = gs.multichoice(f.keys(), title = 'Select Animals to Analyze')
	###go through each animal individually
	for animal in animals:
		print "You are now working on animal " + animal
		##make a dicitonary to store the event data
		event_dict = {}
		##get a list of sessions containing CD data
		session_list = gs.multichoice(f[animal].keys(), title = "Select sessions for "+animal)
		##populate the data dictionary
		for session in session_list:
			try:
				raw = np.asarray(f[animal][session]["event_arrays"][target])
				##convert to binary
				event_dict[session] = np.histogram(raw, bins = raw.max(), range =(0,raw.max()))[0].astype(bool).astype(int)
	 		##some sessions might not have a full set of targets. If that's the case, just fill in with a zero-array
			except KeyError:
				print "No " + target + " data for this file; creating a zero-array in place"
				event_dict[session] = np.zeros(90*60*1000)

		##ask user how many CD sessions are included in this data
		num_CDs =int(raw_input("How many CD sessions are in this data?  "))
		##collect the data for each CD session
		for i in range(num_CDs):
			data_loc = gs.onechoice(session_list, title = "Pick the session for pre-CD, session " + str(i))
			##let user know the total length of the session for the next step
			print "This is session " + data_loc
			print "Total length of this session is " + str(event_dict[data_loc].shape[0]/60.0/1000.0) + " mins"
			##ask user what chunk of the session contains the pre-data
			start = int(raw_input("The start time (in mins) is: "))*60*1000
			stop = int(raw_input("The stop time (in mins) is: "))*60*1000
			pre_counts.append(event_dict[data_loc][start:stop].sum()/float(stop/60/1000-start/60/1000))

			##repeat for the peri-CD part
			data_loc = gs.onechoice(session_list, title = "Pick the session for peri-CD, session " + str(i))
			##let user know the total length of the session for the next step
			print "This is session " + data_loc
			print "Total length of this session is " + str(event_dict[data_loc].shape[0]/60.0/1000.0) + " mins"
			##ask user what chunk of the session contains the pre-data
			start = int(raw_input("The start time (in mins) is: "))*60*1000
			stop = int(raw_input("The stop time (in mins) is: "))*60*1000
			peri_counts.append(event_dict[data_loc][start:stop].sum()/float(stop/60/1000-start/60/1000))

			##repeat for post
			data_loc = gs.onechoice(session_list, title = "Pick the session for post-CD, session " + str(i))
			##let user know the total length of the session for the next step
			print "This is session " + data_loc
			print "Total length of this session is " + str(event_dict[data_loc].shape[0]/60.0/1000.0) + " mins"
			##ask user what chunk of the session contains the pre-data
			start = int(raw_input("The start time (in mins) is: "))*60*1000
			stop = int(raw_input("The stop time (in mins) is: "))*60*1000
			post_counts.append(event_dict[data_loc][start:stop].sum()/float(stop/60/1000-start/60/1000))
	return pre_counts, peri_counts, post_counts


"""
A function to get two lists: one of all the e1 unit binary arrays (added together for each session), 
and one of all the e2 binary arrays. Input is the full path of the 
hdf5 dataset to grab the data from.
"""
def get_ensemble_arrays(f_in, session_range = None, animal = None, session = None):
	##open the file
	myFile = h5py.File(f_in, 'r')
	##create an empty list to store data arrays
	e1_arrays = []
	e2_arrays = []
	indirect_arrays = []
	if animal is None:
		## retrieve a list of animal names saved in the data, and ask user what to include
		animal_list = gs.multichoice(myFile.keys(), title = 'Select Animals to Analyze')
	else:
		animal_list = [animal]
	##for each animal group, get the event arrays for the appropriate sessions
	for animal in animal_list:
		print "Current Animal is " + animal
		##get a handle to the current animal group
		animal_group = myFile[animal]
		##get a list of sessions saved in the file for this animal that are in the specified range
		##if no range is specified, include all sessions
		if session_range is None and session is None:
			sessions_list = animal_group.keys()
		##otherwise, only add sessions that are in the correct range
		elif session is None and session_range is not None:
			sessions_list = []
			for session in animal_group.keys():
				if int(session[5:7]) in range(session_range[0], session_range[1]):
					sessions_list.append(session)
		elif session_range is None and session is not None:
			sessions_list = [session]
		##now that you have the list of sessions, grab the appropriate array data for that 
		##session
		for current_session in sessions_list:
			print "current session is " + current_session
			session_group = animal_group[current_session]
			##get just the binary spiketrain arrays
			e1_keys = [item for item in session_group['e1_units'].keys() if not item.endswith('_wf')]
			e2_keys = [item for item in session_group['e2_units'].keys() if not item.endswith('_wf')]
			try:
				ind_keys = [item for item in session_group['V1_units'].keys() if not item.endswith('_wf')]
			except KeyError:
				ind_keys = []
			##get the length of the arrays (should all be the same)
			duration = session_group['e1_units'][e1_keys[0]].shape[1]
			##add together the values for all the e1 and e2 arrays
			e1_arr = np.zeros((duration, len(e1_keys)))
			e2_arr = np.zeros((duration, len(e2_keys)))
			ind_arr = np.zeros((duration, len(ind_keys)))
			for key in range(len(e1_keys)):
				e1_arr[:, key] = np.asarray(session_group['e1_units'][e1_keys[key]])[0,:]
			for name in range(len(e2_keys)):
				e2_arr[:, name] = np.asarray(session_group['e2_units'][e2_keys[name]])[0,:]
			for name in range(len(ind_keys)):
				ind_arr[:, name] = np.asarray(session_group['V1_units'][ind_keys[name]])[0,:]
#			e1_arrays.append(e1_arr)
#			e2_arrays.append(e2_arr)
	return e1_arr, e2_arr, ind_arr

def get_cursor_vals(f_in, session_range = None, animal = None, session = None, binsize = 200):
	##open the file
	myFile = h5py.File(f_in, 'r')
	##create an empty list to store data arrays
	cursor_vals = []
	if animal is None:
		## retrieve a list of animal names saved in the data, and ask user what to include
		animal_list = gs.multichoice(myFile.keys(), title = 'Select Animals to Analyze')
	else:
		animal_list = [animal]
	##for each animal group, get the event arrays for the appropriate sessions
	for animal in animal_list:
		print "Current Animal is " + animal
		##get a handle to the current animal group
		animal_group = myFile[animal]
		##get a list of sessions saved in the file for this animal that are in the specified range
		##if no range is specified, include all sessions
		if session_range is None and session is None:
			sessions_list = animal_group.keys()
		##otherwise, only add sessions that are in the correct range
		elif session is None and session_range is not None:
			sessions_list = []
			for session in animal_group.keys():
				if int(session[5:7]) in range(session_range[0], session_range[1]):
					sessions_list.append(session)
		elif session_range is None and session is not None:
			sessions_list = [session]
		##now that you have the list of sessions, grab the appropriate array data for that 
		##session
		for current_session in sessions_list:
			print "current session is " + current_session
			session_group = animal_group[current_session]
			##get just the binary spiketrain arrays
			e1_keys = [item for item in session_group['e1_units'].keys() if not item.endswith('_wf')]
			e2_keys = [item for item in session_group['e2_units'].keys() if not item.endswith('_wf')]
			##get the length of the arrays (should all be the same)
			duration = session_group['e1_units'][e1_keys[0]].shape[1]
			##add together the values for all the e1 and e2 arrays
			e1_arr = np.zeros((duration, len(e1_keys)))
			e2_arr = np.zeros((duration, len(e2_keys)))
			for key in range(len(e1_keys)):
				e1_arr[:, key] = np.asarray(session_group['e1_units'][e1_keys[key]])[0,:]
			for name in range(len(e2_keys)):
				e2_arr[:, name] = np.asarray(session_group['e2_units'][e2_keys[name]])[0,:]
			e1_arr = e1_arr.sum(axis = 1)
			e2_arr = e2_arr.sum(axis = 1)
			idx = np.arange(0, duration-binsize, binsize)
			cvals = np.zeros(len(idx))
			for n, i in enumerate(idx):
				e1_val = e1_arr[i:i+binsize].sum()
				e2_val = e2_arr[i:i+binsize].sum()
				cvals[n] = e1_val-e2_val
			cursor_vals.append(cvals)
	
	return cursor_vals

"""a useful function for getting a data window centered around a given index.
	Inputs:
	-Centers: the indices for taking windows arround (1-D np array)
	-pre-win: pre-center window length
	-post-win: post-center window length
	-data: full data trace(s) in the shape (y-axis, time axis)
"""
def get_data_window(centers, pre_win, post_win, data, verbose = True):
	centers = np.squeeze(np.asarray(centers)).astype(np.int64)
	data = np.squeeze(data)
	if len(data.shape) > 1:
		N = data.shape[1]
		num_signals = data.shape[0]
	else:
		N = data.shape[0]
		num_signals = 1
		data = data[None,:]
	try:		
		##figure out if any of the centers are going to be too close to the array edges. If so,
		##copy the previous center in its place
		for j, center in enumerate(centers):
			if center <= pre_win or center + post_win >= N:
				centers[j] = centers[j-1]
				if verbose:
					print "Index too close to start or end to take a full window. Deleting event at "+str(center)
		traces = np.zeros((num_signals, pre_win+post_win, len(centers)))
		##the actual windowing functionality:
		for n, idx in enumerate(centers):
				try:
					traces[:,:,n] = data[:,idx-pre_win:idx+post_win]
				except ValueError:
					print "ValueError: Index deletion safeguard did not work"
					traces[:,:,n] = traces[:,:,n-1]
	except TypeError:
		print "No target hits in this session/time block."		
		traces = np.array([0.0])
	return np.squeeze(traces)


"""a function to return pairwise spike-spike coherence traces for a number of selected
sessions, looking at the average coherence between pairs across trials within
a session. 
Inputs:
	-file to load data from
	-ID of unit targets to take windows around
	-IDs of unit groups
	-window size to take around target (can be asymmetric)
	-sampling rate
	-fpass values
Outputs:
	-2 lists of pairs x frequency values array for each session for each target,
	 and an array of frequency values (f)
"""
def pairwise_coherence(f_in, target1, target2, units1, units2, window, 
	Fs = 1000.0, fpass = [0,200], equate_spikes = True, sigma = 10):
	##open the file
	f = h5py.File(f_in, 'r')
	## retrieve a list of animals from the file
	##and ask user which to include
	animals = gs.multichoice(f.keys(), title = 'Select Animals to Analyze')
	##create a dictionary consisting of lists of sessions to load for 
	##each animal, with animal names as the keys
	sessions_dict = {}
	for an in animals:
		##get a list of sessions from the user
		sessions_dict[an] =  gs.multichoice(f[an].keys(), title = "Sessions for "+an)
	##close the file
	f.close()
	##get the length of the coherence data to be calculated
	N = window[0] + window[1]
	nfft = 2**spec.nextpow2(N)
	f, findx = ss.getfgrid(Fs, nfft, fpass)
	##list to store the results 
	results1 = []
	results2 = []
	##run through all of the sessions and get the data
	for animal in sessions_dict:
		print "Current animal is "+ animal
		for session in sessions_dict[animal]:
			print "Current Session is " + session
			##get the datasets for this session
			t1_spikes1, lfps, t1_names1 = load_single_group_triggered_data(f_in, target1, 
				units1, window, animal = animal, session = session)
			t1_spikes2, lfps, t1_names2 = load_single_group_triggered_data(f_in, target1, 
				units2, window, animal = animal, session = session)
			t2_spikes1, lfps, t2_names1 = load_single_group_triggered_data(f_in, target2, 
				units1, window, animal = animal, session = session)
			t2_spikes2, lfps, t2_names2 = load_single_group_triggered_data(f_in, target2, 
				units2, window, animal = animal, session = session)
			if equate_spikes:
				for i in range(t1_spikes1.shape[0]):
					ss.thin_spikes(t1_spikes1[i,:,:], t2_spikes1[i,:,:], sigma)
				for j in range(t1_spikes2.shape[0]):
					ss.thin_spikes(t1_spikes2[j,:,:], t2_spikes2[j,:,:], sigma)
			##clear some memory
			lfps = 0
			##calculate the pairwise coherence values and add to the results list
			results1.append(ss.get_pairwise_coherence(t1_spikes1, t1_spikes2, Fs = Fs, fpass = fpass))
			results2.append(ss.get_pairwise_coherence(t2_spikes1, t2_spikes2, Fs = Fs, fpass = fpass))
	return results1, results2, f


def sig_pairwise_coherence(f_in, target1, target2, units1, units2, window, 
	Fs = 1000.0, fpass = [0,200], equate_spikes = True, sigma = 10, sig = 0.05, 
	band = [2,4]):
	##open the file
	f = h5py.File(f_in, 'r')
	## retrieve a list of animals from the file
	##and ask user which to include
	animals = gs.multichoice(f.keys(), title = 'Select Animals to Analyze')
	##create a dictionary consisting of lists of sessions to load for 
	##each animal, with animal names as the keys
	sessions_dict = {}
	for an in animals:
		##get a list of sessions from the user
		sessions_dict[an] =  gs.multichoice(f[an].keys(), title = "Sessions for "+an)
	##close the file
	f.close()
	##get the length of the coherence data to be calculated
	N = window[0] + window[1]
	nfft = 2**spec.nextpow2(N)
	f, findx = ss.getfgrid(Fs, nfft, fpass)
	##variable to store the results
	results_t1_u1 = []
	results_t1_u2 = []
	results_t2_u1 = []
	results_t2_u2 = []
	##run through all of the sessions and get the data
	for animal in sessions_dict:
		print "Current animal is "+ animal
		for session in sessions_dict[animal]:
			print "Current Session is " + session
			##get the datasets for this session
			t1_spikes1, lfps, t1_names1 = load_single_group_triggered_data(f_in, target1, 
				units1, window, animal = animal, session = session)
			t1_spikes2, lfps, t1_names2 = load_single_group_triggered_data(f_in, target1, 
				units2, window, animal = animal, session = session)
			t2_spikes1, lfps, t2_names1 = load_single_group_triggered_data(f_in, target2, 
				units1, window, animal = animal, session = session)
			t2_spikes2, lfps, t2_names2 = load_single_group_triggered_data(f_in, target2, 
				units2, window, animal = animal, session = session)
			if equate_spikes:
				for i in range(t1_spikes1.shape[0]):
					ss.thin_spikes(t1_spikes1[i,:,:], t2_spikes1[i,:,:], sigma)
				for j in range(t1_spikes2.shape[0]):
					ss.thin_spikes(t1_spikes2[j,:,:], t2_spikes2[j,:,:], sigma)
			##clear some memory
			lfps = 0
			##get a list of any pairs of units with pairwise coherence significantly
			##greater in the first condition compatred to the second
			sig_t1_u1, sig_t1_u2, sig_t2_u1, sig_t2_u2 = ss.compare_pairwise_coherence(t1_spikes1, 
				t1_names1, t1_spikes2, t1_names2, t2_spikes1, t2_names1, 
				t2_spikes2, t2_names2, Fs, fpass, sig, band)
    		##see if there are any results and if so add them to the dict along with 
    		##animal/session info
			results_t1_u1.append(sig_t1_u1)
			results_t1_u2.append(sig_t1_u2)
			results_t2_u1.append(sig_t2_u1)
			results_t2_u2.append(sig_t2_u2)

	return results_t1_u1, results_t1_u2, results_t2_u1, results_t2_u2


def ensemble_correlations(f_in, window = [120000,30000], tau = 40, dt = 1):
	##open the file
	f = h5py.File(f_in, 'r')
	## retrieve a list of animals from the file
	##and ask user which to include
	animals = gs.multichoice(f.keys(), title = 'Select Animals to Analyze')
	##create a dictionary consisting of lists of sessions to load for 
	##each animal, with animal names as the keys
	sessions_dict = {}
	for an in animals:
		##get a list of sessions from the user
		sessions_dict[an] =  gs.multichoice(f[an].keys(), title = "Sessions for "+an)
	##close the file
	f.close()
	##variable to store the results
	within_e1 = []
	within_e2 = []
	between_e1_e2 = []
	##run through all of the sessions and get the data
	for animal in sessions_dict:
		print "Current animal is "+ animal
		for session in sessions_dict[animal]:
			print "Current Session is " + session
			##get the datasets for this session
			e1_arrays, e2_arrays, ind_arrays = get_ensemble_arrays(f_in, session_range = None, 
				animal = animal, session = session)
			# e1_arrays = e1_arrays[0]
			# e2_arrays = e2_arrays[0]
			try:
				within_e1.append(ss.window_corr(e1_arrays[:,0], e1_arrays[:,1], window, tau, dt))
			except IndexError:
				print "only one unit for e1..."
			try:
				within_e2.append(ss.window_corr(e2_arrays[:,0], e2_arrays[:,1], window, tau, dt))
			except IndexError:
				print "only one unit for e2..."
			try:
				between_e1_e2.append(ss.window_corr(e1_arrays[:,0], e2_arrays[:,0], window, tau, dt))
			except IndexError:
				pass
			try:
				between_e1_e2.append(ss.window_corr(e1_arrays[:,0], e2_arrays[:,1], window, tau, dt))
			except IndexError:
				pass
			try:
				between_e1_e2.append(ss.window_corr(e1_arrays[:,1], e2_arrays[:,0], window, tau, dt))
			except IndexError:
				pass
			try:
				between_e1_e2.append(ss.window_corr(e1_arrays[:,1], e2_arrays[:,1], window, tau, dt))
			except IndexError:
				pass

	longest = 0
	for i in range(len(within_e1)):
		if within_e1[i].size > longest:
			longest = within_e1[i].size
	for i in range(len(within_e2)):
		if within_e2[i].size > longest:
			longest = within_e2[i].size

	for i in range(len(within_e1)):	
		if within_e1[i].shape[0] < longest:
			add = np.empty((longest-within_e1[i].shape[0]))
			add[:] = np.nan
			within_e1[i] = np.hstack((within_e1[i], add))
	for i in range(len(within_e2)):	
		if within_e2[i].shape[0] < longest:
			add = np.empty((longest-within_e2[i].shape[0]))
			add[:] = np.nan
			within_e2[i] = np.hstack((within_e2[i], add))
	for i in range(len(between_e1_e2)):	
		if between_e1_e2[i].shape[0] < longest:
			add = np.empty((longest-between_e1_e2[i].shape[0]))
			add[:] = np.nan
			between_e1_e2[i] = np.hstack((between_e1_e2[i], add))
	within_e1 = np.asarray(within_e1)
	within_e2 = np.asarray(within_e2)
	between_e1_e2 = np.asarray(between_e1_e2)

	return within_e1, within_e2, between_e1_e2


def pairwise_triggered_coherence(f_in, f_out, target1, target2, units1, units2, window, 
	equate_spikes = True, sigma = 10, Fs = 1000.0, fpass = [0,100], winstep = [0.5, 0.05]):
	##open the file
	f = h5py.File(f_in, 'r')
	g = h5py.File(f_out, 'w-')
	## retrieve a list of animals from the file
	##and ask user which to include
	animals = gs.multichoice(f.keys(), title = 'Select Animals to Analyze')
	##create a dictionary consisting of lists of sessions to load for 
	##each animal, with animal names as the keys
	sessions_dict = {}
	for an in animals:
		##get a list of sessions from the user
		sessions_dict[an] =  gs.multichoice(f[an].keys(), title = "Sessions for "+an)
	##close the file
	f.close()
	g.close()
	##run through all of the sessions and get the data
	for animal in sessions_dict:
		print "Current animal is "+ animal
		for session in sessions_dict[animal]:
			g = h5py.File(f_out, 'r+')
			print "Current Session is " + session
			##get the datasets for this session
			t1_spikes1, t1_lfps1, t1_names1 = load_single_group_triggered_data(f_in, target1, 
				units1, window, animal = animal, session = session)
			##throw out unnecessary data
			t1_lfps1 = 0
			t1_spikes2, t1_lfps2, t1_names2 = load_single_group_triggered_data(f_in, target1, 
				units2, window, animal = animal, session = session)
			t1_spikes2 = 0
			t2_spikes1, t2_lfps1, t2_names1 = load_single_group_triggered_data(f_in, target2, 
				units1, window, animal = animal, session = session)
			t2_lfps1 = 0
			t2_spikes2, t2_lfps2, t2_names2 = load_single_group_triggered_data(f_in, target2, 
				units2, window, animal = animal, session = session)
			t2_spikes2 = 0
			if equate_spikes:
				for i in range(t1_spikes1.shape[0]):
					ss.thin_spikes(t1_spikes1[i,:,:], t2_spikes1[i,:,:], sigma)
			##clear some memory
			lfps = 0
			##calculate the pairwise coherence values and add to the results list
			t1_cohgrams = ss.get_pairwise_cohgram(t1_spikes1, t1_lfps2, Fs = Fs, fpass = fpass, movingwin = winstep)
			t2_cohgrams = ss.get_pairwise_cohgram(t2_spikes1, t2_lfps2, Fs = Fs, fpass = fpass, movingwin = winstep)
			g.create_dataset(animal+"/"+session+"/t1", data = t1_cohgrams)
			g.create_dataset(animal+"/"+session+"/t2", data = t2_cohgrams)
			g.close()
	print "Complete!"
	return None


"""
a function similar to that in the SpikeStats2 library, 
except it aggregates the STA from multiple animals and sessions.
input agruments are the same; the only difference is that this function
allows you to average over animals or leave them separate. The keywords for the 
averaging are either "animal", "session", or "all."
"""
def get_STAs(f_in, target, units, lfps, window, lfp_win):
	##open the file
	f = h5py.File(f_in, 'r')
	## retrieve a list of animals from the file
	##and ask user which to include
	animal = gs.onechoice(f.keys(), title = 'Select Animal to Analyze')
	##create a dictionary consisting of lists of sessions to load for 
	##each animal, with animal names as the keys
	sessions=  gs.multichoice(f[animal].keys(), title = "Sessions for "+animal)
	##close the file
	f.close()
	##containers for sta for each animal
	STAs = []
	##run through all of the sessions and get the data
	for s, session in enumerate(sessions):
		print "Current Session is " + session
		##get the datasets for this session
		spikes1, lfps1, names1 = load_single_group_triggered_data(f_in, target, 
			units, window, animal = animal, session = session)
		spikes2, lfps2, names2 = load_single_group_triggered_data(f_in, target, 
			lfps, window, animal = animal, session = session)
		##create a container for the STA for all units in this session
		session_STAs = []
		##get the STA for all unit/lfp pairs 
		for i in range(spikes1.shape[0]):
			for j in range(lfps2.shape[0]):
				session_STAs.append(ss.STA(spikes1[i,:,:], lfps2[j,:,:], lfp_win, trialave = False))
		session_stack = session_STAs[0]
		for p in range(1, len(session_STAs)):
			session_stack = np.hstack((session_stack, session_STAs[p]))
		##add the average animal STA to the global container
		STAs.append(session_stack) 
	#concatenate
	full_stack = STAs[0]
	for q in range(1,len(STAs)):
		full_stack = np.hstack((full_stack, STAs[q]))

	return full_stack


##a function to plot the triggered spike rates of all sorted units so you can figure out
##which ones were E1 and E2 units because you forgot to save the metadata, you dummy
def forgot_to_save_metadata(f_in, t1 = 'Event011', verbose = True):
	##load file
	session_data = plxread.import_file(f_in, AD_channels = range(1,97), save_wf = True, 
		import_unsorted = False, verbose = False)
	##use the last timestamp of an A/D signal to set the recording duration.
	##still have not figured out a better way to do this but I see no reason why
	##it shouldn't work
	##scan through the data dict and find the first A/D ts array
	duration = None
	for arr in session_data.keys():
		if arr.startswith('AD') and arr.endswith('_ts'):
			duration = int(np.ceil((session_data[arr].max()*1000)/100)*100)+1
			break
	##the target event to lock to
	T1 = np.asarray(session_data[t1])*1000.0
	##get a list of the units
	units_list = [item for item in session_data.keys() if item.endswith(('a','b','c','d','e')) 
				and item.startswith('sig')]
	for current_unit in units_list:
		plt.figure()
		print current_unit
		##get the timestamp array and convert to ms
		spiketrain = np.asarray(session_data[current_unit]) * 1000.0
		##convert the spiketrain to a binary array
		spiketrain = np.histogram(spiketrain, bins = duration, range =(0,duration))
        ##ensure that spiketrain is truly binary
		spiketrain = spiketrain[0].astype(bool).astype(int)
		##get the spike triggered windows
		stack = get_data_window(T1, 3000, 3000, spiketrain, verbose = True)
		N = stack.shape[0]    
		e1_arr = ss.masked_avg(ss.zscored_fr(stack, 100))
		x = np.linspace(-N/2.0, N/2.0, e1_arr.shape[0])
		plt.plot(x, e1_arr, color = 'g')
		plt.xlabel("Time to target, ms")
		plt.ylabel("Firing Rate (zscore)")
		plt.title("Mean FR for Target Hits: " + current_unit)
	plt.show()
