##data_set_creator.py

##contains functions to parse plexon files

import plxread
import numpy as np
import scipy.io as spio
import h5py
import RatUnits4_resorted as ru
import GuiStuff as gs
import os

"""
saves all data from selected animals and sessions all
aligned to one event type. This function concatenates all data from 
the various sessions/animals and instead organizes around groups of units and channels.
the output file basically contains a bunch of 3-D arrays, where:
	-dim 0 is time around each event
	-dim 1 is trials
	-dim 2 is units/channels for the group (ie indirect units)
args: 
	f_out: path of output hdf5 file
	event_type: str designation of event to save data around
	datawin: [pre, post] in ms to take window around event timestamp
"""
	##the location to save the data when done
f_out = r"C:\Users\Ryan\Documents\data\non_task_times.hdf5"
event_type = 't1'
datawin=[6000,2000]
def save_triggered(f_out, event_type, datawin=[6000,2000]):
	#create the output file
	F = h5py.File(f_out, 'w-')
	##the dictionary containing all the metadata
	LUT = ru.animals
	##the containing folder for all the data
	data_folder = r"C:\Users\Ryan\Documents\data\V1_BMI"
	##determine which files in this directory overlap with the animals
	##in our lookup table
	animals = list(set(LUT.keys()).intersection(os.listdir(data_folder)))
	##ask the user which of these animals to use for further parsing
	animals = gs.multichoice(animals, title = "Select animals")
	##ask the user which sessions to use for each animal. keep these decisions in a dict
	sessions_dict = {}
	for a in animals:
		##determine which files in each animal dir overlap with what
		##we have in our LUT
		sessions = list(set(LUT[a].keys()).intersection(os.listdir(os.path.join(data_folder,a))))
		##ask user which of these to keep
		sessions = gs.multichoice(sessions, title="Select sessions for "+a)
		##if user picked any sessions, add them to the dictionary
		if len(sessions)>0:
			sessions_dict[a]=sessions
	##let's validate the info we have in our LUT, and set up datasets while we're at it
	##-We need a dataset for each signal type
	##-Each dataset is 3-D, with dimensions:
	##	-time(set by datawin)
	##	-trials, which is the total number of events across all sessions/animals
	##	-units, which is the total number of units/channels for each group across sessions/animals
	for a in sessions_dict.keys():
		for s in sessions_dict[a]:
			signal_types = {} ##this will contain the number of channels/units for each signal type (axis 2)
			num_events = 0 ##the number of trials (axis 0); axis 1 = the time window
			##get the metadata from this session file
			meta = plxread.read_plx_headers(os.path.join(data_folder,a,s))
			##figure out if this session even has any of the events we're interested in
			event_id = LUT[a][s]['events'][event_type][0] ##the string identifyer of the event
			event_num = int(event_id[-3:]) ##the event's number in integer format
			##make sure this event exists in the actual data file
			try:
				num_events = meta['event_counts'][event_num]
			except KeyError:
				print event_id+" does not exist for "+a+" "+s+". skipping..."
				num_events = 0
			##if there are no events for this event type, get rid of this session altogether
			if num_events <= 2:
				print "Not enough events for "+a+" "+s+". skipping..."
				del sessions_dict[a][sessions_dict[a].index(s)]
			else:
				##add the counts of units in each group to the total
				for sig in LUT[a][s]['units'].keys():
					##add this sig type if not there already
					if sig not in signal_types.keys():
						signal_types[sig] = 0
					##the units in this group according to our records
					unit_members = LUT[a][s]['units'][sig]
					##double check that these units are contained in the actual data file
					for i, u in enumerate(unit_members):
						if parse_spk_chan(u) not in meta['spike_counts'].keys():
							print u+" ("+a+" "+s+") not found. Deleting"
							del LUT[a][s]['units'][sig][i]
					##tally the number of units after getting rid of any mistakes
					signal_types[sig]+=len(unit_members)
				##now repeat for LFP channels
				for sig in LUT[a][s]['lfp'].keys():
					##add this sig type if not there already
					if sig not in signal_types.keys():
						signal_types[sig] = 0
					##the units in this group according to our records
					unit_members = LUT[a][s]['lfp'][sig]
					##double check that these units are contained in the actual data file
					for i, u in enumerate(unit_members):
						if parse_AD_chan(u) not in meta['slow_ch_counts'].keys():
							print u+" ("+a+" "+s+") not found. Deleting"
							del LUT[a][s]['lfp'][sig][i]
					##tally the number of units after getting rid of any mistakes
					signal_types[sig]+=len(unit_members)
				##create a group for this session
				grp =F.create_group(a+"_"+s[-7:-4])
				##now let's initialize the datasets to the correct size:
				for x in signal_types.keys():
					grp.create_dataset(x, (num_events, datawin[0]+datawin[1], signal_types[x]))
	##Now that we know exactly how much data we're saving, let's save the data
	for a in sessions_dict.keys():
		for s in sessions_dict[a]:
			##get the handle to the current group we're going to save data in
			try:
				grp = F[a+"_"+s[-7:-4]]
			except KeyError:
				print "Can't find group "+a+"_"+s[-7:-4]
			data = plxread.import_file(os.path.join(data_folder,a,s),AD_channels=range(1,97),save_wf=False,
				import_unsorted=False,verbose=False)
			##get the timestamps for the events; convert to ms
			ts = data[LUT[a][s]['events'][event_type][0]]*1000.0+8000
			##get the window of data for across all signal types
			for j,t in enumerate(ts):
				##start with the spike data
				for g in LUT[a][s]['units'].keys(): ##the group types
					for i, u in enumerate(LUT[a][s]['units'][g]): ##the unit names
						##gets the timestamps in the window around this event ts
						dseg = data[u][np.where((data[u]*1000.0>t-datawin[0])&(data[u]*1000.0<t+datawin[1]))]
						##do a little conversion: convert spike ts to ms, and change the times to be relative to the event
						dseg = (dseg*1000.0)-t
						##convert to binary array
						dseg = np.histogram(dseg, bins= datawin[0]+datawin[1], 
							range=(-datawin[0],datawin[1]))[0].astype(bool).astype(int)
						##add this data to the dataset
						grp[g][j,:,i] = dseg
				##repeat for LFP data
				for g in LUT[a][s]['lfp'].keys(): ##the group types
					for i, u in enumerate(LUT[a][s]['lfp'][g]): ##the channel names
						##this is a little nuanced: Need to check the LFP timestamps to get the 
						##index values for the voltage array (could be gaps due to pauses in the recording)
						##gets the timestamps in the window around this event ts
						dseg = data[u][np.where((data[u+"_ts"]*1000.0>t-datawin[0])&(data[u+"_ts"]*1000.0<=t+datawin[1]))]
						##add this data to the dataset. Could be that the segment is truncated (less than 
						#the full datawin size) due to the start or end of the file. In that case, warn the 
						##user and append zeros
						try:
							grp[g][j,:,i] = dseg
						except TypeError:
							print "Appending zeros to LFP trace for trial "+str(j)+" "+s+" "+a
							print "Trace length is " + str(dseg.size)
							if dseg.size < datawin[0]+datawin[1]:
								temp = np.zeros(datawin[0]+datawin[1])
								temp[0:dseg.size] = dseg
								grp[g][j,:,i] = temp
							else:
								grp[g][j,:,i] = dseg[0:datawin[0]+datawin[1]]
	##all done! Close out the file
	F.close()
	print "Done!"


##helper function to parse spike channel names
def parse_spk_chan(name):
	channel_num = int(name[3:6])
	channel_letter = name[6]
	if channel_letter == 'a':
		unit_number = 1
	elif channel_letter == 'b':
		unit_number = 2
	elif channel_letter == 'c':
		unit_number = 3
	elif channel_letter == 'd':
		unit_number = 4
	elif channel_letter == 'e':
		unit_number = 5
	else:
		unit_number = 6
	return (channel_num, unit_number)


##helper function to parse AD channel name
##into a number that corresponds with metadata
def parse_AD_chan(name):
	channel_num = int(name[-2:])+299
	return channel_num
