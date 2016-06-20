##data_to_gdf.py
##sigh... OK so I want to use the gravity analysis program 
##but the damn thing is written in matlab with fortran
##files and it's just wayyy to much to re-write in python and 
##C++, so I'm just going to conform. 

##unfortunately the program requires all the data to be multiplexed
##into some weird ".gdf" file format...ghey. 

##hence this script, which will be designed to use my existing 
##scripts and basically scramble everything back into one messy 
##GDF file. 
import numpy as np


"""
this function takes in an array of data (spike or event data),
converts the ID to some unique numeric code, and then creates 
ID-timestamp pairs organizes as two matched arrays.
"""
def spike_to_gdf(array_in, arr_id):
	##make sure the array is 1-D
	array_in = np.squeeze(array_in)
	##make sure we are dealing with a binary array_in
	if array_in.max() != 1.0 or array_in.min() != 0.0:
		raise ValueError("This array contains non-binary data!")
	else:
		##get the timestamps where a spike/event occurs
		timestamps = np.where(array_in == 1.0)[0]
		##get the ID for this array
		num_id = gen_spike_id(arr_id)
		##create a matching-length array of ID numbers
		##and concatenate them
		ids = np.tile(num_id, timestamps.size)
		return np.vstack((ids, timestamps))
		
##same as above but for event arrays which are already saved as timestamps
def event_to_gdf(array_in, event_id):
	timestamps = np.squeeze(array_in)
	##get the numeric event id
	num_id = gen_event_id(event_id)
	##create a matching-length array of ID numbers
	##and concatenate them
	ids = np.tile(num_id, timestamps.size)
	return np.vstack((ids, timestamps))

## a function to generate a unique numeric ID given a session string in the 
#format 'R11_BMI_D03.plx_sig025a'
def gen_spike_id(str_in):
	lut = {
	'a':'1',
	'b':'2',
	'c':'3',
	'd':'4'
	}
	result = str_in[1:3]+str_in[9:11]+str_in[-3:-1]+lut[str_in[-1]]
	return int(result)

##same as above but for events
def gen_event_id(str_in):
	lut = {
	"t1":1,
	"t2":2,
	"miss":3
	}
	result = lut[str_in]
	return int(result)

## a function to sort a bunch of arrays so that the timestamps are in ascending order
##but the spike/event IDs still are also sorted to match
##input is a big concatenated array in shape 2 x num_events (index zero is ids, 
#1 is timestamps)
def sort_by_timestamp(concat_arr):
	##split the array into its component parts
	ids = concat_arr[0,:]
	ts = concat_arr[1,:]
	##get the indices of the sorted timestmaps
	idx = np.argsort(ts)
	##apply sort to both arrays
	ids = ids[idx]
	ts = ts[idx]
	return np.vstack((ids, ts))

##this function takes in a corted data array as produced by the 
##above function, and saves it in the proper gdf format at the 
##specified location.
def save_as_gdf(f_out, data):
	##save the data using the correct parameters
	np.savetxt(f_out, data.T, delimiter = " ", fmt = "%i")
	print "Data saved!"


