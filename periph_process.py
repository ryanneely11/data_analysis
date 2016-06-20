##just a quick function to process peripheral recordings.

import numpy as np

def get_data(arr, num_samples):
	## get just the voltage data
	arr1 = arr[:,1]
	##figure out how many sweeps are in the dataset
	num_sweeps = arr1.size/num_samples
	##separate out each sweeps
	split_arr = np.split(arr1, num_sweeps)
	#subtract the mean to zero the data at the origin
	for i in range(len(split_arr)):
		split_arr[i] = split_arr[i] - np.mean(split_arr[i])
	##convert the split arrays into a 2-D array
	arr2 = np.asarray(split_arr)
	return arr2