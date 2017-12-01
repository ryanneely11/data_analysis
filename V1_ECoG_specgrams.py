import h5py
from SpikeStats2 import lfpSpecGram
from scipy.signal import butter, lfilter
import multiprocessing as mp
import numpy as np
import os

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

<<<<<<< HEAD
file_in = "/Volumes/Untitled/Ryan/11_22_16_SiC/PI_1"
file_out = "/Volumes/Untitled/Ryan/11_22_16_SiC/specgrams/PI_1"
angle_list = ['-1', '0', '135', '180', '225', '270', '315', '45', '90']

def save_specgrams(angle):
	f = h5py.File("/Volumes/Untitled/Ryan/11_22_16_SiC/PI_1",'r')
	save_folder = "/Volumes/Untitled/Ryan/11_22_16_SiC/specgrams/PI_1"
=======
# file_in = r"J:\Ryan\11_22_16_SiC\SiC_2.hdf5"
# file_out = r"J:\Ryan\11_22_16_SiC\SiC_2_specgrams"
angle_list = ['-1', '0', '135', '180', '225', '270', '315', '45', '90']

def save_specgrams(angle):
	f = h5py.File(r"J:\Ryan\11_22_16_SiC\SiC_1.hdf5",'r')
	save_folder = r"J:\Ryan\11_22_16_SiC\specgrams\SiC_1"
>>>>>>> ba6a4d86489c3a36bbe545437dbe152d50ff60b2
	#get the data for the given angle
	set_list = f.keys()
	#shape of data for one trial
	trial_shape = f['set_1']['0'].shape
	#the shape of the data container for all trials
	trial_shape = [len(set_list),trial_shape[0],trial_shape[1]]
	##container for the data
	data = np.zeros(trial_shape)
	for i, s in enumerate(set_list):
		data[i,:,:] = np.asarray(f[s][angle])
	f.close()
	##filter the data
	for trial in range(data.shape[0]):
		for chan in range(data.shape[1]):
			data[trial][chan][:] = butter_bandpass_filter(data[trial][chan][:],
                                                     0.5,300,24414,order=1)
	spec_by_channel = []
	for chan in range(data.shape[1]): ##number of channels
		dset = data[:,chan,:]
		spec,t,fr,serr = lfpSpecGram(dset.T,window=[0.5,0.025],Fs=24414,fpass=[0,200])
		spec_by_channel.append(spec)
	result = np.asarray(spec_by_channel)
	f_out = h5py.File(os.path.join(save_folder,angle+".hdf5"),'w-')
	f_out.create_dataset("specgrams",data=result)
	f_out.create_dataset("f",data=fr)
	f_out.create_dataset("t",data=t)
	f_out.close()
	return 0
		
def mp_spec():
	pool = mp.Pool(processes = len(angle_list))
	async_result = pool.map_async(save_specgrams, angle_list)
	pool.close()
	pool.join()
	print "All processes have returned; parsing results"
