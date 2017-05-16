##SpikeStats2: a series of data analysis routines 

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import spectrum as spec
from progressbar import *
from scipy.stats.mstats import zscore
from scipy.stats import sem
#import DataSet3 as ds
import h5py
#import gc
import scipy.stats as stats
import matplotlib as mp
import scipy as sp
import math
#import h5py


def gauss_convolve(array, sigma):
	"""
	takes in an array with dimenstions samples x trials.
	Returns an array of the same size where each trial is convolved with
	a gaussian kernel with sigma = sigma.
	"""
	##remove singleton dimesions and make sure values are floats
	array = array.squeeze().astype(float)
	##allocate memory for result
	result = np.zeros(array.shape)
	##if the array is 2-D, handle each trial separately
	try:
		for trial in range(array.shape[1]):
			result[:,trial] = gaussian_filter(array[:, trial], sigma = sigma, order = 0, mode = "constant", cval = 0.0)
	##if it's 1-D:
	except IndexError:
		if array.shape[0] == array.size:
			result = gaussian_filter(array, sigma = sigma, order = 0, mode = "constant", cval = 0.0)
		else:
			print("Check your array dimenszions!")
	return result

def windowRate(data, movingwin):
	"""
	This function takes in a binary spike array and returns the sliding window spike rate in spikes/sec.
	The arguments are a  np array, a window size (int,ms) and a step size (int,ms)
	It returns an array with the sliding window rate.
	"""
	if len(data.shape) > 1:
		N = data.shape[0]
		num_trials = data.shape[1]
	else:
		N = data.size
		num_trials = 1
		data = data.reshape(N,1)
	Nwin = int(round(movingwin[0])) ##window size in samples
	Nstep = int(round(movingwin[1])) ##step size in samples
	winstart = np.arange(0,N-Nwin,Nstep) ##array of all the window starting values
	nw = winstart.shape[0] ##number of total windows
	result = np.zeros((nw,num_trials))
	for t in range(num_trials):
		for n in range(nw):
			indx = np.arange(winstart[n],winstart[n]+Nwin) ##index values to take from data based on current window
			eventswin = data[indx,t] ##current data window for all lfp segments
			#calculate rate for the given window
			##convert nwin to mins 
			secs = Nwin/1000.0
			result[n,t] = eventswin.sum()/secs
	return result.squeeze()


"""
This function takes in a data array with dimensions values x trials
and calculates the mean and standard error of the mean across trials for each data value. 
The returned arrays are the mean (N-values) and the standard error for each point (N-values).
"""
def mean_and_sem(data):
	mean = data.mean(axis = 1)
	error = sem(data, axis = 1)
	return mean, error

def thin_spikes(data1, data2, sigma):
	"""
	This function takes as args two sets of data, each is a matrix of binary spike
	trains with dimensions samples x trials.
	****DATA MATRICES SHOULD BE THE SAME DIMENSIONS!****
	The function then calculates the probability that spikes need to be thinned
	from one of the conditions at each time bin, and thins the spike trains
	accordingly to equate FRs over both sets of data.
	Sigma is the width of the gaussian kernel used to convolve the spike trains.
	"""
	##check to see if data matrices are the same dimensions
	if data1.shape != data2.shape:
		print("Data matrices are not the same size!")
	print("data1 shape is " + str(data1.shape[0]) + ", " + str(data1.shape[1]))
	print("data2 shape is " + str(data2.shape[0]) + ", " + str(data2.shape[1]))
		
	#convolve spike trains with a gaussian kernel
	set_1 = gauss_convolve(data1, sigma)
	set_2 = gauss_convolve(data2, sigma)

	#get the across-trials average firing rate
	set_1 = set_1.mean(axis = 1)
	set_2 = set_2.mean(axis = 1)

	#get probability that a spike needs to be removed from higher FR condition at each time bin

	#container for prob a spike needs to be removed from higher FR condition
	prob_removal = np.zeros(data1.shape[0])
	#container for which condition has a higher FR at that bin
	higher_fr = np.zeros(data1.shape[0])
	
	for i in range(data1.shape[0]):
		##catch the case where both sets == 0
		if set_1[i] == 0 and set_2[i] ==0:
			prob_removal[i] = 0
		else:
			prob_removal[i] = abs(set_1[i]-set_2[i])/max(set_1[i], set_2[i])
		if set_1[i] > set_2[i]:
			higher_fr[i] = 1
		elif set_2[i] > set_1[i]:
			higher_fr[i] = 2

	#now, going back to the original data sets, remove spikes according to the calculated probability
	##***EDIT*** added in a check to make sure that the algorithm doesn't remove ALL of the spikes; it
	##can result in zero-division errors when used with other analyses.
	print("data1 shape is " + str(data1.shape[0]) + ", " + str(data1.shape[1]))
	print("data2 shape is " + str(data2.shape[0]) + ", " + str(data2.shape[1]))
	print("shape of higher_fr is " + str(higher_fr.shape[0]))
	print("shape of prob_removal is " +str(prob_removal.shape[0]))
	for trial in np.arange(0, data1.shape[1]):
		for bin in np.arange(0,data1.shape[0]):
			if higher_fr[bin] == 1 and np.random.random() < prob_removal[bin] and data1[:,trial].sum() > 1.0:
				data1[bin][trial] = 0.0
			elif higher_fr[bin] == 2 and np.random.random() < prob_removal[bin] and data2[:,trial].sum() > 1.0:
				data2[bin][trial] = 0.0

def spike_field_cohgram(spikes, lfp, movingwin=[0.3,0.01], Fs = 1000.0, fpass = [0, 100], err = None, save_data = False):
	"""
	Calculate the coherence values over time by using a sliding window. Inputs are
	an array of binary spike trains in the format samples x trials, an array
	of ****MATCHING ORDERED**** lfp signals in the same format, a 
	movingwin parameter in the format [window, winstep], sample rate in Hz and 
	the frequency window to restrict analysis to. 
	"""
	if save_data:
		f_path = raw_input("Type file path:  ")

	lfp = lfp.squeeze() ##get rid of singleton dimensions for the next step
	spikes = spikes.squeeze()
	if len(lfp.shape) > 1: ##if there is more than one trace, set N and numTraces appropriately
		numTraces = lfp.shape[1]
	else: ##if there is only 1, set N and numTraces
		lfp = lfp[:,None]
		spikes = spikes[:,None]
		numTraces = 1

	if lfp.shape[0] != spikes.shape[0]:
		if lfp.shape[0] > spikes.shape[0]:
			lfp = lfp[0:spikes.shape[0],:]
		else:
			spikes = spikes[0:lfp.shape[0],:]
	N = lfp.shape[0]
	Nwin = int(round(movingwin[0]*Fs)) ##window size in samples
	Nstep = int(round(movingwin[1]*Fs)) ##step size in samples
	nfft = (2**spec.nextpow2(Nwin)) ##the nfft length for the given window
	winstart = np.arange(0,N-Nwin,Nstep) ##array of all the window starting values
	nw = winstart.shape[0] ##number of total windows

	##get the frequency grid values based on user input
	f,findx = getfgrid(Fs,nfft,fpass)
	Nf = len(f)
	#container for the coherence data
	C = np.zeros((nw,Nf))
	S12 = np.zeros((nw, Nf))
	S1 = np.zeros((nw, Nf))
	S2 = np.zeros((nw, Nf))
	phi = np.zeros((nw, Nf))
	Cerr = np.zeros((2, nw, Nf))
	phistd = np.zeros((nw, Nf))

	zerosp = np.zeros((nw, numTraces))
	#create a progressbar (because this can take a while...)
	pbar = ProgressBar(maxval = nw).start()
	p = 0
	for n in range(nw):
		indx = np.arange(winstart[n],winstart[n]+Nwin) ##index values to take from data based on current window
		spikeswin = spikes[indx,:] ##current data window for all lfp segments
		lfpwin = lfp[indx,:] ##current data window for all segments
		#calculate coherence for the given window
		c, ph, s12, s1, s2, f, zsp, confc, phie, cerr = spike_field_coherence(spikeswin,
			lfpwin, Fs = Fs, fpass = fpass, trialave = True, err = err)
		C[n,:] = c
		phi[n,:] = ph
		S12[n,:] = s12
		S1[n,:] = s1
		S2[n,:] = s2
		zerosp[n,:] = zsp
		if err is not None:
			phistd[n,:] = phie
			Cerr[0,n,:] = cerr[0,:].squeeze()
			Cerr[1,n,:] = cerr[1,:].squeeze()
		pbar.update(p+1)
		p+=1
	pbar.finish()

	C = C.squeeze(); S12 = S12.squeeze(); S1 = S1.squeeze(); S2 = S2.squeeze(); phi = phi.squeeze()
	zerosp = zerosp.squeeze()
	#calculate the time axis values
	winmid=winstart+round(Nwin/2)
	t=winmid/Fs

	if save_data:
		g = h5py.File(f_path, 'w-')
		g.create_dataset("C", data = C)
		g.create_dataset("S12", data = S12)
		g.create_dataset("S1", data = S1)
		g.create_dataset("S2", data = S2)
		g.create_dataset("phi", data = phi)
		g.close()
		print("data saved.")

	return C, phi, S12, S1, S2, t, f, zerosp, confc, phistd, Cerr


##a function for use in multiprocessing cohgrams
##a function to parse the arguments and run the calculation
def mp_cohgrams_sf(args):
	spikes = args[0]
	lfp = args[1]
	thin_spikes(spikes[0:spikes.shape[0]/2,:], spikes[spikes.shape[0]/2:,:],10)
	C, phi, S12, S1, S2, t, f, zerosp, confc, phistd, Cerr= spike_field_cohgram(spikes,lfp)
	return C


def field_field_cohgram(lfp1, lfp2, movingwin = [0.3,0.01], Fs = 1000.0, fpass = [0, 100], err = None, save_data = False):
	"""
	Calculate the coherence values over time by using a sliding window. Inputs are
	an array of lfp traces in the format samples x trials, an array
	of ****MATCHING ORDERED**** lfp signals in the same format, a 
	movingwin parameter in the format [window, winstep], sample rate in Hz and 
	the frequency window to restrict analysis to. 
	"""
	if save_data:
		f_path = raw_input("Type file path:  ")

	lfp1 = lfp1.squeeze() ##get rid of singleton dimensions for the next step
	lfp2 = lfp2.squeeze()
	N = lfp1.shape[0] ##length of LFP segment
	numTraces = lfp1.shape[1]

	Nwin = int(round(movingwin[0]*Fs)) ##window size in samples
	Nstep = int(round(movingwin[1]*Fs)) ##step size in samples
	nfft = (2**spec.nextpow2(Nwin)) ##the nfft length for the given window
	winstart = np.arange(0,N-Nwin,Nstep) ##array of all the window starting values
	nw = winstart.shape[0] ##number of total windows

	##get the frequency grid values based on user input
	f,findx = getfgrid(Fs,nfft,fpass)
	Nf = len(f)
	#container for the coherence data
	C = np.zeros((nw,Nf))
	S12 = np.zeros((nw, Nf))
	S1 = np.zeros((nw, Nf))
	S2 = np.zeros((nw, Nf))
	phi = np.zeros((nw, Nf))
	Cerr = np.zeros((2, nw, Nf))
	phistd = np.zeros((nw, Nf))

	#create a progressbar (because this can take a while...)
	pbar = ProgressBar(maxval = nw).start()
	p = 0
	for n in range(nw):
		indx = np.arange(winstart[n],winstart[n]+Nwin) ##index values to take from data based on current window
		lfp1win = lfp1[indx,:] ##current data window for all lfp segments
		lfp2win = lfp2[indx,:] ##current data window for all segments
		#calculate coherence for the given window
		c, ph, s12, s1, s2, f, confc, phie, cerr = field_field_coherence(lfp1win,
			lfp2win, Fs = Fs, fpass = fpass, trialave = True, err = err)
		C[n,:] = c
		phi[n,:] = ph
		S12[n,:] = s12
		S1[n,:] = s1
		S2[n,:] = s2
		if err is not None:
			phistd[n,:] = phie
			Cerr[0,n,:] = cerr[0,:].squeeze()
			Cerr[1,n,:] = cerr[1,:].squeeze()
		pbar.update(p+1)
		p+=1
	pbar.finish()

	C = C.squeeze(); S12 = S12.squeeze(); S1 = S1.squeeze(); S2 = S2.squeeze(); phi = phi.squeeze()
	#calculate the time axis values
	winmid=winstart+round(Nwin/2)
	t=winmid/Fs

	if save_data:
		g = h5py.File(f_path, 'w-')
		g.create_dataset("C", data = C)
		g.create_dataset("S12", data = S12)
		g.create_dataset("S1", data = S1)
		g.create_dataset("S2", data = S2)
		g.create_dataset("phi", data = phi)
		g.close()
		print("data saved.")

	return C, phi, S12, S1, S2, t, f, confc, phistd, Cerr

##a function for use in multiprocessing cohgrams
##a function to parse the arguments and run the calculation
def mp_cohgrams(args):
	lfp_1 = args[0]
	lfp_2 = args[1]
	C, phi, S12, S1, S2, t, f, confc, phistd, Cerr= field_field_cohgram(lfp_1,lfp_2)
	return C


def spike_spike_cohgram(spikes1, spikes2, movingwin, Fs = 1000.0, fpass = [0,100], err = None, save_data = False):
	"""
	Calculate the coherence values over time by using a sliding window. Inputs are
	2 arrays of binary spike trains in the format samples x trials;
	***MATCHING ORDERED****. A 
	movingwin parameter in the format [window, winstep], sample rate in Hz and 
	the frequency window to restrict analysis to. 
	"""
	if save_data:
		f_path = raw_input("Type file path:  ")

	spikes1 = spikes1.squeeze() ##get rid of singleton dimensions for the next step
	spikes2 = spikes2.squeeze()
	N = spikes1.shape[0] ##length of LFP segment
	numTraces = spikes1.shape[1]

	Nwin = int(round(movingwin[0]*Fs)) ##window size in samples
	Nstep = int(round(movingwin[1]*Fs)) ##step size in samples
	nfft = (2**spec.nextpow2(Nwin)) ##the nfft length for the given window
	winstart = np.arange(0,N-Nwin,Nstep) ##array of all the window starting values
	nw = len(winstart) ##number of total windows

	f,findx = getfgrid(Fs,nfft,fpass)
	Nf = len(f)

	#container for the coherence data
	C = np.zeros((nw,Nf))
	S12 = np.zeros((nw, Nf))
	S1 = np.zeros((nw, Nf))
	S2 = np.zeros((nw, Nf))
	phi = np.zeros((nw, Nf))
	Cerr = np.zeros((2, nw, Nf))
	phistd = np.zeros((nw, Nf))

	zerosp = np.zeros((nw, numTraces))
	#create a progressbar (because this can take a while...)
	pbar = ProgressBar(maxval = nw).start()
	p = 0
	for n in range(nw):
		indx = np.arange(winstart[n],winstart[n]+Nwin) ##index values to take from data based on current window
		spikeswin1 = spikes1[indx,:] ##current data window for all lfp segments
		spikeswin2 = spikes2[indx,:] ##current data window for all segments
		#calculate coherence for the given window
		c, ph, s12, s1, s2, f, zsp, confc, phie, cerr = spike_spike_coherence(spikeswin1,
			spikeswin2, Fs = Fs, fpass = fpass, trialave = True, err = err)
		C[n,:] = c
		phi[n,:] = ph
		S12[n,:] = s12
		S1[n,:] = s1
		S2[n,:] = s2
		zerosp[n,:] = zsp
		if err is not None:
			phistd[n,:] = phie
			Cerr[0,n,:] = cerr[0,:].squeeze()
			Cerr[1,n,:] = cerr[1,:].squeeze()
		pbar.update(p+1)
		p+=1
	pbar.finish()

	C = C.squeeze(); S12 = S12.squeeze(); S1 = S1.squeeze(); S2 = S2.squeeze(); phi = phi.squeeze()
	zerosp = zerosp.squeeze()
	#calculate the time axis values
	winmid=winstart+round(Nwin/2)
	t=winmid/Fs

	if save_data:
		g = h5py.File(f_path, 'w-')
		g.create_dataset("C", data = C)
		g.create_dataset("S12", data = S12)
		g.create_dataset("S1", data = S1)
		g.create_dataset("S2", data = S2)
		g.create_dataset("phi", data = phi)
		g.close()
		print("data saved.")

	return C, phi, S12, S1, S2, t, f, zerosp, confc, phistd, Cerr


def spike_spike_coherence(spikes1, spikes2, Fs = 1000.0, fpass = [0,100], trialave = True, err = None):

	spikes1 = spikes1.squeeze() ##get rid of singleton dimensions for the next step
	spikes2 = spikes2.squeeze()
	N = spikes1.shape[0] ##length of LFP segment
	numTraces = spikes1.shape[1] ##number of traces in dataset

	nfft = 2**spec.nextpow2(N) ##pad data to the next power of 2 from length of data (makes computation faster)

	zerosp = np.zeros((1,numTraces)) ##vector to store info about trials where no spikes occurred

	f, findx = getfgrid(Fs, nfft, fpass)

	J1, Msp1, Nsp1 = fft_spikes(spikes1)
	J2, Msp2, Nsp2 = fft_spikes(spikes2)
	zerosp = np.zeros(numTraces)
	zerosp[Nsp1==0] = 1
	zerosp[Nsp2==0] = 1 ##set the trials where no spikes were found to have zerosp = 1
	J1 = J1[findx,:,:]
	J2 = J2[findx,:,:]
	S12 = np.squeeze(np.mean(J1.conj()*J2,axis = 1))
	S1 = np.squeeze(np.mean(J1.conj()*J1,axis = 1))
	S2 = np.squeeze(np.mean(J2.conj()*J2, axis = 1))
	if trialave:
		S12 = np.squeeze(np.mean(S12,axis = 1))
		S1 = np.squeeze(np.mean(S1, axis = 1))
		S2 = np.squeeze(np.mean(S2, axis = 1))
	C12 = S12/np.sqrt(S1*S2)
	C = abs(C12)
	phi = np.angle(C12)
	if err is not None:
		confC, phistd, Cerr = coherr(C, J1, J2, err, trialave)
	else:
		confC = 0; phistd = 0; Cerr = 0

	return C, phi, S12, S1, S2, f, zerosp, confC, phistd, Cerr

def field_field_coherence(lfp1, lfp2, Fs = 1000.0, fpass = [0,100], trialave = True, err = None):
	"""
	This is the field-field coherence algorithm used in the
	Koralek(2012) and Koralek(2013), and Gregoriou et al(2009).
	Inputs are an array of lfp traces 
	trains in the format samples x trials, and LFP2 traces in the same
	order, the sample rate in Hz, and the frequency range to analyze
	"""

	lfp1 = lfp1.squeeze() ##get rid of singleton dimensions for the next step
	lfp2 = lfp2.squeeze()
	N = lfp1.shape[0] ##length of LFP segment
	numTraces = lfp1.shape[1] ##number of traces in dataset

	nfft = 2**spec.nextpow2(N) ##pad data to the next power of 2 from length of data (makes computation faster)

	f, findx = getfgrid(Fs, nfft, fpass)

	J1 = fft_lfp(lfp1) ##multi-taper fft of lfp data
	J2 = fft_lfp(lfp2) #mtfft of spike data
	##pare down the FFT data to incude only freqs in the fpass range
	J1 = J1[findx,:,:]
	J2 = J2[findx,:,:]
	#calculate the cross-spectrum of the two signals,
	#taking the average of the tapered components
	S12 = np.squeeze(np.mean(J1.conj()*J2,axis = 1))
	#calculate the autospectra of the two signals
	S1 = np.squeeze(np.mean(J1.conj()*J1,axis = 1))
	S2 = np.squeeze(np.mean(J2.conj()*J2, axis = 1))
	#average the cross- and auto-spectra before calculating coherence
	##(as per Koralek et al,(2012, 2013) and Gregoriou et al, 2009)
	if trialave:
		S12 = np.squeeze(np.mean(S12,axis = 1))
		S1 = np.squeeze(np.mean(S1, axis = 1))
		S2 = np.squeeze(np.mean(S2, axis = 1))
	##calculate the coherence
	C12 = S12/np.sqrt(S1*S2)
	#return the absolute value of the coherence (and the freq grid values)
	C = abs(C12)
	phi = np.angle(C12)
	if err is not None:
		confC, phistd, Cerr = coherr(C, J1, J2, err, trialave)
	else:
		confC = 0; phistd = 0; Cerr = 0

	return C, phi, S12, S1, S2, f, confC, phistd, Cerr


def spike_field_coherence(spikes, lfp, Fs = 1000.0, fpass = [0,100], trialave = True, err = None):
	"""
	This is the spike-field coherence algorithm used in the
	Koralek(2012) and Koralek(2013), and Gregoriou et al(2009).
	Inputs are an array of binary spike 
	trains in the format samples x trials, and LFP traces in the same
	order, the sample rate in Hz, and the frequency range to analyze
	"""

	lfp = lfp.squeeze() ##get rid of singleton dimensions for the next step
	spikes = spikes.squeeze()
	if len(lfp.shape) > 1: ##if there is more than one trace, set N and numTraces appropriately
		N = lfp.shape[0]
		numTraces = lfp.shape[1]
	else: ##if there is only 1, set N and numTraces
		N = len(lfp)
		lfp = lfp[:,None]
		spikes = spikes[:,None]
		numTraces = 1

	nfft = 2**spec.nextpow2(N) ##pad data to the next power of 2 from length of data (makes computation faster)

	zerosp = np.zeros(numTraces) ##vector to store info about trials where no spikes occurred

	f, findx = getfgrid(Fs, nfft, fpass)

	J1 = fft_lfp(lfp) ##multi-taper fft of lfp data
	J2, Msp2, Nsp2 = fft_spikes(spikes) #mtfft of spike data
	zerosp[Nsp2==0] = 1 ##set the trials where no spikes were found to have zerosp = 1
	##pare down the FFT data to incude only freqs in the fpass range
	J1 = J1[findx,:,:]
	J2 = J2[findx,:,:]
	#calculate the cross-spectrum of the two signals,
	#taking the average of the tapered components
	S12 = np.mean(J1.conj()*J2,axis = 1)
	#calculate the autospectra of the two signals
	S1 = np.mean(J1.conj()*J1,axis = 1)
	S2 = np.mean(J2.conj()*J2, axis = 1)
	#average the cross- and auto-spectra before calculating coherence
	##(as per Koralek et al,(2012, 2013) and Gregoriou et al, 2009)
	if trialave:
		S12 = np.mean(S12,axis = 1)
		S1 = np.mean(S1, axis = 1)
		S2 = np.mean(S2, axis = 1)
	##calculate the coherence
	C12 = S12/np.sqrt(S1*S2)
	#return the absolute value of the coherence (and the freq grid values)
	C = abs(C12)
	phi = np.angle(C12)
	if err is not None:
		confC, phistd, Cerr = coherr(C, J1, J2, err, trialave)
	else:
		confC = 0; phistd = 0; Cerr = 0

	return C, phi, S12, S1, S2, f, zerosp, confC, phistd, Cerr


def lfpSpecGram(data, window=[0.75,0.05], Fs = 1000.0, fpass = [0,100], err = None, 
	sigType = 'lfp', norm = True):
	"""
	Basically just a moving window average of the spectrum. Window 
	and inputs should be in sec: [win length, win step]
	fpass should be [fmin, fmax]. Input data is an array; samples x trials
	"""
	data = data.squeeze() ##get rid of singleton dimensions for the next step
	if len(data.shape) > 1: ##if there is more than one trace, set N and numTraces appropriately
		N = data.shape[0]
		num_traces = data.shape[1]
	else: ##if there is only 1, set N and numTraces
		N = len(data)
		num_traces = 1
		data = data[:,None]
	#Fs = 1000.0
	Nwin = int(round(window[0]*Fs)) ##window size
	Nstep = int(round(window[1]*Fs)) ##step size
	nfft = (2**spec.nextpow2(Nwin)) ##the nfft length for the given window

	f, findx =getfgrid(Fs,nfft,fpass)
	Nf=f.shape[0]
	winstart = np.arange(0,N-Nwin,Nstep) ##array of all the window starting values
	nw = winstart.shape[0] ##number of total windows

	S = np.zeros([nw, Nf]) ##output array of size num total windows x num freqs
	Serr = np.zeros((2,nw,Nf))

	pbar = ProgressBar(maxval = nw).start()
	p = 0
	for n in range(nw):
		indx = np.arange(winstart[n],winstart[n]+Nwin) ##index values to take from data based on current window
		datawin = data[indx, :] ##current data window for all segments
		##compute each spectrogram, for the window, then average them all together
		s, f, serr = mtspectrum(datawin, Fs = Fs, fpass = fpass, trialave = True, err = err, sigType = sigType)
		S[n,:] = np.squeeze(s) ##add the spectrum of this window to the total array
		Serr[0,n,:] = np.squeeze(serr[0,:])
		Serr[1,n,:] = np.squeeze(serr[1,:])
		pbar.update(p+1)
		p+=1
	pbar.finish()
	S = np.squeeze(S); Serr = Serr.squeeze()
	winmid=winstart+round(Nwin/2)
	t=winmid/Fs
	if norm:
		for i in range(S.shape[1]):
			S[:,i] = S[:,i]/S[:,i].mean()
	return S, t, f, Serr

"""
A function that takes the FFT of one or a set of binary spike trains individually.
The data parameter is an array of lfp traces in the format samples x trials,
Fs is the sample rate.
"""
def fft_spikes(data):
	Fs = 1000.0
	#print 'Calculating multi-taper fft of spike data...'
	data = data.squeeze() ##get rid of singleton dimensions for the next step
	if len(data.shape) > 1: ##if there is more than one trace, set N and numTraces appropriately
		N = data.shape[0]
		numTraces = data.shape[1]
	else: ##if there is only 1, set N and numTraces
		N = len(data)
		numTraces = 1
		##add a singleton dimension to make data "2d"
		data = data[:,None]

	nfft = 2**spec.nextpow2(N)
	tapers, eigs = spec.dpss(N, 3, 5) ##produce multi-taper windows. don't need to normalize like in chronux (integral of the square of each taper = 1)
	tapers = tapers*np.sqrt(Fs)

	tapers2 = np.zeros([tapers.shape[0], tapers.shape[1], data.shape[1]]) ##add trial indices to tapers
	for i in range(tapers2.shape[2]):
		tapers2[:,:,i] = tapers

	fft_tapers = np.fft.fft(tapers, nfft, axis = 0) ##take the fft of the tapers

	H = np.zeros([fft_tapers.shape[0],fft_tapers.shape[1],numTraces],dtype = np.complex64) ##add trace/trial indices to fft_tapers
	for i in range(numTraces):
		H[:,:,i] = fft_tapers

	Nsp = np.sum(data, axis = 0) ##number of spikes in each trial
	Msp = Nsp/N ##mean rate for each channel
	meansp = np.zeros([Msp.shape[0],tapers.shape[1],H.shape[0]]) ##add taper and freq indices to meansp
	for i in range(meansp.shape[1]):
		for j in range(meansp.shape[2]):
			meansp[:,i,j] = Msp
	meansp = np.transpose(meansp)
	data2 = np.zeros([data.shape[0],data.shape[1],tapers.shape[1]]) ##add taper indices to data
	for i in range(data2.shape[2]):
		data2[:,:,i] = data
	data2 = np.transpose(data2,(0,2,1)) ##get data into the same dimensions as H
	data_proj = data2*tapers2 ##multiply data by tapers
	J = np.fft.fft(data_proj,nfft,axis = 0) ##fft of projected data
	##account for spike rate
	J = J-H*meansp
	#print '...Done!'
	return J, Msp, Nsp


"""
A function that takes the FFT of one or a set of LfP traces individually.
The data parameter is an array of lfp traces in the format samples x trials,
Fs is the sample rate.
"""

def fft_lfp(data, Fs = 1000):
	#print "Calculating multi-taper fft of lfp data..."
	data = data.squeeze() ##get rid of singleton dimensions for the next step
	if len(data.shape) > 1: ##if there is more than one trace, set N and numTraces appropriately
		N = data.shape[0]
		numTraces = data.shape[1]
	else: ##if there is only 1, set N and numTraces
		N = len(data)
		numTraces = 1
		##add a singleton dimension to make data "2d"
		data = data[:,None]

	nfft = 2**spec.nextpow2(N) ##next power of 2 from length of data (makes computation faster)
	tapers, eigs = spec.dpss(N, 3, 5) ##produce multi-taper windows. don't need to normalize like in chronux (integral of the square of each taper = 1)
	tapers = tapers*np.sqrt(Fs)

	tapers2 = np.zeros([tapers.shape[0], tapers.shape[1], numTraces]) ##add trial indices to tapers
	for i in range(tapers2.shape[2]):
		tapers2[:,:,i] = tapers

	H = np.fft.fft(tapers,nfft,axis = 0) ##fouier transform of the tapers
	Nsp = data.sum(axis = 0) ##number of spikes in each trial
	Msp = Nsp/N ##mean rate for each channel

	data2 = np.zeros([N, numTraces, tapers.shape[1]]) ##add taper indices to data
	for i in range(data2.shape[2]):
		data2[:,:,i] = data

	data2 = np.transpose(data2,(0,2,1)) ##get data into the same dimensions as tapers2
	data_proj = data2*tapers2 ##multiply data by tapers
	J = np.fft.fft(data_proj,nfft, axis = 0)/Fs ##fft of projected data
	J
	#print'...Done!'

	return J

"""
a helper function for the spectral analyses
that returns the frequency grid given the params
"""
def getfgrid(Fs, nfft, fpass):
	
	##a helper function to return indices
	def indices(a, func):
		return [i for (i, val) in enumerate(a) if func(val)]
	
	Fs = float(Fs)
	df = Fs/nfft
	f = np.linspace(0,Fs,Fs/df+1)
	f = f[0:nfft]
	findx = indices(f, lambda x: x >= fpass[0] and x<=fpass[1])
	f = f[findx]

	return f, findx

"""
A function that returns the z-scored version of a binary spike train array.
The function uses a gaussian window to convolve the raw spiketrain
and then computes the zscore. Inputs are an array of binary spike trains
with the format samples x trials, and the sigma value to use for the gaussian
convolution.
"""
def zscored_fr(data, sigma):
	if len(data.shape) < 2:
		N = data.size
		data = data.reshape(N, 1)
	##convolve the data using a gaussian window
	result = gauss_convolve(data, sigma)
	if len(result.shape) < 2:
		result = result.reshape(N,1)
	##run through each array and compute the zscore
	for arr in range(result.shape[1]):
		result[:,arr] = zscore(result[:,arr])
	result = np.nan_to_num(result)
	return result.squeeze()


"""
takes as an argument an array of event times and
the duration of recording in SECONDS
and converts it to a binary numpy array
	"""
def event_times_to_binary(eventTrain, duration):
	##get recodring length
	duration = float((duration)*1000)
	##set the number of bins as the next multiple of 100 of the recoding duration;
	#this value will be equivalent to the number of milliseconds in the recording (plus a bit more)
	numBins = np.ceil(duration/100)*100

	##do a little song and dance to ge the spike train times into a binary format
	bTrain = np.histogram(eventTrain, bins = numBins, range =(0,numBins))
	bTrain = bTrain[0].astype(bool).astype(int)

	return bTrain

"""
A simple function to detect peaks in a signal.
Inputs are the data (1-D array), the peak threshold value,
and the amount of time after the peak to ignore above-threshold
values.
"""
def peak_detect(data, thresh, post_ignore):
	##make sure the data is 1-D
	data.squeeze()
	##figure out how many data points you have
	total_pts = data.shape[0]
	##a count parameter
	count = 0
	##a list of peak indexes
	peaks = []
	##run through the data and grab the indexes
	while count <= total_pts-1:
		if data[count] > thresh:
			peaks.append(count)
			count += post_ignore
		else:
		 count += 1
	return peaks




"""
A function to bin arrays of data. The original purpose of this function
was for getting the value of a binary spike array over 50-ms bins (in other words,
	the ensemble value given a spike train)
Inputs are the data and the bin size. Output is the binned data.
****ASSUMES THAT INPUT IS 1-D**********
"""
def bin_data(data, bin_size):
	##determine the number of bins that will fit in the data
	numBins = (data.size-1)/bin_size
	##allocate memory
	result = np.zeros(numBins)
	##bin the data!
	for i in range(numBins):
		result[i] = data[i*50:(i+1)*50].sum()
	return result

"""
A function to get the reconstructed cursor value given the enseble spike data.
Inputs are:
	-e1: a list containing the binary spike arrays for e1_units
	-e2: a list containing the binary spike arrays for e2_units
	-bin_size: the cursor sample rate used in the experiment
Output is an array of timestamps corresponding to the other outupt, the array of cursor vals.
"""
def get_cursor_val(e1, e2, bin_size):
	##add the ensemble arrays together
	e1_vals = np.zeros(e1[0].size)
	for i in range(len(e1)):
		e1_vals += e1[i]
	e2_vals = np.zeros(e2[0].size)
	for j in range(len(e2)):
		e2_vals += e2[j]
	e1_vals = bin_data(e1_vals,bin_size)
	e2_vals = bin_data(e2_vals,bin_size)
	cval = e1_vals-e2_vals
	return cval


"""
A program to truncate two lists of arrays so they're all the length of the
shortest one.
"""
def truncate_arrs(e1_arrays, e2_arrays):
	## get the number of sessions we're working with ##should be the same for both e1 and e2
	num_sessions = len(e1_arrays)
	##determine the shortest duration recording to fit everything into the same size array
	##start with the first one
	shortest_duration = e1_arrays[0].size
	for i in range(num_sessions):
		if e1_arrays[i].size < shortest_duration:
			shortest_duration = e1_arrays[i].size
	##allocate memory for arrays
	e1s = np.zeros((len(e1_arrays), shortest_duration))
	e2s = np.zeros((len(e1_arrays), shortest_duration))
	##fill the arrays with the resized data
	for i in range(num_sessions):
		e1s[i,:] = e1_arrays[i][:shortest_duration].squeeze()
		e2s[i,:] = e2_arrays[i][:shortest_duration].squeeze()
	return e1s, e2s


def get_cursor_hist_vals(e1_arrs, e2_arrs, bin_size):
	e1s, e2s = truncate_arrs(e1_arrs, e2_arrs)
	e1s = e1s.mean(axis = 0)
	e2s = e2s.mean(axis = 0)
	e1s = bin_data(e1s, bin_size)
	e2s = bin_data(e2s, bin_size)
	cval = e1s-e2s
	return cval


"""A function that takes in windowed data in the format
n units x m samples x p trials and returns the pairwise coherence
of each pair between 2 data groups averaged over trials."""
def get_pairwise_coherence(data1, data2, Fs = 1000.0, fpass = [0,200]):
	##determine the length of the coherence data that will be returned
	N = data1.shape[1]
	nfft = 2**spec.nextpow2(N)
	f, findx = getfgrid(Fs, nfft, fpass)
	##allocate memory
	all_coh = np.zeros((data1.shape[0]*data2.shape[0],f.size))
	##take the pairwise coherence and store it in the array
	for i in range(data1.shape[0]):
		for j in range(data2.shape[0]):
		   all_coh[data2.shape[0]*i+j, :],f = spike_spike_coherence(data1[i,:,:], data2[j,:,:], Fs = Fs, fpass = fpass, trialave = True)
 
	return all_coh

def get_pairwise_cohgram(spikes, lfps, Fs = 1000.0, fpass = [0,100], movingwin = [0.5, 0.05]):
	N = spikes.shape[1] ##length of segments
	Nwin = int(round(movingwin[0]*Fs)) ##window size in samples
	Nstep = int(round(movingwin[1]*Fs)) ##step size in samples
	nfft = (2**spec.nextpow2(Nwin)) ##the nfft length for the given window
	winstart = np.arange(0,N-Nwin,Nstep) ##array of all the window starting values
	nw = winstart.shape[0] ##number of total windows

	##get the frequency grid values based on user input
	f,findx = getfgrid(Fs,nfft,fpass)
	Nf = len(f)
	#container for the coherence data
	C = np.zeros((spikes.shape[0]*lfps.shape[0],nw,Nf))
	for i in range(spikes.shape[0]):
		for j in range(lfps.shape[0]):
		   C[lfps.shape[0]*i+j, :,:],phi, S12, S1, S2, t, f, zerosp, confc, phistd, Cerr = spike_field_cohgram(spikes[i,:,:], lfps[j,:,:], movingwin = movingwin, Fs = Fs, fpass = fpass)
	return C

def compare_pairwise_coherence(t1_data1, t1_names1, t1_data2, t1_names2, 
	t2_data1, t2_names1, t2_data2, t2_names2, Fs = 1000.0, fpass = [0,100], 
	sig = 0.05, band = [2,4]):
	N = t1_data1.shape[1]
	nfft = 2**spec.nextpow2(N)
	##make sure that the order of the units in both groups is the same
	if t1_names1 != t2_names1 or t1_names2 != t2_names2:
		print("Unit order mismatch!!!")
	## a container of significant pairs
	sig_t1_u1 = None
	sig_t1_u2 = None
	sig_t2_u1 = None
	sig_t2_u2 = None
	##get the spike-spike coherence for each pair for T1 and T2
	for i in range(t1_data1.shape[0]): ##of units in data1
		for j in range(t1_data2.shape[0]): ##number of units in data2
			##get the coherence; DON'T average over trials
			C1, t, fgrid = spike_spike_cohgram(t1_data1[i,:,:], t1_data2[j,:,:], 
				[0.5, 0.05], Fs = Fs, fpass = fpass)
			#C2, fgrid = spike_spike_coherence(t2_data1[i,:,:], t2_data2[j,:,:], 
			 #   Fs = Fs, fpass = fpass, trialave = False)
			##make comparisons over all of the specified ranges
			timerange = (t > band[0]) * (t < band[1])
			idx = np.where(timerange)[0]
			##get the mean value of the coherence in the time range of interest
			interest_mean = C1[idx,:].mean(axis = 0)
			other = np.ma.array(C1, mask = False)
			other.mask[idx,:] = True
			other_mean = other.mean(axis = 0)
			# for r in range(len(bands)):
			#     ##get the indices of values lying in the current band of interest
			#     f, findx = getfgrid(Fs, nfft, bands[r])
			#     ##calculate the mean coherence across this band for 
			#     ##each trial in both conditions
			#     c1 = C1[findx, :].mean(axis = 0)
			#     c2 = C2[findx, :].mean(axis = 0)
			#     ##decide if there is a significant difference between the
				##coherence in this band across the two conditions
			t, p = stats.ttest_rel(interest_mean, other_mean)
			if p <= sig and interest_mean.mean() > other_mean.mean():
				print("Found something significant according to SS2")
				if sig_t1_u1 is not None:
					sig_t1_u1 = np.hstack((sig_t1_u1,t1_data1[i,:,:]))
					sig_t1_u2 = np.hstack((sig_t1_u2,t1_data2[j,:,:]))
					sig_t2_u1 = np.hstack((sig_t2_u1,t2_data1[i,:,:]))
					sig_t2_u2 = np.hstack((sig_t2_u2,t2_data2[j,:,:]))
				else:
					sig_t1_u1 = t1_data1[i,:,:]
					sig_t1_u2 = t1_data2[j,:,:]
					sig_t2_u1 = t2_data1[i,:,:]
					sig_t2_u2 = t2_data2[j,:,:]
				print("Successfully added data to SS2 results")
	return sig_t1_u1, sig_t1_u2, sig_t2_u1, sig_t2_u2


def cross_corr(p1,p2,winrad,dt):
	"""
	Compute the cross correlation function of processes p1 and p2
	dt is the binsize used in the representation of p1 and p2
	winrad is the window radius over which to compute the CCG
	H is the cross correlation function and X is the time domain
	of the function (X=-winrad:dt:winrad).
	H,X=cross_corr(p1,p2,winrad,dt)
	"""
	
	n1=p1.size
	n2=p2.size
	if n1!=n2:
		print('Inputs have different sizes.')
	
	n=n1    # n is the number of time units in each process (the size of the vectors)
	T=n*dt  # T is the length of the spike train in the correct units
	
	
	# Change the units of winrad to the binsize
	winrad=int(np.floor(winrad/dt))
	
	
	# Estimate the rates of each process
	n1=np.sum(p1)
	n2=np.sum(p2)
	
	
	# Initialize the vector that will hold the correlation function
	H=np.zeros(2*winrad)
	
	# Get the spike times of p2
	s=mp.mlab.find(p1);
	
	# Fill the un-normalized correlation function vector
	p2_temp=np.concatenate((np.zeros(winrad), p2, np.zeros(winrad)))      # stick some zeros on either end of p1 for easier indexing
	for i in range(s.size):
		H=H+p2_temp[s[i]:s[i]+2*winrad]   #p1_temp(ti:ti+2*winrad) is really just p1(ti-winrad:ti+winrad) with zeros for the non-existent entries
	
	
	#Normalize to get the correlation function
	H=(H-n1*n2/n)/(T*dt)
	
	
	# Return the time domain 
	X=np.linspace(-winrad*dt,winrad*dt,2*winrad/dt)
	
	return H,X

def cov_coeff(p1,p2,tau,dt):
	"""
	Compute the covariance coefficient between the spike trains
	p1 and p2 by taking the integral of the crosscorrelation function
	over a window of radius tau and correcting.
	Multiply this number by tau to get the covariance of the
	spike count over an interval of size tau.
	c=cov_coeff(p1,p2,tau,dt)
	dt is the binsize
	NOTE: p1 and p2 should be the same length
	"""
	
	# T is the length of p1 in the correct units
	n=p1.size
	T=n*dt
	
	# Define the correction term 'function'
	correctionterm=1-abs(np.linspace(-tau,tau,2*tau/dt))/tau
	
	# Calculate the correlation function
	cc12,x=cross_corr(p1,p2,tau,dt)
	
	# Integrate the correlation function
	# multiplying by the correctino term
	c=np.sum(correctionterm*cc12)*dt
	
	return c
	
def var_coeff(p,tau,dt):
	"""
	Compute the uncorrected variance coefficient of a spike train p
	by taking the integral of the autocorrelation function
	over a window of radius tau and correcting.
	Multiply this number by tau to get the variance of the 
	spike count over an interval of size tau.
	v=var_coeff(p,tau,dt)
	"""
	
	#The variance coefficient is just the
	#cov coefficient of p with itself.
	v=cov_coeff(p,p,tau,dt)
	
	return v
	
def corr_coeff(p1,p2,tau,dt):
	"""
	Compute the correlation coefficient between spike trains
	p1 and p2 over a window of size tau;
	dt is the binsize
	cc=corr_coeff(p1,p2,tau,dt)
	"""
	
	#The correlation coefficient can be computed using
	# the covariance and variance coefficients
	cc=cov_coeff(p1,p2,tau,dt)/np.sqrt(var_coeff(p1,tau,dt)*var_coeff(p2,tau,dt))
	
	return cc

def window_corr(data1, data2, window, tau, dt):
	"""
	This functions takes in two binary spike train arrays of the same size
	and returns a vector representing correlation over time, generated by calculating
	the correlation coefficient over a sliding window.
	"""
	N = data1.size
	Nwin = int(round(window[0])) ##window size in samples
	Nstep = int(round(window[1])) ##step size in samples
	winstart = np.arange(0,N-Nwin,Nstep) ##array of all the window starting values
	nw = winstart.shape[0] ##number of total windows
	result = np.zeros(nw)
	for n in range(nw):
		indx = np.arange(winstart[n],winstart[n]+Nwin) ##index values to take from data based on current window
		datawin1 = data1[indx]
		datawin2 = data2[indx] ##current data window for all lfp segments
		#calculate rate for the given window
		result[n] = corr_coeff(datawin1, datawin2, tau, dt)
	return result

"""
A function to compute the spike triggered average. Input array
should be in the format samples x trials. lfp_win is in the format [ms,ms],
and describes the window around each spike to take the lfp. trialave = True
averages over trials (returning a single trace), while trialave = False
returns the STA for each trial.
"""
def STA(spikes, lfps, lfp_win, trialave = True):
	##create a container for STA traces (samples x trials)
	avg_traces = np.zeros((lfp_win[0]+lfp_win[1], spikes.shape[1]))
	##run through each trial and compute the STA
	for i in range(spikes.shape[1]):
		##get the indices where a spike occurred
		spike_idx = list(np.nonzero(spikes[:,i])[0])
		z_lfp = zscore(lfps[:,i])
		##get rid of indices that conflict with the bounds
		#of STA window
		spike_idx = [x for x in spike_idx if x > lfp_win[0] and x < spikes.shape[0]-lfp_win[1]]
		if spike_idx != []:
			##make a container to store individual traces
			traces = np.zeros((lfp_win[0]+lfp_win[1], len(spike_idx)))
			##get the lfp trace around each spike in the trial
			for j, index in enumerate(spike_idx):
				traces[:,j] = z_lfp[index-lfp_win[0]:index+lfp_win[1]] 
			##add the average trace to the outer container
			avg_traces[:,i] = masked_avg(traces, axis = 1)
	if trialave:
		avg_traces = masked_avg(avg_traces, axis =1)
	return avg_traces

"""
a short function to make a masked array
so that when you take the mean, you can ignore
the NANs
"""
def masked_avg(data_in,axis = 1):
	mdata = np.ma.masked_array(data_in, np.isnan(data_in))
	mean_data = np.mean(mdata, axis = axis)
	return mean_data    

"""
returns the multi-taper spectrum of an FFT
"""
def mtspectrum(data, Fs = 1000.0, fpass = [0,100], trialave = True, err = None, sigType = 'lfp'):
	N = data.shape[0]
	if len(data.shape) <= 1 or data.shape[1] == 1:
		trialave = False
	nfft = 2**spec.nextpow2(N)
	f, findx = getfgrid(Fs, nfft, fpass)
	if sigType == 'lfp':
		fft = fft_lfp(data)
	elif sigType == 'spikes':
		fft, Msp, Nsp = fft_spikes(data)
	J = fft[findx,:,:]
	S = np.squeeze(np.mean(J.conj()*J, axis = 1))
	if trialave:
		S = np.nanmean(S, axis=1)
	if err is not None:
		Serr = specerr(S, J, err, trialave)
	else: 
		if trialave:
			Serr = np.zeros((2, S.shape[0]))
		else:
			try: 
				Serr = np.zeros((2, S.shape[0], S.shape[1]))
			except IndexError:
				Serr = np.zeros((2, S.shape[0], 1))

	return S, f, Serr

"""
function to compute the lower and upper confidence intervals
on the spectrum.
Inputs:
	-S: spectrum
	-J: tapered fourier transforms
	-err: [errtype, p] 
	-trialave
	-numsp: number of spikes in each channel
"""
def specerr(S, J, err, trialave, numsp = None):
	# N = data.shape[0]
	# nfft = 2**spec.nextpow2(N)
	# f, findx = getfgrid(Fs, nfft, fpass)
	# J = fft_lfp(data, Fs)
	# J = J[findx,:,:]
	# S = np.squeeze(np.mean(J.conj()*J, axis = 1))
	# if trialave:
	#     S = np.squeeze(np.mean(S, axis = 1))

	nf, K, C = J.shape
	errchk = err[0]
	p = err[1]
	pp = 1-p/2.0
	qq = 1-pp

	if trialave:
		dim = K*C
		C = 1
		dof = 2*dim
		if numsp is not None:
			dof = np.fix(1/(1/dof+1/(2*np.sum(numsp))));
		J = J.reshape(nf, dim)
	else:
		dim = K
		dof = 2*dim*np.ones((C))
		for ch in range(C):
			if numsp is not None:
				dof[ch] = np.fix(1/(1/dof+1/(2*numsp(ch))))
	Serr = np.zeros((2, nf, C)).squeeze()
	if errchk == 1:
		Qp = stats.chi2.ppf(pp, dof) ###Need to define chi2inv
		Qq = stats.chi2.ppf(qq, dof)
		try:
			Serr[0,:,:] = dof*np.ones((nf))*S/Qp*np.ones((nf))
			Serr[1,:,:] = dof*np.ones((nf))*S/Qq*np.ones((nf))
		except IndexError:
			Serr[0,:] = dof*np.ones((nf))*S/Qp*np.ones((nf))
			Serr[1,:] = dof*np.ones((nf))*S/Qq*np.ones((nf))

	elif errchk == 2:
		tcrit = stats.t.ppf(pp, dim-1)
		
		if trialave:
			Sjk = np.zeros((dim, J.shape[0]))
		else:
			Sjk = np.zeros((dim, J.shape[0], J.shape[-1]))
		#pbar = ProgressBar(maxval = dim).start()
		pcount = 0
		for k in range(dim):
			indices = np.setdiff1d(np.arange(0,dim), np.asarray(k))
			if not trialave:
				Jjk = J[:,indices,:]
				eJjk = np.squeeze(np.sum(Jjk*Jjk.conj(), axis = 1))
				Sjk[k,:,:] = eJjk/(dim-1)
			else:
				Jjk = J[:,indices]
				eJjk = np.squeeze(np.sum(Jjk*Jjk.conj(), axis = 1))
				Sjk[k, :] = eJjk/(dim-1)
			#pbar.update(pcount+1)
			pcount+=1
		#pbar.finish()
		sigma = np.sqrt(dim-1)*np.squeeze(np.std(np.log(Sjk), axis = 0))
		conf = np.squeeze(np.multiply(np.kron(np.ones((nf)),tcrit),sigma))
		try:
			Serr[0,:,:] = S*np.exp(-conf)
			Serr[1,:,:] = S*np.exp(conf)
		except IndexError:
			Serr[0,:] = S*np.exp(-conf)
			Serr[1,:] = S*np.exp(conf) 
	Serr = Serr.squeeze()
	return Serr


def coherr (C, J1, J2, err, trialave):
	nf, K, Ch = J1.shape
	errchk = err[0]
	p = err[1]
	pp = 1-p/2.0
	##find the number of degrees of freedom
	if trialave:
		dim = K*Ch
		dof = 2*dim
		Ch = 1
		J1 = J1.reshape(nf, dim)
		J2 = J2.reshape(nf, dim)
	else:
		dim = K
		dof = 2*dim*np.ones((Ch))

	##variance of the phase:
	if dof <= 2:
		confC = 1
	else:
		df = 1/((dof/2.0)-1)
		confC = np.sqrt(1-p**df)

	##phase standard deviation
	if errchk == 1:
		totnum = nf*Ch
		phistd = np.zeros((totnum))
		CC = C.reshape(totnum)
		indx = mp.mlab.find(np.abs(CC-1)>=1e-16)
		dof = np.kron(np.ones((nf)),dof)
		dof.reshape(totnum)
		phistd[indx] = np.sqrt((2/dof[indx]*(1/(C[indx]**2)-1)))
		phistd = phistd.reshape(nf,Ch)
		Cerr = 0
	elif errchk == 2:
		#print "computing jacknife error"
		tcrit = stats.t.ppf(pp, dof-1)
		if trialave:
			atanhCxyk = np.zeros((dim, J1.shape[0]))
			phasefactoryxyk = np.zeros((dim, J1.shape[0]))
		else:
			atanhCxyk = np.zeros((dim, J1.shape[0], J1.shape[1]))
			phasefactoryxyk = np.zeros((dim, J1.shape[0], J1.shape[1]))
		pbar = ProgressBar(maxval = dim).start()
		pcount = 0
		for k in range(dim):
			indxk = np.setdiff1d(np.arange(0,dim),np.asarray(k))
			try:
				J1k = J1[:,indxk,:]
				J2k = J2[:,indxk,:]
			except IndexError:
				J1k = J1[:,indxk]
				J2k = J2[:,indxk]     
			eJ1k = np.squeeze(np.sum(J1k*J1k.conj(), axis = 1))
			eJ2k = np.squeeze(np.sum(J2k*J2k.conj(), axis = 1))
			eJ12k = np.squeeze(np.sum(J1k.conj()*J2k, axis = 1))
			Cxyk = eJ12k/np.sqrt(eJ1k*eJ2k)
			absCxyk = np.abs(Cxyk)
			try:
				atanhCxyk[k,:,:] = np.sqrt(2*dim-2)*np.arctanh(absCxyk)
				phasefactoryxyk[k,:,:] = Cxyk/absCxyk
			except IndexError:
				atanhCxyk[k,:] = np.sqrt(2*dim-2)*np.arctanh(absCxyk)
				phasefactoryxyk[k,:] = Cxyk/absCxyk
			pbar.update(pcount+1)
			pcount+=1
		pbar.finish()
		atanhC = np.sqrt(2*dim-2)*np.arctanh(C)
		sigma12 = np.sqrt(dim-1)*np.squeeze(np.std(atanhCxyk,axis = 0))
		if trialave:
			Cerr = np.zeros((2, J1.shape[0]))
		else:
			Cerr = np.zeros((2, J1.shape[0],J1.shape[1]))
		if Ch == 1:
			sigma12 = sigma12.T
		Cu = atanhC+tcrit*(np.ones((nf)))*sigma12
		Cl = atanhC-tcrit*(np.ones((nf)))*sigma12
		try:
			Cerr[0,:,:] = np.maximum(np.tanh(Cl/np.sqrt(2*dim-2)),0)
			Cerr[1,:,:] = np.tanh(Cu/np.sqrt(2*dim-2))
		except IndexError:
			Cerr[0,:] = np.maximum(np.tanh(Cl/np.sqrt(2*dim-2)),0)
			Cerr[1,:] = np.tanh(Cu/np.sqrt(2*dim-2))   
		phistd = np.sqrt((2*dim-2)*(1-np.abs(np.squeeze(np.mean(phasefactoryxyk, axis = 0)))))
		if trialave:
			phistd = phistd.T
	return confC, phistd, Cerr

def calc_unit_snr(waveforms):
	"""
	this function calculates the SNR of sorted units using the method from
	koralek 2013 A/(2*SDnoise), where A is Peak-to-peak voltage of the mean waveform,
	and SDnoise is the standard devation of the residuals from each waveform after
	the mean waveform has been subtracted.

	arguments are an array of waveforms- dimenstions spike x time
	(should already been in this format from plexon file)
	"""
	#get rid of singleton dimensions
	waveforms = np.asarray(waveforms).squeeze()

	#calculate the mean waveform
	mean_wf = waveforms.mean(axis = 0)

	#calculate peak-to-peak value of the mean WF
	a = abs(mean_wf.max() - mean_wf.min())

	#subtract the mean wf from the individual traces to get the residuals
	residuals = waveforms - mean_wf

	#get the standard deviation of the residuals
	sd_noise = residuals.std()

	#calculate the snr
	snr = a/(2*sd_noise)

	return snr
