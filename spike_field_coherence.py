##a script that calculates cross-condition spike-field coherence 
##using the rate-correcting algorithm described in Aoi et al, 2015

import h5py
import scipy.interpolate 
import spectrum as spec
import numpy as np
import matplotlib as mp


##a function to calculate the spike-field coherence; for use 
##in multiprocessing contexts
def mp_sfc(args):
	spikes = args[0]
	lfp = args[1]
	##interpolate the LFP around the spikes to eliminate bleed-thru
	assert lfp.shape == spikes.shape
	for i in range(spikes.shape[1]):
		if spikes[:,i].sum() > 0:
			lfp[:,i] = interp_LFP(lfp[:,i],spikes[:,i])
	win_secs = (spikes.shape[0]/1000.0)/2.0 ##half the full time window in seconds
	comparison_spikes = spikes[0:spikes.shape[0]/2,:] ##spike rates before the start of target acquisition
	target_spikes = spikes[spikes.shape[0]/2:,:] #spike rates during target acquisition
	target_rate_est = target_spikes.sum()/win_secs/target_spikes.shape[1] ##target spike rate est
	comparison_rate_est = comparison_spikes.sum()/win_secs/comparison_spikes.shape[1] ##comparitson spike rate est
	C,phi,S12,S1,S2,t,f,zerosp,confc,phistd,Cerr = spike_field_cohgram(spikes,lfp,
		comparison_rate_est,target_rate_est,[0.5,0.05],Fs=1000.0,fpass=[0,100],err=None)
	return C

##a function to calculate the spike-field coherence; for use 
##in multiprocessing contexts
def mp_sf_coherence(args):
	spikes = args[0]
	lfp = args[1]
	##interpolate the LFP around the spikes to eliminate bleed-thru
	assert lfp.shape == spikes.shape
	for i in range(spikes.shape[1]):
		if spikes[:,i].sum() > 0:
			lfp[:,i] = interp_LFP(lfp[:,i],spikes[:,i])
	win_secs = (spikes.shape[0]/1000.0)/2.0 ##half the full time window in seconds
	comparison_spikes = spikes[0:spikes.shape[0]/2,:] ##spike rates before the start of target acquisition
	target_spikes = spikes[spikes.shape[0]/2:,:] #spike rates during target acquisition
	target_lfp = lfp[lfp.shape[0]/2:,:]
	target_rate_est = target_spikes.sum()/win_secs/target_spikes.shape[1] ##target spike rate est
	comparison_rate_est = comparison_spikes.sum()/win_secs/comparison_spikes.shape[1] ##comparitson spike rate est
	C,phi,S12,S1,S2,f,zerosp,confC,phistd,Cerr = spike_field_coherence(target_spikes,target_lfp, ##only computing for target times!!!
		comparison_rate_est,target_rate_est,Fs=1000.0,fpass=[0,100],trialave=True,err=[1,0.05])
	return [C, confC]

"""
this function grabs data from an hdf5 database, and parses it 
into time-locked snippets from two conditions.
Params:
-f_in: address of the HDF5 database
-animal: code name of the animal to get data from
-session: name of the session to get data from
-unit_group: name of the unit group to analyze
-lfp_group: name of the lfp group to analyze
-target: event name to time-lock results to
-window: period of time relative to the target analyze data from in ms; format [-3000,1000]
-offset_mean: mean value of the window center to offset the control condition from the target timestamp
"""
def calc_coherence(f_in, animal, session, spike_units, pre_win, post_win, lfp_units = None, 
	target = "t1", offset_mean = 5000):
	f = h5py.File(f_in, 'r')
	##time window in seconds
	win_secs = (pre_win + post_win)/1000.0
	##probably want to do things on a per-unit basis, so figure out how many units we have
	unit_list = [unit for unit in f[animal][session][spike_units].keys() if not unit.endswith("_wf")]
	##get a list of the LFP channels that we will be using. If LFP group is None, use the LFP
	##from the spike channels. 
	if lfp_units is not None:
		lfp_list = [unit for unit in f[animal][session][lfp_units].keys() if not unit.endswith("_wf")]
	else:
		lfp_list = unit_list
	##also grab the list of event times to lock to 
	target_ts = np.asarray(f[animal][session]['event_arrays'][target])
	##generate comparison trial timestamps by selecting random values around the target ts
	##as specified by the offset_mean parameter.
	comparison_ts  = np.zeros(target_ts.shape)
	for i in range(comparison_ts.size):
		comparison_ts[i] = target_ts[i] + np.random.randn()*1000+offset_mean
	##generate spike-lfp paired datasets for each timestamp in the target and comparison arrays
	C1 = []
	C2 = []

	##first get the full data arrays for this session
	for unit in unit_list:
		spike_array = np.asarray(f[animal][session][spike_units][unit][0])
		if lfp_units == None:
			##get the lfp array from the same channel
			lfp_array = np.asarray(f[animal][session][spike_units][unit][1])
			##spike waveforws will bleed through; so need to interpolate the LFP to account for this
			lfp_array = interp_LFP(lfp_array, spike_array)
		else:
			lfp_array = np.asarray(f[animal][session][lfp_units][lfp_list][np.random.randint(0,len(lfp_list))][1])
		##get the matched spike-lfp data windows; format is samples x trials
		target_spikes = get_data_window(target_ts, pre_win, post_win, spike_array)
		target_lfp = get_data_window(target_ts, pre_win, post_win, lfp_array)
		comparison_spikes = get_data_window(comparison_ts, pre_win, post_win, spike_array)
		comparison_lfp = get_data_window(comparison_ts, pre_win, post_win, lfp_array)
		target_rate_est = target_spikes.sum()/win_secs/target_spikes.shape[1]
		comparison_rate_est = comparison_spikes.sum()/win_secs/comparison_spikes.shape[1]
		C_t, phi_t, S12_t, S1_t, S2_t, fr_t, zerosp_t, confC_t, phistd_t, Cerr_t = spike_field_coherence(
			target_spikes, target_lfp, comparison_rate_est, target_rate_est, Fs = 1000.0, fpass = [0,100], trialave = True, err = None)
		C_c, phi_c, S12_c, S1_c, S2_c, fr_c, zerosp_c, confC_c, phistd_c, Cerr_c = spike_field_coherence(
			comparison_spikes, comparison_lfp, target_rate_est, comparison_rate_est, Fs = 1000.0, fpass = [0,100], trialave = True, err = None)
		C1.append(C_t)
		C2.append(C_c)
	return np.asarray(C1), np.asarray(C2)

def get_coherence_by_animal(f_in, animal, session_list, spike_units, pre_win, post_win, lfp_units = None, 
	target = "t1", offset_mean = 5000):
	C1 = []
	C2 = []
	for session in session_list:
		c1, c2 = calc_coherence(f_in, animal, session, spike_units, pre_win, post_win, lfp_units = lfp_units, 
	target = target, offset_mean = offset_mean)
		C1.append(c1)
		C2.append(c2)
	return np.asarray(C1), np.asarray(C2)


def interp_LFP(lfp_array, spike_array, interp_win = 200, replace_win = 5, interp_method = "linear"):
	##find the index values of the spike times
	spike_idx = np.where(spike_array == 1)[0]
	for idx in spike_idx:
		begsmp = idx-replace_win
		endsmp = idx+replace_win
		beg_interp = begsmp-interp_win
		end_interp = endsmp+interp_win
		if beg_interp > 1 and end_interp < lfp_array.size:
			xall = np.arange(beg_interp, end_interp)
			x = np.hstack((np.arange(beg_interp, begsmp), np.arange(endsmp, end_interp)))
			y = lfp_array[x]
			f = scipy.interpolate.interp1d(x,y, kind = interp_method)
			lfp_array[xall] = f(xall)
		else:
			pass
	return lfp_array


"""a useful function for getting a data window centered around a given index.
	Inputs:
	-Centers: the indices for taking windows arround (1-D np array)
	-pre-win: pre-center window length
	-post-win: post-center window length
	-data: full data trace(s) in the shape (y-axis, time axis)
"""
def get_data_window(centers, pre_win, post_win, data):
	centers = list(centers.astype(np.int64))
	data = np.squeeze(data)
	N = data.size
	to_remove = []
	for j in range(len(centers)):
		if centers[j] <= pre_win or centers[j] + post_win >= N:
			to_remove.append(j)
	if len(to_remove) != 0:
		centers = [i for j, i in enumerate(centers) if j not in to_remove]
	traces = np.zeros((pre_win+post_win, len(centers)))
		##the actual windowing functionality:
	for n in range(len(centers)):
		traces[:,n] = data[centers[n]-pre_win:centers[n]+post_win]
	return traces



def spike_field_coherence(spikes, lfp, StarRate, rate, Fs = 1000.0, fpass = [0,100], trialave = True, err = None):
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
	##calculate coherence using the spike rate correction from Aoi et al, 2015
	c0 = StarRate/rate
	C12 = S12/np.sqrt(S1*S2)/np.sqrt((1-c0)*rate/S2/c0+1)
	#return the absolute value of the coherence (and the freq grid values)
	C = abs(C12)
	phi = np.angle(C12)
	if err is not None:
		confC, phistd, Cerr = coherr(C, J1, J2, err, trialave)
	else:
		confC = 0; phistd = 0; Cerr = 0

	return C, phi, S12, S1, S2, f, zerosp, confC, phistd, Cerr


def spike_field_cohgram(spikes, lfp, StarRate, Rate, movingwin, Fs = 1000.0, fpass = [0, 100], err = None):
	"""
	Calculate the coherence values over time by using a sliding window. Inputs are
	an array of binary spike trains in the format samples x trials, an array
	of ****MATCHING ORDERED**** lfp signals in the same format, a 
	movingwin parameter in the format [window, winstep], sample rate in Hz and 
	the frequency window to restrict analysis to. 
	"""
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

	for n in range(nw):
		indx = np.arange(winstart[n],winstart[n]+Nwin) ##index values to take from data based on current window
		spikeswin = spikes[indx,:] ##current data window for all lfp segments
		lfpwin = lfp[indx,:] ##current data window for all segments
		#calculate coherence for the given window
		c, ph, s12, s1, s2, f, zsp, confc, phie, cerr = spike_field_coherence(spikeswin,
			lfpwin, StarRate, Rate, Fs = Fs, fpass = fpass, trialave = True, err = err)
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

	C = C.squeeze(); S12 = S12.squeeze(); S1 = S1.squeeze(); S2 = S2.squeeze(); phi = phi.squeeze()
	zerosp = zerosp.squeeze()
	#calculate the time axis values
	winmid=winstart+round(Nwin/2)
	t=winmid/Fs

	return C, phi, S12, S1, S2, t, f, zerosp, confc, phistd, Cerr


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
	#account for spike rate
	J = J-H*meansp
	#print '...Done!'
	return J, Msp, Nsp

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