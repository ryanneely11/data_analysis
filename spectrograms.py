##spectrograms.py
##code for calculating time varying spectrums using the 
##muti-taper mwthod

import numpy as numpy
from progressbar import *
import matplotlib as mp
import scipy as sp
import spectrum as spec

def lfpSpecGram(data, window=[0.5,0.05], Fs = 1000.0, fpass = [0,100], err = None, 
	sigType = 'lfp', norm = True):
	"""
	Basically just a moving window average of the spectrum. 
	Input data is an array; samples x trials
	Window and inputs should be in sec: [win length, win step]
	fpass should be [fmin, fmax]. 
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