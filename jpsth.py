##a series of functions to calculate the joint-
#peristumulus time histogram

import numpy as np
import scipy as sp

##main function: JPSTH. Outputs a dictionary with all of the
##results. Inputs are arrays of binned spikes in the format
##trials x time and the width of the coincidence histogram.
def jpsth(n_1, n_2, coinWidth = 10.0, bin = 1):
	##if bins are bigger than 1 ms, bin the data
	lag = 50
	coinWidth = int(coinWidth)
	if bin > 1:
		n_1 = bin_data(n_1, bin)
		n_2 = bin_data(n_2, bin)
		lag = int(np.ceil(lag/bin))
		coinWidth = int(np.ceil((coinWidth*1.0)/bin))
	##caclulate the psth for each signal
	psth_1, psth_1_sd, psth_1_var = psth(n_1)
	psth_2, psth_2_sd, psth_2_var = psth(n_2)

	##calulate PSTH components (equations from Artsem et al, 1989):
	rawJPSTH = equation3(n_1, n_2)
	psthOuterProduct = np.outer(psth_1, psth_2)
	unnormalizedJPSTH = rawJPSTH - psthOuterProduct
	normalizer = np.outer(psth_1_sd, psth_2_sd)
	normalizedJPSTH = unnormalizedJPSTH/normalizer

	##account for any divide by zero errors by
	##replacing NaNs with zeros
	normalizedJPSTH = np.nan_to_num(normalizedJPSTH)

	##run JPSTH analyses
	xcorrHist = crossCorrelationHistogram(normalizedJPSTH)
	pstch = pstCoincidenceHistogram(normalizedJPSTH, coinWidth)
	covariogram, sigHigh, sigLow = covariogramBrody(n_1, n_2, psth_1, 
		psth_2, psth_1_var, psth_2_var)
	#sigPeakEndpoints = significantSpan(covariogram, sigHigh)
	#sigTroughEndpoints = significantSpan(-covariogram, -sigLow)

	##create the output dictionary
	results = {
	'psth_1':psth_1,
	'psth_2':psth_2,
	'normalizedJPSTH':normalizedJPSTH,
	'xcorrHist':xcorrHist,
	'pstch':pstch,
	'covariogram':covariogram,
	'sigLow':sigLow,
	'sigHigh':sigHigh,
	#'sigPeakEndpoints':sigPeakEndpoints,
	#'sigTroughEndpoints':sigTroughEndpoints
	}
	return results


##a function to compute the psth of a set of binned spike trains in the 
##format trials x time
def psth(spikes):
	if spikes.sum() == 0:
		raise ValueError("There are no spikes in the data!")
	else:
		psthMean = spikes.mean(axis = 0)
		psthStdDev = spikes.std(axis = 0)
		psthVariance = spikes.var(axis = 0)

		return psthMean, psthStdDev, psthVariance


##this is equation # 3 from Artsen et al (1989). It computes the
##raw JPSTH given arrays of spike trains in the format trials x time.
def equation3(spike_1, spike_2):
	rawJPSTH = np.zeros((spike_1.shape[1], spike_1.shape[1]))
	for u in range(spike_1.shape[1]):
		for v in range(spike_2.shape[1]):
			rawJPSTH[u,v] = np.mean(spike_1[:,u]*spike_2[:,v])
	return rawJPSTH


##this function calculates the cross correlation histogram at a given time
#lag given a JPSTH and a time lag as inputs.
def crossCorrelationHistogram(jpsth, lag = 50):
	xcorrHist = np.zeros((len(np.arange(-lag,lag))))
	for i in range(-lag,lag):
		xcorrHist[i+lag] = (np.sum(np.diag(jpsth, i)))/(len(jpsth)-abs(i))
	return xcorrHist

## this function calculates a coincidence histogram of a given width
##for a JPSTH
#***Not sure this will work***
def pstCoincidenceHistogram(jpsth, coinWidth = 10):
	widthVector = np.arange(-coinWidth, coinWidth)
	pstch = np.zeros(jpsth.shape[0])
	for i in widthVector:
		pad = abs(i)
		d = np.diag(jpsth, i)
		pstch[pad:] = pstch[pad:] + d
	return pstch


##this function calculates a covariogram of spike_1 and spike_2 at a given time lag
##p1 and p2 are the psth's for spike 1 and 2, and s1 and s2 are the psth variances for 
##spike 1 and spike 2.
def covariogramBrody(spike_1, spike_2, p1, p2, s1, s2, lag = 50):
	trials = spike_1.shape[0]
	trialLength = spike_1.shape[1]

	s1s2 = np.zeros(2*lag)
	p1s2 = np.zeros(2*lag)
	s1p2 = np.zeros(2*lag)
	crossCorr = np.zeros(2*lag)
	shuffleCorrector = np.zeros(2*lag)

	for i in range(2*lag):
		currentLag = i - lag
		if currentLag < 0:
			jVector = np.arange(currentLag,trialLength)
		else:
			jVector = np.arange(0, trialLength-currentLag)
		for j in jVector:
			crossCorr[i] = crossCorr[i] + np.mean(spike_1[:,j] * spike_2[:,j+currentLag])
			shuffleCorrector[i] = shuffleCorrector[i] + p1[j] * p2[j+currentLag]
			s1s2[i] = s1s2[i] + np.dot(np.square(s1[j]), np.square(s2[j+currentLag]))
			p1s2[i] = p1s2[i] + np.dot(np.square(p1[j]), np.square(s2[j+currentLag]))
			s1p2[i] = s1p2[i] + np.dot(np.square(s1[j]), np.square(p2[j+currentLag]))

	thisCovariogram = crossCorr - shuffleCorrector
	sigma = np.sqrt((s1s2+p1s2+s1p2)/trials)
	sigHigh = 2*sigma
	sigLow = -2*sigma

	return thisCovariogram, sigHigh, sigLow


##this function identifies the longest timespan in a JPSTH that exceeds significance
#if one exists. 
def significantSpan(vectorA, sig):
	if len(sig) ==1:
		temp = np.zeros(len(vectorA))
		for i in range(len(vectorA)):
			temp[i] = sig[0]
	##find indices that exceed significance
	difference = vectorA > sig

	##after th is you will see -1 on the index before a span
	##starts; you will see 1 on the index where a span ends
	divDifference = np.append(difference[1:], 0) - difference
	##find all beginning indices
	beginnings = np.where(divDifference ==1)[0] + 1
	endings = np.where(divDifference == -1)[0]
	##if the first index is significant we would have missed it
	if endings.size != 0 and len(beginnings) < len(endings):
		beginnings = np.append(1, beginnings)
	#ID the longest span(s)
	maxIndices = (endings-beginnings)==max(endings-beginnings)
	#find indices of the longest spans
	spanEndpoints = []
	theIndices = np.where(maxIndices==1)
	for i in range(len(theIndices)):
		theIndex = theIndices[i]
		spanEndpoints = np.hstack((spanEndpoints, beginnings[theIndex], endings[theIndex]))
	return spanEndpoints

## a function to bin data 
def bin_data(array_in, binSize):
	##handle arrays of different dimensions. Assume inputs are 
	##trials x time
	array_in = np.squeeze(array_in)
	if len(array_in.shape) > 1:
		num_trials = array_in.shape[0]
		N = array_in.shape[1]
	else:
		num_trials = 1
		N  = array_in.shape[0]
		array_in = array_in[None,:]
	num_bins = int(np.floor(N/binSize))
	result = np.zeros((num_trials, num_bins))
	for i in range(num_bins):
		result[:,i] = array_in[:,i*binSize:(i+1)*binSize].sum(axis = 1)
	return np.squeeze(result)










