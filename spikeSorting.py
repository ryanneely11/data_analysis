## a series of functions used for analyzing raw ephys data and extracting spikes
##created by Ryan Neely (ryanneely11@gmail.com)

from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy import signal
import copy
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import pdist
from pandas.tools.plotting import scatter_matrix
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from mpl_toolkits.mplot3d import Axes3D
import matplotlib


"""
spikeBandFilter: this function uses a Butterworth bandpass filter to extract 
the spikeband data.

Params:

data: 1-D numpy array of raw ehphys data
lowcut: the low frequency corner (Hz)
highcut: the high freq corner
fs: the sample rate of the data
order: the order of the butterworth filter to uses

returns: a 1-D numpy array of filtered data
"""

def spikeBandFilter(data, lowcut = 300, highcut = 5000, 
	fs = 24414.0625, order = 5, plot = True):
	
	##check the data dimensions
	data = np.squeeze(data)
	if len(data.shape) > 1:
		raise ValueError("Needs 1-D array!")

	##define filter functions
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

	filtered = butter_bandpass_filter(data, lowcut, highcut, fs, order)

	if plot:
		timebase = np.linspace(0,int(data.size/fs), data.size)
		fig, (ax1,ax2) = plt.subplots(2)
		fig.set_size_inches(8,9)
		ax1.plot(timebase[:int(5*fs)], data[:int(5*fs)], color = 'k')
		ax2.set_xlabel("time (s)")
		ax1.set_title("Raw data")
		ax2.plot(timebase[:int(5*fs)], filtered[:int(5*fs)], color = 'r')
		ax2.set_title("Filtered data")
		fig.suptitle("Filtering results (first 5 sec)", fontsize = 15)

	#process the data
	return filtered


""" 
Helper: Returns the median absolute deviation.

**Parameters**

x : double
    Data array

**Returns**

The median absolute deviation of input signal x
"""
def mad(x):

	return 1.4826 * np.median(np.abs(x - np.median(x)))

"""
MADnormalize: This function normalizes the data by the median absolute deviation (MAD)

Params: 

data: 1-D array of data to normalize

"""

def MADnormalize(data, fs=24414.0625, plot=True):
	""" Data renormalization (divise by mad) """
	mad_values = mad(data)
	median_values = np.median(data)
	timebase = np.linspace(0,(data.shape[0]/fs), data.shape[0])

	result = (data - median_values)/mad_values

	if plot:
		fig, ax = plt.subplots(1)
		ax.plot(timebase[:int(0.1*fs)],result[:int(0.1*fs)],color="black")
		ax.axhline(y=1,color="red",label="MAD")
		ax.axhline(y=-1,color="red")
		ax.axhline(y=np.std(result),color="blue",linestyle="dashed",label="std dev")
		ax.axhline(y=-np.std(result),color="blue",linestyle="dashed")
		ax.set_xlabel('Time (s)',fontsize=12)
		fig.set_size_inches(9,8)
		ax.legend()
		fig.suptitle("MAD and SD; first 100 ms",fontsize = 16)
	return result


"""
detectPeaks: This function detects peaks in the data, basically using a threshold
to identify putative spikes. First it filters the data using a boxcar filter (which averages
the data over a small window). Then it detects threshold crossings with a user-defined threshold.

Params:

data: 1-D numpy array of spikeband filtered and normalized data
thresh: threshold value to use to detect peaks
fs: sampling frequency

Returns:
thresh_data: a filtered and rectified version of the data that can below
later used to determine the timestamp of the actual peaks/spikes.

"""
def detectPeaks(data, thresh, fs=24414.0625, plot=True):

	###the boxcar filter
	win = np.array([1., 1., 1., 1., 1.])/5.
	##apply the filter 
	data = signal.fftconvolve(data, win, 'same')
	##take the MAD of the filtered data
	mad_value = mad(data)
	data = data/mad_value
	##set all points above/below the threshold = 0
	thresh_data = copy.copy(data)
	if thresh > 0:
		thresh_data[data < thresh] = 0
	elif thresh < 0:
		thresh_data[data > thresh] = 0

	if plot:
		fig, ax = plt.subplots(1)
		timebase = np.linspace(0,(data.shape[0]/fs), data.shape[0])
		ax.plot(timebase,thresh_data, label = 'threshold_detect', color = 'r', lw = 3)
		ax.plot(timebase,data, label = 'raw_data', color = 'k')
		ax.axhline(y=thresh,color="b",label="threshold",linestyle = 'dashed')
		ax.set_xlabel("time (s)",fontsize = 12)
		ax.legend()
		ax.set_xlim(0,5)
		fig.suptitle("Spike thresholding results (first 5 sec)", fontsize = 16)
		fig.set_size_inches(10,8)
	return thresh_data


def peaks(norm_data, thresh_data, minimalDist=1, notZero=1e-3, fs = 24414.0625, thresh = -4.5, plot = True):
        """ Detects the peaks over filtered and rectified data from the above function

            **Parameters**
            norm_data: input normalized data

            thresh_data : double
                Input thresholded data

            minimalDist : int
                The minimal distance between two successive peaks (in ms, NOT samples).
                (only putative maxima that are farther apart than minimalDist
                 sampling points are kept)

            notZero : double
                The smallest value above which the absolute value of the
                derivative is considered not null.

            fs = sample rate in Hz

            thresh: is the threshold used to detect the peaks 'positive' or 'negative'?

            **Returns**
            An array of (peak) indices is returned.

        """
    ##window filter to convolve with
	win = np.array([1, 0, -1])/2.
	##convert the minimal distance into samples
	minimalDist = minimalDist*int(fs/1000)
	x = np.asarray(thresh_data)
	##convolve the data with the window
	dx = signal.fftconvolve(x, win, 'same')
	#find the peaks that fit the threshold requirement; all other points set to zero
	dx[np.abs(dx) < notZero] = 0
	dx = np.diff(np.sign(dx))
	#find the timestamps of the points
	if thresh < 0:
		pos = np.arange(len(dx))[dx > 0]
	if thresh > 0:
		pos = np.arange(len(dx))[dx < 0]
	##filter out spikes that occur too colse together
	peaks = pos[1:][np.diff(pos) > minimalDist]

	if plot:
		timebase = np.linspace(0,int(norm_data.shape[0]/fs),norm_data.shape[0])
		fig, ax = plt.subplots(1)
		ax.plot(timebase,norm_data,color = 'k')
		ax.plot(timebase[peaks], norm_data[peaks], 'ro')
		ax.axhline(y=thresh,color="b",linestyle = 'dashed')
		ax.set_xlim(0,5)
		ax.set_xlabel("time, (s)")
		fig.suptitle("Peak detection, first 5 sec", fontsize = 16)
		fig.set_size_inches(10,9)
	return peaks


def cutEvents(x, timestamps, before=14, after=30):
	""" Constructs a list of all the events from the input data.

	**Parameters**

	x : double (array)
	Input data


	timestamps : int (array)
	the points at which events were detected by the previous functions

	before : int
	The number of points before of an peak

	after : int
	The number of points after a peak (both parameters, before and
	after, define the size of an event segment)

	**Returns**
	A matrix with as many rows as events and whose rows are the cuts
	on the different recording sites glued one after the other.
	"""

	res = np.zeros((timestamps.shape[0],(before+after+1)))
	for i, p in enumerate(timestamps):
		res[i, :] = cut_sgl_evt(x, p, before, after)
	return copy.deepcopy(res)

def cut_sgl_evt(x, position, before, after):
	""" Draw a singles event from the input data (Helper function)

	**Parameters**

	x : double (array)
	Input data

	position : int
	The index (location) of the (peak of) the event.

	before : int
	How many points should be within the cut before the reference
	index / time given by position.

	after : int
	How many points should be within the cut after the reference
	index / time given by position.

	**Returns**
	A vector with the cut data waveforms.
	"""
	dl = x.shape[0]             # Number of sampling points    
	cs = before + after + 1                # The 'size' of a cut
	cut = np.zeros(cs)
	idx = np.arange(-before, after + 1)
	keep = idx + position
	within = np.bitwise_and(0 <= keep, keep < dl)
	kw = keep[within]
	cut[within] = x[kw].copy()
	return cut

def plotEvents(event_array):
	"""
	a function to plot the events that were cut using the above functions"

	Params:
	event_array: the events x samples array of events

	"""
	##caclulate some statistics 
	events_median = np.median(event_array,axis = 0)
	events_mad = np.apply_along_axis(mad,0,event_array)

	##Probably don't want to plot all of the traces, so only plot a max of 200
	if event_array.shape[0] > 200:
		samples = event_array[:200,:]
	else:
		samples = event_array

	x = np.linspace
	
	fig, (ax1, ax2) = plt.subplots(2)
	ax1.plot(events_median, color='red', lw=2, label = "Median")
	ax1.axhline(y=0, color='black')
	ax1.plot(events_mad, color='blue', lw=2, label = "MAD")
	for i in np.arange(0,event_array.shape[1],10): 
		ax1.axvline(x=i, color='grey')
	ax1.legend()
	ax1.set_title("Waveform statistics", fontsize = 14)

	for i in range(samples.shape[0]):
		ax2.plot(samples[i,:], 'k', lw=0.1)
	ax2.plot(events_median, 'r', lw = 1)
	ax2.plot(events_mad, 'b', lw = 1)
	ax2.set_title("Sample waveforms", fontsize = 14)
	ax2.set_xlabel("samples")
	fig.set_size_inches(10,10)
	fig.subplots_adjust(hspace = 0.5)

def cutNoise(data, timestamps, before, after, safety_factor=2, size=2000):
	""" Computes the noise events from a dataset

	**Parameters**

	data : double
	    Input data

	timestamps: int
		the timestamps of the (non-noise) events to reference

	before and after:
		"cut length," or number of samples before and after timestamps used to make the events
		(see above function)

	safety_factor : int
	    A number by which the cut length is multiplied and which
	    sets the minimal distance between the reference times

	size : int
	    The maximal number of noise events one wants to cut

	**Returns**

	A matrix with as many rows as noise events and whose rows are
	the cuts of noise events
	"""

	sl = before + after + 1   # cut length
	i1 = np.diff(timestamps)        # inter-event intervals
	minimal_length = round(sl*safety_factor)
	# Get next the number of noise sweeps that can be
	# cut between each detected event with a safety factor
	nb_i = (i1 - minimal_length)//sl
	# Get the number of noise sweeps that are going to be cut
	nb_possible = min(size, sum(nb_i[nb_i > 0]))
	res = np.zeros((nb_possible, sl))
	# Create next a list containing the indices of the inter event
	# intervals that are long enough
	idx_l = [i for i in range(len(i1)) if nb_i[i] > 0]
	# Make next an index running over the inter event intervals
	# from which at least one noise cut can be made
	interval_idx = 0
	# noise_positions = np.zeros(nb_possible,dtype=numpy.int)
	n_idx = 0
	while n_idx < nb_possible:
		within_idx = 0  # an index of the noise cut with a long
		                # enough interval
		i_pos = int(timestamps[idx_l[interval_idx]] + minimal_length)
		# Variable defined next contains the number of noise cuts
		# that can be made from the "currently" considered long-enough
		# inter event interval
		n_at_interval_idx = nb_i[idx_l[interval_idx]]
		while within_idx < n_at_interval_idx and n_idx < nb_possible:
			res[n_idx, :] = cut_sgl_evt(data, i_pos, before, after)
			## noise_positions[n_idx] = i_pos
			n_idx += 1
			i_pos += sl
			within_idx += 1
		interval_idx += 1
	return res


def getGoodEvents(x, thr=3):
	""" Try to detect events that have superimposed spikes 
		(And get rid of them).

		**Parameters**

		x : double
		    Data array

		thr : double
		    Threshold of filtering

		**Returns**
		A vector containing all the detected "good" events.
	"""
    
	def f(x, median, mad, thr):
		""" Auxiliary function, used by good_evts_fct.

			**Parameters**

			x : double
			    Data array

			median : double
			    Array contains median values

			mad : double
			    Array contains mad values

			thr : double
			    Filtering threshold

			**Returns**
			A numpy array containing the data for which the |X-median(X)|/mad(X) <
			thr.
		"""
		return np.ndarray.all(np.abs((x - median)/mad) < thr)

	samp_median = np.apply_along_axis(np.median, 0, x)
	samp_mad = np.apply_along_axis(mad, 0, x)
	above = samp_median > 0

	samp_r = copy.copy(x)

	for i in range(len(x)):
		samp_r[i][above] = 0

	samp_median[above] = 0
	res = np.apply_along_axis(f, 1, samp_r, samp_median,
		samp_mad, thr)
	return res

class pca_clustering(object):
	""" Clustering methods and dimension-reduction techniques """
	def __init__(self, events, noise):
		""" Performs the cleaning of the events and a singular value
		decomposition in order to obtain the principal components of the
		data.

		**Parameters**

		events : double
		    the array of isolated events (event x sample)

		noise : double
		    A numpy array that contains the noise events (event x sample)
		"""
		self.evts = events
		self.noise = noise

		varcovmat = np.cov(self.evts.T)
		# Perform a singular value decomposition
		self.U, self.S, self.V = svd(varcovmat)

	def plotMeanPca(self, start=0):
		""" Plots the mean of the data plus-minus the principal components """
		evt_idx = range(self.evts.shape[1])
		evts_mean = np.mean(self.evts, 0)
	    
		for i,j in enumerate(range(start,start+4)):
			plt.subplot(2, 2, i+1)
			plt.plot(evt_idx, evts_mean, 'k', label = "mean wf")
			plt.plot(evt_idx, evts_mean + 5 * self.U[:, j], 'r', label = "mean+5*PC")
			plt.plot(evt_idx, evts_mean - 5 * self.U[:, j], 'b', label = "mean-5*PC")
			plt.title('PC' + str(j) + ': ' + str(round(self.S[j]/sum(self.S) *
				100)) + '%')
		matplotlib.pyplot.gcf().get_axes()[1].legend()
		fig = matplotlib.pyplot.gcf()
		fig.set_size_inches(10, 10)
		fig.suptitle("Variance captured by PCs", fontsize = 14)

	def pcaVariance(self, n_pca):
		""" Returns the variance of the principal components.

		**Parameters**

		n_pca : int
		    Number of principal components to be taken into account

		**Returns**

		The variance of the principal component analysis.
		"""
		noiseVar = sum(np.diag(np.cov(self.noise.T)))
		evtsVar = sum(self.S)
		return [(i, sum(self.S[:i]) + noiseVar - evtsVar) for i in range(n_pca)]

	def plotPcaProjections(self, pca_components=(0, 4)):
		""" Plots the principal components projected on the data.

		**Parameters**

		pca_components : int (tuple)
		    The number of the principal components to be projected
		"""
		tmp = np.dot(self.evts,self.U[:, pca_components[0]:pca_components[1]])
		df = pd.DataFrame(tmp)
		scatter_matrix(df, alpha=.2, s=4, c='k', figsize=(10, 10), diagonal='kde', marker=".")

	def plot_PCA(projection_data):
		x = []
		y = []
		z = []
		for item in projection_data:
			x.append(item[0])
			y.append(item[1])
			z.append(item[2])

		#plt.close('all') # close all latent plotting windows
		fig1 = plt.figure() # Make a plotting figure
		ax = Axes3D(fig1) # use the plotting figure to create a Axis3D object.
		pltData = [x,y,z] 
		ax.scatter(pltData[0], pltData[1], pltData[2], 'bo') # make a scatter plot of blue dots from the data

		# make simple, bare axis lines through space:
		xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0)) # 2 points make the x-axis line at the data extrema along x-axis 
		ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r') # make a red line for the x-axis.
		yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0)) # 2 points make the y-axis line at the data extrema along y-axis
		ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r') # make a red line for the y-axis.
		zAxisLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2]))) # 2 points make the z-axis line at the data extrema along z-axis
		ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r') # make a red line for the z-axis.

		# label the axes 
		ax.set_xlabel("x-axis") 
		ax.set_ylabel("y-axis")
		ax.set_zlabel("z-axis")
		ax.set_title("PCA results")
		fig1.set_size_inches(10,10)

	def k_means(self, n_clusters, init='k-means++', n_init=100, max_iter=100, n_pca=(0, 3)):
		""" It computes the k-means clustering over the dimension-reducted
		data.

		**Parameters**

		n_clusters : int
		The number of the clusters

		init : string
		Method for initialization (see scikit-learn K-Means for more
		information)

		n_init : int
		Number of time the k-means algorithm will be run with different
		centroid seeds

		max_iter : int
		Maximum number of iterations of the k-means algorithm for a
		single run

		n_pca : int (tuple)
		Chooses which PCs are used

		**Returns**
		The indices for each neuron cluster.
		"""
		km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter)
		km.fit(np.dot(self.evts, self.U[:, n_pca[0]:n_pca[1]]))
		c = km.fit_predict(np.dot(self.evts, self.U[:, n_pca[0]:n_pca[1]]))

		c_med = list([(i, np.apply_along_axis(np.median, 0,
			self.evts[c == i, :])) for i in range(10) if sum(c == i) > 0])
		c_size = list([np.sum(np.abs(x[1])) for x in c_med])
		new_order = list(reversed(np.argsort(c_size)))
		new_order_reverse = sorted(range(len(new_order)),
			key=new_order.__getitem__)
		return [new_order_reverse[i] for i in c]

	def GMM(self, n_comp, cov_type, n_iter=100, n_init=100, init_params='wmc',
            n_pca=(0, 3)):
		""" It clusters the data points using a Gaussian Mixture Model.

		** Parameters **

		n_comp : int
		    Number of mixture components

		cov_type : string
		    Covarianve parameters to use

		n_iter : int
		    Number of EM iterations to perform

		n_init : int
		    Number of initializations to perform

		init_params : string
		    Controls which parameters are updated in the training process.

		n_pca : int (tuple)
		    Controls which PCs are used

		**Returns**
		The indices for each cluster.
		"""
		gmm = GMM(n_components=n_comp, covariance_type=cov_type, n_iter=n_iter, 
			n_init=n_init, init_params=init_params)

		gmm.fit(np.dot(self.evts, self.U[:, n_pca[0]:n_pca[1]]))

		c = gmm.predict(np.dot(self.evts, self.U[:, n_pca[0]:n_pca[1]]))

		c_med = list([(i, np.apply_along_axis(np.median, 0, 
			self.evts[c == i, :])) for i in range(10) if sum(c == i) > 0])

		c_size = list([np.sum(np.abs(x[1])) for x in c_med])
		new_order = list(reversed(np.argsort(c_size)))
		new_order_reverse = sorted(range(len(new_order)), key=new_order.__getitem__)
		
		return [new_order_reverse[i] for i in c]

	    # TODO: To finish the bagged clustering routine
	def bagged_clustering(self, n_bootstraps, n_samples, n_iter, show_dendro=False, n_pca=(0, 3)):
		""" Performs a bagged clustering (using hierarchical clustering and
		    k-means) on the events data.

		** Parameters **

		n_bootstraps : int
		    Number of bootstraped samples to create

		n_samples : int
		    Number of samples each bootstraped set contains

		n_iter : int
		    The maximum number of k-Means iterations

		show_dendro : boolean
		    If it's true the method displays the dendrogram

		n_pca : int (tuple)
		    The number of PCs which are used
		"""

		B, N = n_bootstraps, n_samples
		data = np.dot(self.evts, self.U[:, n_pca[0]:n_pca[1]])
		size_r, size_c = data.shape[0], data.shape[1]

		if n_samples > data.shape[0]:
			print 'Too many sample points'
			return -1

		# Construct B bootstrap training samples and run the base cluster
		# method - KMeans
		C = []
		for i in range(B):
			centroids, _ = kmeans(data[np.random.randint(0, size_r, (N,)), :], k_or_guess=N, iter=n_iter)
			C.extend(centroids)

		# Run a hierarchical clustering
		distMatrix = pdist(C, 'euclidean')
		D = linkage(distMatrix, method='single')

		# Create the dendrogram
		if show_dendro == 'True':
			dendrogram(D)

		# Cut the tree
		F = fcluster(D, 2, criterion='maxclust')
		return F

	def plot_sorted_PCA(self,projection_data,k_means_result):
    
		##make sure the data is in array format
		k_means_result = np.asarray(k_means_result)
		fig1 = plt.figure()
		# use the plotting figure to create a Axis3D object.
		ax = Axes3D(fig1) 
		##figure out how many clusters were identified with kmeans
		num_clusters = max(k_means_result)+1
		##plot the first three dimensions of each cluster separately
		for i in range(num_clusters):
			sub_data = projection_data[k_means_result==i]
			x = []
			y = []
			z = []
			for item in sub_data:
				x.append(item[0])
				y.append(item[1])
				z.append(item[2])
			pltData = [x,y,z] 
			ax.scatter(pltData[0], pltData[1], pltData[2], edgecolors = 'face', 
				c = np.random.rand(3,))

		# make simple, bare axis lines through space:
		xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0)) # 2 points make the x-axis line at the data extrema along x-axis 
		ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r') # make a red line for the x-axis.
		yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0)) # 2 points make the y-axis line at the data extrema along y-axis
		ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r') # make a red line for the y-axis.
		zAxisLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2]))) # 2 points make the z-axis line at the data extrema along z-axis
		ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r') # make a red line for the z-axis.

		# label the axes 
		ax.set_xlabel("x-axis") 
		ax.set_ylabel("y-axis")
		ax.set_zlabel("z-axis")
		ax.set_title("PCA results")
		fig1.set_size_inches(10,10)


	def plot_event(self, x, n_plot=None, events_color='black', events_lw=0.1, 
		show_median=True, median_color='red', median_lw=0.5, show_mad=True, 
		mad_color='blue', mad_lw=0.5):
		""" Plots an event after clustering.

		**Parameters**

		x : double (list or array)
		Data to be plotted
		n_plot : int
		Number of events that will be plotted
		events_color : string
		Lines color
		events_lw : float
		Line width
		show_median : boolean
		If it's True the median will appear in the figure
		median_color : strin
		Median curve color
		median_lw : float
		Median curve width
		show_mad : boolean
		It it's true the mad will appear in the figure
		mad_color : string
		Mad curve color
		mad_lw : float
		Mad curve width
		"""
		x = np.asarray(x)

		if n_plot is None:
			n_plot = x.shape[0]

		for i in range(n_plot):
			plt.plot(x[i, :], color=events_color, lw=events_lw)

		if show_median:
			MEDIAN = np.apply_along_axis(np.median, 0, x)
			plt.plot(MEDIAN, color=median_color, lw=median_lw)

		if show_mad:
			MAD = np.apply_along_axis(mad, 0, x)
		plt.plot(MAD, color=mad_color, lw=mad_lw)


	def plotClusters(self, clusters, Size=(11, 8)):
		""" Plots events belong to five different clusters.

		**Parameters**

		clusters : int (array or list)
		The index of the cluster from which the events will be plotted
		"""
		num_clusters = max(clusters)+1

		fig = plt.figure(figsize=Size)
		fig.subplots_adjust(wspace=.3, hspace=.3)

		for i in range(num_clusters):
			ax = fig.add_subplot(1,num_clusters,i+1, sharey=plt.gca())
			self.plot_event(self.evts[np.array(clusters) == i, :])



def get_spiketrains(peak_times, k_means_result):
	"""
	a function that takes the clustering results and separates
	the detected peak times into spiketrains containing the spike times
	of the individual putative units

	***Parameters***
	peak_times: the originally detected threshold crossings (with the "bad" events 
		removed)

	k_means_result:
	the "sort codes" from the clustering results

	"""

	peak_times = np.asarray(peak_times).squeeze()
	k_means_result = np.asarray(k_means_result).squeeze()

	if peak_times.shape != k_means_result.shape:
		raise ValueError("Dimension mismatch")
	else:
		##number of units sorted
		num_clusters = k_means_result.max()+1
		##container
		spiketrains = []
		for i in range(num_clusters):
			spiketrains.append(peak_times[k_means_result==i])

		return spiketrains

def plot_spiketrains(data, spiketrains, fs = 24414.0625):
	timebase = np.linspace(0,int(data.shape[0]/fs2),data.shape[0])
	fig, ax = plt.subplots(1)
	ax.plot(timebase,data,color = 'k')
	for i in range(len(spiketrains)):
		ax.plot(timebase[spiketrains[i]], data[spiketrains[i]], 
				marker = 'o', linestyle = "None",
				color = np.random.rand(3,), label = "Unit "+str(i))
	ax.set_xlim(0,5)
	ax.set_xlabel("time, (s)")
	ax.legend()
	fig.suptitle("Sorting results, first 5 sec", fontsize = 16)
	fig.set_size_inches(10,9)



