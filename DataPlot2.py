import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import SpikeStats2 as ss
import matplotlib as ml
"""
Plot a line with error bars.
Inputs are x-data, y-data, error values, title, x-axis label,
y-axis label
"""
def line_with_error(x, y, err, n= None, title = "MyPlot", xlab = "values", ylab = "values", lab = "data"):
	##create a figure
	#plt.figure()
	##create an errorbar Plot
	plt.plot(x, y, color = np.random.rand(3,), label = lab)
	plt.title(title, fontsize = 20)
	plt.xlabel(xlab, fontsize = 20)
	plt.ylabel(ylab, fontsize = 20)
	plt.fill_between(x, y-err/2.0, y+err/2.0, alpha = 0.5, facecolor = np.random.rand(3,))
	if n is not None:
		plt.text(2, y.max()+y.max()/3.0, "average of "+str(n)+" trials", fontsize = 12)
	#plt.show()

"""
For creating the contingency degredation plot.
Inputs are the paired value arrays for percent correct rates
for pre-cd, peri-cd, and post-cd. 
"""
def plot_cd_data(pre_arr, peri_arr, post_arr):
	
	# Custom function to draw the p-value bars
	def label_diff(i,j,text,X,Y):
		x = (X[i]+X[j])/2 ##center of the p-val bar
		y = max(Y[i], Y[j])
		
		props = {'connectionstyle':'bar','arrowstyle':'-',\
					 'shrinkA':20,'shrinkB':20,'lw':2}
		ax.annotate(text, xy=(x,y+0.1), zorder=10)
		ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)

	##create a numpy array containing the mean vals for the bar chart
	means = np.array([pre_arr.mean(), peri_arr.mean(), post_arr.mean()])
	##get the standard error values
	errs = np.array([stats.sem(pre_arr), stats.sem(peri_arr), stats.sem(post_arr)])
	##calculate the p-values between each of the sets
	p_pre_peri = np.round(stats.ttest_rel(pre_arr, peri_arr)[1], 3)
	p_pre_post = np.round(stats.ttest_rel(pre_arr, post_arr)[1], 3)
	p_peri_post = np.round(stats.ttest_rel(peri_arr, post_arr)[1], 3)
	##put all the arrays into one big array to plot the
	##individual lines
	all_arr = np.zeros((3,pre_arr.size))
	all_arr[0,:] = pre_arr
	all_arr[1,:] = peri_arr
	all_arr[2,:] = post_arr

	##formatting stuff
	idx  = np.arange(3)    # the x locations for the groups
	width= 0.8
	labels = ('Pre', 'CD', 'Reinstatement')

	# Pull the formatting out here
	bar_kwargs = {'width':width,'color':'g','linewidth':2,'zorder':5}
	err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}

	X = idx+width/2 ##position of the center of the bars

	fig, ax = plt.subplots()
	ax.p1 = plt.bar(idx, means, alpha = 0.5, **bar_kwargs)
	ax.errs = plt.errorbar(X, means, yerr=errs, **err_kwargs)

	##plot the individual lines on their own axis
	ax2 = ax.twinx()
	ax2.lines = plt.plot(np.linspace(0,3,3), all_arr)
	ax2.set_ylabel("Percent correct")


	# Call the function
	label_diff(0,1,'p='+str(p_pre_peri),X,means)
	label_diff(0,2,'p='+str(p_pre_post),X,means)
	label_diff(1,2,'p='+str(p_peri_post),X,means)

	ax.set_ylim(ymax=means.max()+0.3)
	plt.xticks(X, labels, color='k')
	plt.title("Performance during contingency degredation")
	ax.set_ylabel("Percent correct")
	plt.show()


"""
the code used to generate the dark/light switch performance plots
"""
def plot_dark_performance(list_of_4_arrays):
	##make the plot window
	fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
	##the x-axis values
	x = np.linspace(0,60, list_of_4_arrays[0].size)
	##plot the arrays and do the shading
	ax1.plot(x, list_of_4_arrays[0]*10)
	ax2.plot(x, list_of_4_arrays[1]*10)
	ax3.plot(x, list_of_4_arrays[2]*10)
	ax4.plot(x, list_of_4_arrays[3]*10)
	ax1.axvspan(0,x.max()/2, alpha = 0.5, color = 'k')
	ax1.axvspan(x.max()/2,x.max(), alpha = 0.5, color = 'y')
	ax2.axvspan(0,x.max()/2, alpha = 0.5, color = 'k')
	ax2.axvspan(x.max()/2,x.max(), alpha = 0.5, color = 'y')
	ax3.axvspan(0,x.max()/2, alpha = 0.5, color = 'k')
	ax3.axvspan(x.max()/2,x.max(), alpha = 0.5, color = 'y')
	ax4.axvspan(0,x.max()/2, alpha = 0.5, color = 'k')
	ax4.axvspan(x.max()/2,x.max(), alpha = 0.5, color = 'y')
	ax1.set_xlim([0,60])
	ax2.set_xlim([0,60])
	ax3.set_xlim([0,60])
	ax4.set_xlim([0,60])
	ax1.set_ylabel("hit rate")
	ax2.set_ylabel("hit rate")
	ax3.set_ylabel("hit rate")
	ax4.set_ylabel("hit rate")
	ax2.set_xlabel("time, mins")
	ax4.set_xlabel("time, mins")
	ax1.set_xlabel("time, mins")
	ax3.set_xlabel("time, mins")
	ax1.set_title("Animal V01")
	ax2.set_title("Animal V02")
	ax3.set_title("Animal V03")
	ax4.set_title("Animal V04")
	plt.suptitle("Performance During Light Change Sessions", fontsize = 20)
	plt.show()

"""
Given a dataset of 2 arrays of spiketrains, this function plots
the average firing rate for each for 2 user-defined sub chunks of time
in each set, and compares them all in a bar graph.
"""
def plot_fr_means(arrs1, arrs2, chunk1 = (0,10), chunk2 = (35,45), n = None):

	##grab the specified chunks
	arrs1_early = arrs1[:,chunk1[0]*60*1000:chunk1[1]*60*1000]
	arrs1_late = arrs1[:,chunk2[0]*60*1000:chunk2[1]*60*1000]
	arrs2_early = arrs2[:,chunk1[0]*60*1000:chunk1[1]*60*1000]
	arrs2_late = arrs2[:,chunk2[0]*60*1000:chunk2[1]*60*1000]
	##calculate the means across all the arrays
	means =np.array([arrs1_early.mean(), 
		arrs2_early.mean(), arrs1_late.mean(), 
		arrs2_late.mean()])*1000
	##get the across session means
	m_arrs1_early = arrs1_early.mean(axis = 1)*1000
	m_arrs2_early = arrs2_early.mean(axis = 1)*1000
	m_arrs1_late = arrs1_late.mean(axis = 1)*1000
	m_arrs2_late = arrs2_late.mean(axis = 1)*1000
	##get an array of SEM mesurements for the error bars
	errs = np.array([stats.sem(m_arrs1_early,axis = None), 
		stats.sem(m_arrs2_early,axis = None),
		stats.sem(m_arrs1_late,axis = None), 
		stats.sem(m_arrs2_late, axis = None)])
	##calculate the t-tests
	p_e1s = stats.ttest_rel(m_arrs1_early, m_arrs1_late)
	p_e2s = stats.ttest_rel(m_arrs2_early, m_arrs2_late)
	p_e12_early = stats.ttest_rel(m_arrs1_early, m_arrs2_early)
	p_e12_late = stats.ttest_rel(m_arrs1_late, m_arrs2_late)
	##print the ttest results
	print "p_e1s = " + str(p_e1s)
	print "p_e2s = " + str(p_e2s)
	print "p_e12_early = " + str(p_e12_early)
	print "p_e12_late = " + str(p_e12_late)
	##plot the bar graph
	##formatting stuff
	idx  = np.arange(4)    # the x locations for the groups
	width= 0.8
	labels = ('E1 early', 'E2_early', 'E1_late', 'E2_late')

	# Pull the formatting out here
	bar_kwargs = {'width':width,'color':'g','linewidth':2,'zorder':5}
	err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}

	X = idx+width/2 ##position of the center of the bars

	fig, ax = plt.subplots()
	ax.p1 = plt.bar(idx, means, alpha = 0.5, **bar_kwargs)
	ax.errs = plt.errorbar(X, means, yerr=errs, **err_kwargs)

	ax.set_ylim(ymax=means.max()+means.max()/6.0)
	plt.xticks(X, labels, color='k')
	plt.title("Average firing rate within sessions")
	ax.set_ylabel("FR (Hz)")
	if n is not None:
		plt.text(0.2, means.max()+means.max()/10, "n= "+str(n)+" sessions")

	plt.show()

def plot_locked_frs(e1_arr, e2_arr, id_arr, sigma):
	N = e1_arr.shape[0]    
	e1_arr = ss.masked_avg(ss.zscored_fr(e1_arr, sigma))
	e2_arr = ss.masked_avg(ss.zscored_fr(e2_arr, sigma))
	id_arr = ss.masked_avg(ss.zscored_fr(id_arr, sigma))
	x = np.linspace(-N/2.0, N/2.0, e1_arr.shape[0])
	plt.plot(x, e1_arr, label = "E1 Units", color = 'g')
	plt.plot(x, e2_arr, label = "E2 Units", color = 'r')
	plt.plot(x, id_arr, label = "Indirect Units", color = 'k')
	plt.xlabel("Time to target, ms")
	plt.ylabel("Firing Rate (zscore)")
	plt.legend()
	plt.title("Mean FR for Target Hits, Matlab Code")
	plt.vlines(0, e1_arr.max(), e2_arr.min(), linestyle = 'dashed')
	plt.show()
	
def plot_modulation_depth(arr_early, arr_late, sigma):
	arr_early = ss.zscored_fr(arr_early, sigma).max(axis = 0)
	arr_early = np.nan_to_num(arr_early)
	arr_late = ss.zscored_fr(arr_late, sigma).max(axis = 0)
	arr_late = np.nan_to_num(arr_late)
	if arr_early.size > arr_late.size:
		arr_early = np.random.choice(arr_early, size = arr_late.size, replace = False)
	if arr_late.size > arr_early.size:
		arr_late = np.random.choice(arr_late, size = arr_early.size, replace = False)
	early_sem = stats.sem(arr_early)
	early_mean = arr_early.mean()
	late_sem = stats.sem(arr_late)
	late_mean = arr_late.mean()
	p_val = stats.ttest_rel(arr_early, arr_late)
	print "p val is = " + str(p_val)
	# Pull the formatting out here
	width = 0.8	
	bar_kwargs = {'width':width,'color':'g','linewidth':2,'zorder':5}
	err_kwargs = {'zorder':0,'fmt':None,'lw':2,'ecolor':'k'}	
	means = np.array([early_mean, late_mean])
	errs = np.array([early_sem, late_sem])
	idx = np.arange(2)
	X = idx+width/2	
	labels = ['E1 early', 'E1_late']
	plt.bar(idx, means, alpha = 0.5,**bar_kwargs)
	plt.errorbar(X, means, yerr = errs,**err_kwargs)
	plt.xticks(X, labels)
	plt.ylabel('z-scored modulation depth')
	plt.title('Change in modulation depth from early in session to late in session')
	plt.show()
	
##takes in an array of wfs in the format spikes x samples and plots the wf + std
##along with the calculated SNR
def plot_wf(wf_array):
	mean = wf_array.mean(axis = 0)
	std = wf_array.std(axis = 0)
	x = np.linspace(0,800,mean.size)
	color = np.random.rand(3,)
	plt.figure()
	plt.plot(x,mean, color = color)
	plt.fill_between(x,mean+std, mean-std, alpha = 0.5, facecolor = color)
	snr = ss.calc_unit_snr(wf_array)
	plt.text(600, -1*mean.max()/2, "SNR= "+str(np.round(snr,3)))
	plt.xlabel("time, uS")
	plt.ylabel('uV')
	
def spike_raster(spike_trains, midpoint, color = 'k'):
	for train in range(spike_trains.shape[1]):
		plt.vlines(ml.mlab.find(spike_trains[:,train] >0), 0.5+train, 1.5+train, color = color)

	plt.vlines(midpoint, 0, spike_trains.shape[1], color = 'r')
	plt.title('Raster plot')
	plt.xlabel('time')
	plt.ylabel('trial')
	plt.show()