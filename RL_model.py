"""
RL_0model.py
contains functions for implementing a reinforcement learning model
adapted from Samejima et al, 2005
"""

import numpy as np

#a function to implement the Rescola-Wagner learning
##update rule
##args:
##a: variable represeting an action choice
##r: variable representing a result from action a
##particle: samples represeting PDF of hidden variables
def rescola_wagner(a, r, particle):
	##generate the learning weight
	alpha = np.exp(particle[2,:])
	##update particles with the r-w learning ruls
	particle[a,:] = particle[a,:]+alpha*(r-particle[a,:])
	return particle

## a function to generate bolzmann selection action probability
##args:
## a: action choice
##particle: PDF of hidden variables
def bolzmann(a, particle):
	##get the beta weight
	beta = np.exp(particle[3,:])
	##get the probability of action selection using the update rule
	P = 1/(1+np.exp(-2*(a-1.5)*(beta*np.diff(particle[0:2,:],axis=0))))
	return P

##a function to generate a multinomial distribution approximation
##args:
## m: number of samples
## pdf: density vector
def pdf2rand(pdf, mr):
	c = np.max(pdf)
	r = pdf/c
	r = r.squeeze()
	nn = mr
	sample = np.array([])
	N = pdf.size
	accept = np.zeros(mr)==1
	firstrnd = np.zeros(mr)
	randsample = np.zeros((2,mr))

	while nn > 0:
		randsample = np.random.uniform(size=(2,nn))
		firstrnd = np.floor(randsample[0,:]*N)+1
		accept = r[firstrnd>randsample[1,:]]
		accept = accept.astype(bool)
		sample = np.hstack((sample, firstrnd[accept]))
		nn = np.sum(accept != True)
	return sample

##a subfunction for the sequential monte carlo function
def resampling(w, r_type, n):
	if r_type == 'simple':
		#n = length(w)
		index = pdf2rand(w,n)
	elif r_type == "residual":
		m = n
		#length(w)
		k = np.floor(w*m)
		k = k.squeeze()
		mr = m-np.sum(k)
		index = np.array([])
		for i in np.arange(k.max()):
			index = np.hstack((index, np.where(k>=i)[0]))		
		w = (w*m-k)
		w = w/w.sum()
		resindex = pdf2rand(w,mr)
		index = np.hstack((index, resindex))
	return index

"""
##Sequential monte carlo function.
Inputs:
	-actions: array of action choices; 1 or 0
	-outcomes: array of outcomes; 1 or 0
	-N: number of hidden variables (defaults to 4)
	-M: number of particles 
	-sd_justter: standard deviation for the jitter param
	-particles: initial particles
Returns:
	-result_e: expectation value of hidden params
	-results_s: variances of hidden parameters
"""
def SMC(actions, outcomes, N, M, sd_jitter, particles = None):
	##if particles have not been initialized, do that now
	if particles is None:
		yp = np.random.randn(N,M)
		yph = yp
	else:
		yp = particles
		(N,M) = yp.shape

	trials = actions.size
	result_e = np.zeros((N, trials)) #expectation value of hidden params
	result_s = np.zeros((N,trials)) #variances of hidden params

	for t in range(1,trials):
		yph = rescola_wagner(actions[t-1], outcomes[t-1], yp)
		result_e[:,t] = np.mean(yph,axis=1)
		result_s[:,t] = np.var(yph,axis=1)

		#importance weight
		w = bolzmann(actions[t], yph)
		w = w/np.sum(w) #normalize
		
		#resampling
		index = resampling(w, 'residual', M)
		yp = yph[:,index]

		##add jitter
		yp = yp + np.random.randn(N,M)*(sd_jitter*np.ones(M))

	return result_e, result_s

def init_particles(num_vars=4, num_particles=1000):
	##function to set the initial distribution of particles
	initp = np.random.randn(num_vars,num_particles)+0.5 #Q
	initp[2,:] = np.random.uniform(num_particles)+np.log(0.1) #alpha
	initp[3,:] = np.random.uniform(num_particles) #beta
	return initp
