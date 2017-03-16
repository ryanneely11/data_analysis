##lin_regression2.py

##a function to run linear regression with scikit_learn; 
##does x-validation, accuracy testing an permutation testing 
##from scratch (I don't like scikit-learn's)

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import multiprocessing as mp

"""
A function to fit a cross-validated Ridge regression model, and return the 
model accuracy using three-fold cross-validation.
inputs:
	X: the independent data; could be spike rates over some window.
		in shape samples x features (ie, trials x spike rates)
	y: the class data; should be binary. In shape (trials,)
	n_iter: the number of times to repeat the x-validation (mean is returned)
Returns:
	accuracy: mean proportion of test data correctly predicted by the model.
"""
def lin_fit(X,y,n_iter=5):
	##get X in the correct shape for sklearn function
	if len(X.shape) == 1:
		X = X.reshape(-1,1)
	##init the model class
	lr = linear_model.LinearRegression(fit_intercept=True)
	R2 = np.zeros(n_iter)
	for i in range(n_iter):
		##split the data into train and test sets
		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
		##now fit to the test data
		lr.fit(X_train,y_train)
		##now try to predict the test data
		y_pred = lr.predict(X_test)
		##get the r2 value for this prediction
		r2 = calc_R2(y_test,y_pred)
		##lastly, compare the accuracy of the prediction
		R2[i] = r2
	return R2.mean()


"""
A function to perform a permutation test for significance
by shuffling the training data. Uses the cross-validation strategy above.
Inputs:
	args: a tuple of arguments, in the following order:
		X: the independent data; trials x features
		y: the class data, in shape (trials,)
		n_iter_cv: number of times to run cv on each interation of the test
		n_iter_p: number of times to run the permutation test
returns:
	accuracy: the computed accuracy of the model fit
	p_val: proportion of times that the shuffled accuracy outperformed
		the actual data (significance test)
"""
def permutation_test(args):
	##parse the arguments tuple
	X = args[0]
	y = args[1]
	n_iter_cv = args[2]
	n_iter_p = args[3]
	##get the accuracy of the real data, to use as the comparison value
	r2_actual = lin_fit(X,y,n_iter=n_iter_cv)
	#now run the permutation test, keeping track of how many times the shuffled
	##accuracy outperforms the actual
	times_exceeded = 0 
	for i in range(n_iter_p):
		y_shuff = np.random.permutation(y)
		r2_shuff = lin_fit(X,y_shuff,n_iter=n_iter_cv)
		if r2_shuff > r2_actual:
			times_exceeded += 1
	return r2_actual, float(times_exceeded)/n_iter_p

"""
a function to run permutation testing on multiple
y datasets that all correspond to one X dataset; ie
many y's that may correspond to the same X. For example, data from several
units recorded simultaneously. We will be
using python's multiprocessing function to speed things up.
Inputs:
	X: dependent data in the form n_trials x n_samples
		ie, n_trials x n_units
	y: datasets to predict, in the form n_trials x n_datasets
	n_iter_cv: the number of cross-validation iterations to run
	n_iter_p: the number of permutation iterations to run
Returns:
	R2: an array of the r2 values for each dataset
	p_vals: an array of the significance values for each dataset
"""
def permutation_test_multi(X,y,n_iter_cv=5,n_iter_p=500):
	##setup multiprocessing to do the permutation testing
	arglist = [(X,y[:,n],n_iter_cv,n_iter_p) for n in range(y.shape[1])]
	pool = mp.Pool(processes=mp.cpu_count())
	async_result = pool.map_async(permutation_test,arglist)
	pool.close()
	pool.join()
	results = async_result.get()
	##parse the results
	R2s = np.zeros(y.shape[1])
	p_vals = np.zeros(y.shape[1])
	for i in range(len(results)):
		R2s[i] = results[i][0]
		p_vals[i] = results[i][1]
	return R2s,p_vals

"""
A helper function to do R2 calculation, given true and predicted data.
Inputs:
	y_true: the true values of y
	y_pred: the predicted values of y
returns:
	r2: the coefficient of determination
"""
def calc_R2(y_true,y_pred):
	u = ((y_true-y_pred)**2).sum() ##the regression sum of squares
	v = ((y_true-y_true.mean())**2).sum() ##the residual sum of squares
	if v > 0:
		r2 = 1-float(u)/float(v)
	else:
		r2 = 0
	return r2