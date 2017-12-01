###lin_regression.py
###functions to do linear regression

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, explained_variance_score

"""
A function to perform the regression using the regression class
from scikit-learn. Optimized here for use with multiple
independent and dependent variables. Returns a score of how well the 
data predicts the outcome.
Inputs:
	-X: independent variable, here this will probably be spike rates
		over some window
	-y: dependent, catagorical variable. Here this is trial outcome, or
		lever choice, etc.
Returns:
	score: the mean score from 3-fold cross validation
"""
def run_cv(X,y):
	##get X im the correct shape
	if len(X.shape) == 1:
		X = X.reshape(-1,1)
	lr = linear_model.RidgeCV(fit_intercept=True)
	##make a scorer object using explained variance
	scorer = make_scorer(explained_variance_score)
	##make a cross validation object to use for x-validation
	kf = KFold(n_splits=3,shuffle=True)
	score = cross_val_score(lr,X,y,n_jobs=1,cv=kf,scoring=scorer) ##3-fold x-validation using kappa score
	return score.mean()


""" 
a function to run a permutation test on cross-validation data.
The idea is to test how well an independent variable (ie spike rates)
can predict an outcome, ie lever choice. To test this, we will compute the
cross validation score for the model fit with the actual data, then
shuffle the arrays and see how well the data is fit when shuffled. 
We will do this many times and see how frequently the shuffled data predicts better
than the actual data. The idea is that if the actual data is in fact predictive,
it should almost always outperform the shuffled data.
Inputs:
	args: tuple of arguments in the following order:
	 -X: independent variable, here this will probably be spike rates
		over some window
	-y: dependent, catagorical variable. Here this is trial outcome, or
		lever choice, etc.
Returns:
	p_val: percentage of the time that the shuffled data outperformed the
		actual data
"""
def permutation_test(X,y):
	repeat = 500
	if len(X.shape) == 1:
		X = X.reshape(-1,1) ##only needed in the 1-D X case
	lr = linear_model.RidgeCV(fit_intercept=True) ##set up the model
	##make a scorer object using matthews correlation
	scorer = make_scorer(explained_variance_score)
	##make a cross validation object to use for x-validation
	kf = KFold(n_splits=3,shuffle=True) ##VERY important that shuffle == True (not default in sklearn)
	##get the accuary score for the actual data
	f1_actual = cross_val_score(lr,X,y,n_jobs=1,scoring=scorer,cv=kf).mean() ##3-fold x-validation using f1 score
	##now repeat with shuffled data
	times_exceeded = 0 ##numer of times the suffled data predicted better then the actual
	for i in range(repeat):
		y_shuff = np.random.permutation(y)
		f1_test = cross_val_score(lr,X,y_shuff,n_jobs=1,scoring=scorer,cv=kf).mean()
		if f1_test > f1_actual:
			times_exceeded += 1
	return float(times_exceeded)/repeat