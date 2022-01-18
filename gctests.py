import numpy as np
import scipy.stats

"""
Test the null hypothesis that Var(X)=Var(Y) at a significance level of alpha.
Returns true if the null hypothesis is rejected, false otherwise.
"""
def F_test(X, Y, alpha):
	n = len(X)
	m = len(Y)
	EX = np.mean(X)
	EY = np.mean(Y)
	varX = 1/(n-1)*np.sum((X-EX)**2)
	varY = 1/(m-1)*np.sum((Y-EY)**2)
	p_value = scipy.stats.f.cdf(varX/varY, n, m)
	return p_value < alpha
