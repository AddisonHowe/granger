from models import AR
import numpy as np
import matplotlib.pyplot as plt


def strong_and_weak_force_test(N):
	np.random.seed(0)
	# create inputs for 1st order with 3 variables
	num_vars = 3
	order = 1
	# 2nd variable strongly forces 1st variable and 3rd variable weakly forces 1st variable
	# first column corresponds to the constant terms

	# variable index          0    1    2
	# lag                     1    1    1
	coeffs =  np.array([[1,   0,   100, 0.5], #eqn for X_0
						[1,   0,   0,   0],   #eqn for X_1
						[1,   0,   0,   0]])  #eqn for X_2

	std_devs = np.random.uniform(.1,.3,num_vars)
	init_vals = np.random.uniform(-1,1,(num_vars, order))
	# create AR model object
	ar = AR(num_vars,coeffs,std_devs,init_vals)
	# generate data using model
	ar.generate_data(N)
	ar.causality(order,0.05)

	# plot data
	for i in range(num_vars):
		plt.plot(ar.data[i], label="$X_{:d}$".format(i))
	plt.legend()
	plt.show()


def basic_test():
	np.random.seed(0)
	# create inputs for 2nd order with 4 variables
	num_vars = 4
	order = 2

	# first column corresponds to the constant terms
	# variable index          0   0   1   1   2   2   3   3
	# lag                     2   1   2   1   2   1   2   1
	coeffs =  np.array([[40,  1, -3,  1, -2,  1,  3,  0,  0],   #eqn for variable 0
						[30,  0,  2,  0,  0,  0,  0,  0,  0],   #eqn for variable 1
						[10,  0,  0,  0,  0, -3,  1,  0,  0],   #eqn for variable 2 
						[0,   0,  0,  0,  0,  0,  0,  0,  0],]) #eqn for variable 3

	std_devs = np.array([0.1, 0.1, 0.1, .1])
	init_vals = np.random.uniform(-1,1,(num_vars, order))
	N=10000
	# create AR model object
	ar = AR(num_vars,coeffs/10,std_devs,init_vals)
	# generate data using model
	ar.generate_data(N)
	ar.causality(order,0.05)

	# plot data
	for i in range(num_vars):
		plt.plot(ar.data[i], label="$X_{:d}$".format(i))
	plt.legend()
	plt.show()


def nontransitive_test():
	"""
	Demonstrates that if X->Y and Y->Z, X need not cause Z.
	"""
	np.random.seed(3)
	# create inputs for 2nd order with 3 variables
	num_vars = 3
	order = 1

	# first column corresponds to the constant terms
	# variable index         0  1  2
	# lag                    1  1  1
	coeffs =  np.array([[0,  0, 0, 0],   #eqn for variable 0
						[0,  1, 0, 0],   #eqn for variable 1
						[0,  0, 1, 0],]) #eqn for variable 2 

	std_devs = np.array([1., 1., 1.])
	init_vals = np.ones((num_vars, order))
	N=100
	# create AR model object
	ar = AR(num_vars,coeffs,std_devs,init_vals)
	# generate data using model
	ar.generate_data(N)
	ar.causality(order,0.05)


def information_subset_test():
	"""
	Data (X,Y,Z) is generated with Z->X and Z->Y, which is evident for information sets
	{X,Y,Z}, {X,Z}, and {Y,Z}. But X->Y in {X,Y} but not {X,Y,Z}.
	"""
	np.random.seed(3)
	# create inputs for 2nd order with 3 variables
	num_vars = 3
	order = 2

	# first column corresponds to the constant terms
	# variable index         0  0  1  1  2  2  
	# lag                    2  1  2  1  2  1
	coeffs =  np.array([[0,  0, 0, 0, 0, 0, 1],   #eqn for variable 0
						[0,  0, 0, 0, 0, 1, 0],   #eqn for variable 1
						[0,  0, 0, 0, 0, 0, 0],]) #eqn for variable 2 

	std_devs = np.array([1., 1., 1.])
	init_vals = np.random.uniform(-1,1,(num_vars, order))
	N = 1000
	# create AR model object
	ar = AR(num_vars,coeffs,std_devs,init_vals)
	# generate data using model
	xyz = ar.generate_data(N)
	ar.causality(order,0.05)

	xz = xyz[[0,2],:]  # remove Y
	yz = xyz[[1,2],:]  # remove Z
	xy = xyz[[0,1],:]  # remove X

	ar.num_vars = 2
	
	print("*** Information Set J(XZ) ***")
	ar.data = xz
	ar.causality(order, 0.05)

	print("*** Information Set J(YZ) ***")
	ar.data = yz
	ar.causality(order, 0.05)

	print("*** Information Set J(XY) ***")
	ar.data = xy
	ar.causality(order, 0.05)
