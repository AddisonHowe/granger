import numpy as np
import matplotlib.pyplot as plt

class AR():
	### INPUTS ###
	## coeffs: matrix of the coefficients used in the AR model
	##         if inconsistent number of lagged vals used for certain vars,
	##		   just add zeros
	## num_vars: number of variables used in the multivariate model
	## N: total number of time steps for generating data
	## std_devs: vector of standard deviations for noise component of each var
	## init_vals: matrix of initial values, must be consistent with coeff matrix
	def __init__(self,num_vars,N,coeffs,std_devs,init_vals=None):
		self.num_vars=num_vars
		self.N=N
		self.coeffs=coeffs
		self.std_devs = std_devs
		if init_vals is not None:
			self.init_vals = init_vals
		else:
			self.init_vals = np.random.uniform(-1, 1, size=[self.init_vals,self.coeffs.shape[0]//self.init_vals])
		self._check_consistency()

	### OUTPUT ###
	## self.data: data generated from AR model with num_vars rows
	##            and N+init_val.shape[1] columns
	def generate_data(self):
		self.data = np.zeros((self.num_vars,self.init_vals.shape[1]+self.N))
		self.data[0:self.num_vars,0:self.init_vals.shape[1]]=self.init_vals
		aug_vector = np.zeros(self.num_vars * self.init_vals.shape[1])
		for n in range(0,self.N):
			#fill augmented vector with past values
			for i in range(0,self.num_vars):
				aug_vector[i*self.init_vals.shape[1]:(i+1)*self.init_vals.shape[1]] = self.data[i,n:n+self.init_vals.shape[1]]
			self.data[:,n+self.init_vals.shape[1]]=np.matmul(self.coeffs, aug_vector) + np.random.normal(0, self.std_devs, self.num_vars)
		return self.data

	#check that the sizes of the various inputs are consistent
	def _check_consistency(self):
		num_vars=self.num_vars
		coeffs=self.coeffs
		std_devs=self.std_devs
		init_vals=self.init_vals
		# check that number of variables is correct in coeff matrix
		assert coeffs.shape[0]==num_vars, "Number of rows in coeff =/= num_vars"

		# check that number of variables is correct in std_devs
		assert std_devs.shape[0]==num_vars, "Length of std_devs =/= num_vars"

		# check that number of variables is correct in init_vals
		assert init_vals.shape[0]==num_vars, "Number of rows in init_vals =/= num_vars"

		# check that enough initial values are provided
		assert init_vals.shape[1]==coeffs.shape[1]//num_vars , "Number of initial values provided not correct"


class AR1:

	def __init__(self, d, b0=None, b1=None, sig=None, y0=None):

		self.dim = d  # dimension of state variable Y

		self.b0 = self._process_args(b0, d, 'b0')
		self.b1 = self._process_args(b1, d, 'b1')
		self.sig = self._process_args(sig, d, 'sig')
		self.y = self._process_args(y0, d, 'y0')

	def __repr__(self):
		return f"<AR(1) Model: d={self.dim}>"

	def __str__(self):
		return "AR(1) : Y = {}\n  b0:\n{}\n  b1:\n{}\n  sig: {}".format(
			np.array_str(self.y, precision=3),
			np.array_str(self.b0, precision=3),
			np.array_str(self.b1, precision=3),
			self.sig)

	def simulate(self, n):
		"""
		Take n steps. Returns ndarray of shape [d, n+1], the history of the state Y 
		including initial and final values.
		"""
		history = np.zeros([self.dim, n+1])
		history[:,0] = self.y
		for i in range(1, n+1):
			history[:, i] = self.step()
		return history	

	def step(self):
		"""
		Take a step based on the AR(1) formula. State Y is updated and returned.
		"""
		eps = np.random.normal(0, self.sig)
		self.y = self.b0 @ np.ones(self.b0.shape[1]) + self.b1 @ self.y + eps
		return self.y

	def _process_args(self, x, d, argname):
		
		if x is None:
			if argname in ['b0', 'b1']:
				return np.random.uniform(-1, 1, size=[d, d])
			elif argname == 'sig':
				return np.ones(d)
			elif argname == 'y0':
				return np.random.uniform(-1, 1, size=d)

		allowed_shapes = {'b0' : [(), (d,), (d,d)],
				  		  'b1' : [(), (d,), (d,d)],
				  		  'sig': [(), (d,)],
				  		  'y0' : [(), (d,)]}
		
		expected = allowed_shapes[argname]
		
		x = np.array(x, dtype=float)  # convert input to a numpy array
		s = x.shape
		
		assert s in expected, f"Shape {s} not allowed for {argname}. Allowed: {expected}."
		
		if argname in ['b0', 'b1']:  # should be matrices of shape (d,d)
			if s == ():
				return x * np.identity(d)
			elif s == (d,):
				return np.diag(x)
		
		if argname in ['sig', 'y0'] and s == ():  # should be arrays of length (d)
			return x * np.ones(d, dtype=float)
		
		return x



def test_ar1():
	n = 100

	b0s 	= [1.0,  	1.0, 	0.1, 	0.0]
	b1s 	= [0.8, 	-0.8, 	1.0, 	1.1]
	sigmas 	= [0.1, 	0.1, 	0.5, 	0.5]
	y0s 	= [4.7, 	0.7, 	4.0, 	4.0]

	ar1 = AR1(4, b0s, b1s, sigmas, y0s)

	print(ar1)
	
	hist = ar1.simulate(n)

	fig, axes = plt.subplots(2,2)
	for i in range(4):
		ax = axes.flatten()[i]
		b0 = b0s[i]
		b1 = b1s[i]
		sig = sigmas[i]

		ax.plot(range(hist.shape[1]), hist[i,:])
		ax.set_xlabel(f'$t$')
		ax.set_ylabel(f'$y_{i}$')
		ax.set_title(f"$b_0={b0}, b_1={b1}, \\sigma={sig}$")

	plt.tight_layout()
	plt.show()


# test_ar1()
