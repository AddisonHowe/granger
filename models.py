import numpy as np
import matplotlib.pyplot as plt

class AR1:

	def __init__(self, d, b0=None, b1=None, sig=None, y0=None):
		"""
		d: dimension of state variable Y
		b0: constant, length d list or array, or d by d matrix; b0 coefficient matrix
		b1: constant, length d list or array, or d by d matrix; b1 coefficient matrix
		sig: constant or length d list or array; standard deviations for each state component
		y0: constant or length d list or array; initial state
		"""

		self.dim = d

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

	def generate_data(self, n):
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
	
	hist = ar1.generate_data(n)

	fig, axes = plt.subplots(2,2)
	for i in range(4):
		ax = axes.flatten()[i]
		b0 = b0s[i]
		b1 = b1s[i]
		sig = sigmas[i]
		data = hist[i, :]

		mu = np.mean(data)
		sd = np.std(data)
		ax.plot(range(len(data)), data, label=f"data\n$\\mu={mu:.2f}$\n$\\sigma={sd:.2f}$")
		ax.set_xlabel(f'$t$')
		ax.set_ylabel(f'$y$')
		ax.set_title(f"$b_0={b0}, b_1={b1}, \\sigma={sig}$")
		ax.legend()

	plt.tight_layout()
	plt.show()


test_ar1()
