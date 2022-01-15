import numpy as np
import matplotlib.pyplot as plt

class AR1:

	def __init__(self, d, b0=None, b1=None, sig=None, y0=None):

		self.dim = d  # dimension of state variable Y
		
		# Initialize parameters and state if not given
		if b0 is None:
			self.b0 = np.random.uniform(-1, 1, size=[d,d])
		else:
			assert b0.shape == (d, d), f"Bad shape for b0. Expected ({d},{d}). Got {b0.shape}"
			self.b0 = b0

		if b1 is None:
			self.b1 = np.random.uniform(-1, 1, size=[d,d])
		else:
			assert b1.shape == (d, d), f"Bad shape for b1. Expected ({d},{d}). Got {b1.shape}"
			self.b1 = b1

		if sig is None:
			self.sig = np.ones(d)
		else:
			assert sig.shape == (d,), f"Bad shape for sig. Expected ({d},). Got {sig.shape}"
			self.sig = sig

		if y0 is None:
			self.y = np.random.uniform(-1, 1, size=d)
		else:
			assert y0.shape == (d,), f"Bad shape for y0. Expected ({d},). Got {y0.shape}"
			self.y = y0

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


def test_ar1():
	n = 100

	b0s 	= [1.0,  	1.0, 	0.1, 	0.0]
	b1s 	= [0.8, 	-0.8, 	1.0, 	1.1]
	sigmas 	= [0.1, 	0.1, 	0.5, 	0.5]
	
	y0s 	= [4.7, 	0.7, 	4.0, 	4.0]

	ar1 = AR1(4, np.diag(b0s), np.diag(b1s), np.array(sigmas), np.array(y0s))
	print(ar1)
	
	hist = ar1.simulate(n)

	fig, axes = plt.subplots(2,2)
	for i in range(4):
		ax = axes.flatten()[i]
		b0 = b0s[i]
		b1 = b1s[i]
		sig = sigmas[i]

		ax.plot(range(hist.shape[1]), hist[i,:])
		ax.set_xlabel(r'$t$')
		ax.set_ylabel(r'$y_{i}$')
		ax.set_title(f"$b_0={b0}, b_1={b1}, \\sigma={sig}$")

	plt.tight_layout()
	plt.show()


test_ar1()
