from models import AR
import numpy as np
import matplotlib.pyplot as plt

# create inputs for 3rd order with 3 variables
coeffs = np.random.uniform(-.5,.5,(3,9))
std_devs = np.random.uniform(.1,.5,3)
init_vals = np.random.uniform(-1,1,(3,3))
N=30
# create AR model object
ar = AR(3,N,coeffs,std_devs,init_vals)
# generate data using model
ar.generate_data()
# plot data
plt.plot(ar.data[0])
plt.plot(ar.data[1])
plt.plot(ar.data[2])
plt.show()