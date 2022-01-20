from models import AR
import numpy as np
import matplotlib.pyplot as plt

# create inputs for 1st order with 3 variables
# 2nd variable strongly forces 1st variable and 3rd variable weakly forces 1st variable
# first column corresponds to the constant terms
coeffs = np.array([[1,0,100,1],[1,0,0,0],[1,0,0,0]])
print(coeffs.shape)
std_devs = np.random.uniform(.1,2,3)
init_vals = np.random.uniform(-1,1,(3,1))
N=20
# create AR model object
ar = AR(3,coeffs,std_devs,init_vals)
# generate data using model
ar.generate_data(N)
order=1
ar.causality(order,0.05)

# plot data
plt.plot(ar.data[0])
plt.plot(ar.data[1])
plt.plot(ar.data[2])
plt.show()