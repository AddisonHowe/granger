import numpy as np
class AR():
    def __init__(self,order,num_vars,coeffs,std_devs,init_vals,N):
        # if order is a constant
        self.order=order
        self.num_vars=num_vars
        self.coeffs = coeffs
        self.std_devs = std_devs
        self.init_vals = init_vals
        self.N = N

    #generate the data using the AR models with the specified parameters
    def generate_data(self):
        data = np.zeros(num_vars,N)
        data[:,0] = self.init_vals
        for t in range(1,N):
            data_mat = np.zeros(order*)
            data[t,:] = np.matmul(coeffs,data_mat)

        return data

    # return mean and std dev of generated data
    def get_stats(self):
        pass

