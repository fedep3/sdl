import scipy.io

mat = scipy.io.loadmat('data2.mat')

#Variable types
mat['Info'][0][0][1][0]

#Variable values
mat['Info'][0][0][0][0]
