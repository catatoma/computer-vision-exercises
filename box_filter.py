#necessary imports
import numpy as np

def boxfilter(dim):
    #initialize a matrix with all elements = 1
    box = np.ones((dim,dim))

    #use the 1 matrix to create the actual filter
    kernel = box/pow(dim,2)

    #return the new matrix
    return kernel
