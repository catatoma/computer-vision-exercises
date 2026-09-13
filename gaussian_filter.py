#necessary imports
from numpy import pi,exp,mgrid

def gaussfilt(sigma,dim):

    #create x, y each of dim x dim dimension
    x, y = mgrid[-(dim//2): dim//2+1, -(dim//2): dim//2+1]

    #create the actual filter(the matrix) using the formula for Gaussian filter
    kernel = 1/(2*pi*pow(sigma,2))*exp(-(pow(x,2)+pow(y,2))/(2*pow(sigma,2)))

    #return the matrix
    return kernel
