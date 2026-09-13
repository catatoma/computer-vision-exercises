#necessary imports
import numpy as np
import convolution_fromscratch
import matplotlib.pyplot as plt

def sobelFilt(image):
    #here are obtained the vertical edges
    xrespect = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    newimagex = convolution_fromscratch.conv(image,xrespect)
    plt.imshow(newimagex, cmap='gray')
    plt.show()

    # here are obtained the horizontal edges
    yrespect = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    newimagey = convolution_fromscratch.conv(image, yrespect)
    plt.imshow(newimagey, cmap='gray')
    plt.show()

    #compute both vertical and horizontal edges together the using approximation of the gradient
    g_xy = np.sqrt(pow(newimagex,2) + pow(newimagey,2))

    #return the final result
    return g_xy