#necessary imports
import numpy as np

def conv(image,kernel):
    #transform the RGB image in a grayscale one
    if image.ndim == 3 and image.shape[2] == 3:
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        image = 0.2989 * R + 0.5870 * G + 0.1140 * B

    #grab the dimensions needed for iterating through the image
    dimKer = kernel.shape[0]
    rowImg = image.shape[0]
    colImg = image.shape[1]

    #add a 0 pad to the image in order to keep the initial dimensions after convolution
    pad_dim = (dimKer-1)//2
    image = np.pad(image, pad_dim, mode='constant')

    #variable to store the new image
    output = np.zeros((rowImg, colImg))

    #iterate through image, do the actual convolution, save the result for each conv in output
    for i in np.arange(pad_dim, rowImg + pad_dim):
        for j in np.arange(pad_dim, colImg + pad_dim):
            part_from_image = image[i - pad_dim:i + pad_dim + 1, j - pad_dim:j + pad_dim + 1]
            actual_conv = (part_from_image * kernel).sum()
            output[i - pad_dim, j - pad_dim] = actual_conv

    #return the result
    return output