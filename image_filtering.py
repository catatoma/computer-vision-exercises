import boxfilter
import convolution_fromscratch
import gaussianfilter
import edgetfilterSobel

import cv2
import matplotlib.pyplot as plt

#different dimensions for boxfilter: 3,7,9,15,25,51
kernel = boxfilter.boxfilter(25)

#different parameters for gaussian filter: (1,3), (1,9), (1,11), (7,11),(25,51)
kernel_2 = gaussianfilter.gaussfilt(30,53)


#try for each image
#image = cv2.imread('Gura_Portitei_Scara_0025.jpg')
#image = cv2.imread('Gura_Portitei_Scara_010.jpg')
#image = cv2.imread('Gura_Portitei_Scara_020.jpg')
#image = cv2.imread('Gura_Portitei_Scara_040.jpg')
#image = cv2.imread('Gura_Portitei_Scara_080.jpg')
image = cv2.imread('Gura_Portitei_Scara_100.jpg')
plt.imshow(image)
plt.show()

#save the result in a new image
#newimage = convolution_fromscratch.conv(image,kernel)

#newimage_2 = convolution_fromscratch.conv(image,kernel_2)

newimage_3 = edgetfilterSobel.sobelFilt(image)

#show the image after filtering
plt.imshow(newimage_3, cmap ='gray')
plt.show()


#plt.imshow(newimage_2, cmap ='gray')
#plt.show()

#plt.imshow(newimage_3, cmap ='gray')
#plt.show()