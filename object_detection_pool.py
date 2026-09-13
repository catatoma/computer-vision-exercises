#necessary imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

#image read
#image = cv2.imread('Gura_Portitei_Scara_0025.jpg')
#image = cv2.imread('Gura_Portitei_Scara_010.jpg')
#image = cv2.imread('Gura_Portitei_Scara_020.jpg')
#image = cv2.imread('Gura_Portitei_Scara_040.jpg')
#image = cv2.imread('Gura_Portitei_Scara_080.jpg')
image = cv2.imread('Gura_Portitei_Scara_100.jpg')

#different blurr for each resolution
#blurr used to get rid of things like reflective water
image_blur = cv2.medianBlur(image,41)#1, 5, 9, 15, 21, 33
plt.imshow(image_blur, cmap='gray')
plt.show()

#turn the image into grayscale for thresholding opperation
image_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
#plt.imshow(image_blur_gray, cmap='gray')
#plt.show()

#chose the interest treshold and binarize the image - 190 for the intensity of the blue pool
image_res ,image_thresh = cv2.threshold(image_blur_gray,190,255,cv2.THRESH_BINARY_INV)
#plt.imshow(image_thresh, cmap='gray')
#plt.show()

#filtering in order to get rid of some small black or white areas that are not of interest
kernel = np.ones((43,43),np.uint8)#(1,1), (3,3), (3,3), (11,11), (), ()
opening = cv2.morphologyEx(image_thresh,cv2.MORPH_OPEN,kernel)
plt.imshow(opening, cmap='gray')
plt.show()

#find the contours
contours, hierarchy= cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

#show the outlined areas
plt.imshow(cv2.drawContours(image.copy(), contours, -1, (0,255,0), 7), cmap='gray')
plt.show()
