#necessary imports
import cv2
import matplotlib.pyplot as plt

#image read
#image = cv2.imread('Gura_Portitei_Scara_0025.jpg')
#image = cv2.imread('Gura_Portitei_Scara_010.jpg')
#image = cv2.imread('Gura_Portitei_Scara_020.jpg')
#image = cv2.imread('Gura_Portitei_Scara_040.jpg')
#image = cv2.imread('Gura_Portitei_Scara_080.jpg')
image = cv2.imread('Gura_Portitei_Scara_100.jpg')

#turn the image into grayscale for canny filter
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

#canny for house
#edged = cv2.Canny(gray, 65,255)

#canny for landing site
edged = cv2.Canny(gray, 125,255)
plt.imshow(edged, cmap='gray')
plt.show()

#find the contours
contours, hierarchy= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

#show the outlined areas
plt.imshow(cv2.drawContours(image.copy(), contours, -1, (0,255,0), 2), cmap='gray')
plt.show()
