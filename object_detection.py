import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

import edge_filter_sobel
import convolution_from_scratch
import box_filter

image = cv2.imread('Gura_Portitei_Scara_100.jpg')
print(image.shape)
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.show()

kernel = box_filter.boxfilter(3)
image_blur = convolution_from_scratch.conv(image, kernel)
plt.imshow(image_blur, cmap='gray')
plt.show()


image_res,image_thresh = cv2.threshold(image_blur,220,255,cv2.THRESH_BINARY_INV)
plt.imshow(image_thresh, cmap='gray')
plt.show()
image_thresh = edge_filter_sobel.sobelFilt(image_thresh)
plt.imshow(image_thresh, cmap='gray')
plt.show()

image_thresh = image_thresh.astype(np.uint8)

dist_transform = cv2.distanceTransform(image_thresh,cv2.DIST_C,5)
#print("dist", dist_transform)
ret, last_image =  cv2.threshold(dist_transform, 0.3*dist_transform.max(),255,0)
plt.imshow(last_image, cmap='gray')
plt.show()

last_image = np.uint8(last_image)
#plt.imshow(last_image, cmap='gray')
#plt.show()

cnts = cv2.findContours(last_image.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#print(cnts)


for (i, c) in enumerate(cnts):
    ((x, y), _) = cv2.minEnclosingCircle(c)
    cv2.putText(image, "#{}".format(i + 1), (int(x) - 45, int(y) + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

plt.imshow(image, cmap='gray')
plt.show()
