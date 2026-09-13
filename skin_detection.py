import cv2
import matplotlib.pyplot as plt
import numpy as np

for i in range(10):
    num=i+1
    img = cv2.imread('images/'+str(num)+'.jpg')
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # threshold the image based on a certain hue range, in order to get regions containing human skin
    hsv_lower = np.asarray([0, 30, 100])
    hsv_upper = np.asarray([15, 180, 255])
    mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)

    # remove the smallest, noisy connected components
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # dilate and erode the remaining blobs
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(opening, kernel, iterations=2)
    erode = cv2.erode(dilate, kernel, iterations = 2)

    # get contours
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draw the contour on the image
    image = original.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 20000:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('HSV Mask')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(opening, cmap='gray')
    axes[1, 0].set_title('Removing blobs')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(erode, cmap='gray')
    axes[1, 1].set_title('After dilation and erosion')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

    plt.imshow(image)
    plt.title('Skin detection')
    plt.show()


