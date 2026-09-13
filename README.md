# Computer Vision Exercises

A collection of computer vision exercises and implementations developed while learning the fundamentals of image processing and computer vision.

The goal of this repository is to understand the concepts behind common computer vision techniques by implementing some of them from scratch using Python and NumPy, while also using OpenCV for comparison and higher-level operations.

## Topics Covered

* Image representation and manipulation
* Grayscale image conversion
* 2D convolution
* Image filtering
* Box filtering
* Sobel edge detection
* Image thresholding
* Gradient magnitude
* Distance transforms
* Contour detection
* Basic object detection

## Exercises

### Convolution From Scratch

Implementation of 2D image convolution using NumPy.

The implementation handles:

* Kernel-based image filtering
* Zero padding
* Preservation of the original image dimensions
* Conversion of color images to grayscale

### Box Filter

Implementation of a box (mean) filter and application of the filter through the custom convolution implementation.

### Sobel Edge Detection

Implementation of the Sobel edge detection algorithm from scratch.

Two Sobel kernels are applied to obtain:

* Vertical edges
* Horizontal edges

The final gradient magnitude is calculated using:

```text
G = √(Gx² + Gy²)
```

### Basic Object Detection

A simple object-detection pipeline combining several image-processing techniques:

1. Image smoothing
2. Binary thresholding
3. Sobel edge detection
4. Distance transformation
5. Thresholding
6. Contour detection
7. Object visualization

### Skin Detection using HSV Color Space

Segments regions corresponding to human skin using HSV color thresholding. The resulting mask is refined using morphological opening, dilation, and erosion. Contours are then detected and regions above a specified area threshold are highlighted with bounding boxes.

## Technologies

* Python
* NumPy
* OpenCV
* Matplotlib


## Learning Goals

These exercises are intended to build a fundamental understanding of how image-processing algorithms work internally rather than relying exclusively on high-level computer vision functions.
