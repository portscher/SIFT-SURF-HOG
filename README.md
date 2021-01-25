# Image Classification with SVM

In this project we're comparing the image classification performance of SIFT (Scale-Invariant Feature Transform), SURF (Speeded-Up Robust Features) and HOG (Histogram of Gradients). For training and testing we're using images from the [Caltech-256 image dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/).

## Prerequisites:
You'll need OpenCV and OpenCV Contrib 3.4.2.6 for Python:

```
pip install opencv-python==3.4.2.16 opencv-contrib-python==3.4.2.16
```

It's essential to use this version, since newer versions do not contain SIFT and SURF as they're patented.

## Usage:
```
python main.py <sift/surf/hog> -c <image classes> -k <number of clusters>
```
