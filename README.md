# Image Classification with SVM

In this project we're comparing the image classification performance of SIFT (Scale-Invariant Feature Transform), SURF (Speeded-Up Robust Features) and HOG (Histogram of Gradients). For training and testing we're using images from the [Caltech-256 image dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/).

## Contents
- img: Contains the training data
- matrices: Contains the training and test matrices with the images, and the features they contain
- report.pdf
- 

## Prerequisites:
You'll need OpenCV and OpenCV Contrib 3.4.2.6 for Python:

```
pip install opencv-python==3.4.2.16 opencv-contrib-python==3.4.2.16
```

It's essential to use this version, since newer versions do not contain SIFT and SURF as they're patented.

## Usage:
### Parameters:
- Method: Choose feature extraction method: SIFT SURF or HOG
- Classes: Choose the names of the folder containing the image classes
- K: Choose the number of bins used for clustering (optional. Default: 100)
- Splits: Choose the number of splits (optional. Default: 3)
- Cross Validation: Set to True to use Cross Validation or to false for the opposite (optional. Default: True)
```
python main.py <sift/surf/hog> -c <image classes> -k <number of clusters> -s <number of splits> -cval <True for using CrossValidation>
```

