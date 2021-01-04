# object_recognition

Set up:

```
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git
 
$ cd opencv
$ mkdir build && cd build
 
$ cmake -DCMAKE_BUILD_TYPE=Release \
    -D OPENCV_ENABLE_NONFREE:BOOL=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=ON \
    -D HAVE_opencv_python3=ON \
    -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
    ..
 
$ make -j4
$ sudo make install
```


Feature detection with SURF, SVC with RBF kernel (takes much longer than with linear kernel)
```
              precision    recall  f1-score   support
      cactus       0.39      0.66      0.49     13023
    ketch101       0.41      0.29      0.34      6224
     raccoon       0.32      0.22      0.26      9231
   spaghetti       0.30      0.22      0.26      9973
       sushi       0.39      0.31      0.35      9830
    accuracy                           0.37     48281
   macro avg       0.36      0.34      0.34     48281
weighted avg       0.36      0.37      0.35     48281
```


Feature detection with SURF, linear SVC with linear kernel 
```
              precision    recall  f1-score   support
      cactus       0.35      0.77      0.49     13149
    ketch101       0.31      0.23      0.26      6203
     raccoon       0.29      0.06      0.10     10955
   spaghetti       0.33      0.16      0.22     10061
       sushi       0.33      0.32      0.33      8389
    accuracy                           0.34     48757
   macro avg       0.32      0.31      0.28     48757
weighted avg       0.33      0.34      0.29     48757
```