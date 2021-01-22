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

