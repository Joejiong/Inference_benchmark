#!/bin/bash

 cd build
 rm -f CMakeCache.txt

 cmake ..
 make

 mv image_classification ../