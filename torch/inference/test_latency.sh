#!/bin/bash

dirname=$1

for i in {1..5}
do 
  echo "start run iter : $i ..."
  ./image_classification_exe $dirname &
  wait
done
