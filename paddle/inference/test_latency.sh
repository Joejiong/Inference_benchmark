#!/bin/bash

batch_size=$1
dirname=$2

for i in {1..5}
do
  echo "start iter : $i ..."
  ./image_classification --use_gpu --batch_size=$batch_size --repeat_time=100 --dirname=$dirname &
  wait
done
