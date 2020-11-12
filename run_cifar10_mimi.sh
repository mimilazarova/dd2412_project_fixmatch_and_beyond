#!/bin/bash

export dir=/home/mimiglazarova/data

for size in 250 10; do
    for seed in 3 5 1; do
	      python3 main.py $dir/SSL2 cifar10 $seed $size 10 $dir >> cifar10-$seed-$size-mimi.log 
    done
done
