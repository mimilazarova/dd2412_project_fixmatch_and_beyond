#!/bin/bash

export dir=/home/mimiglazarova/data

for size in 250; do
    for seed in 3 5 1; do
	python3 main.py $dir/SSL2 svhn $seed $size 10 $dir >> svhn-$seed-$size.log 
    done
done

