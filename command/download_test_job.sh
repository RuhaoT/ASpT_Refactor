#!/bin/bash

cd ..
mkdir data
cd data

# step1: download the small matrices
# web-Stanford.tar.gz is a small matrix
# wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Stanford.tar.gz
# web-Google.tar.gz is a small matrix
# wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz

# step2: extract the matrices
find . -name '*.tar.gz' -exec tar xvf {} \;
rm *.tar.gz

# step3: convert the matrices 
# the 'conv.c' here convert all kinds of matrices to real/symetric matrices
cp ../misc/conv.c .
gcc -O3 -o conv conv.c

for i in `ls -d */`
do
cd ${i}
ii=$(echo "$i" | sed 's/\///g')
mv ${ii}.mtx ${ii}.mt0
../conv ${ii}.mt0 ${ii}.mtx 
rm ${ii}.mt0
cd ..
done

cd ..
cd command
