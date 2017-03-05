#!/bin/bash
echo "n,CPU,GPU" >> bench.csv
for n in {1..26};
do 
    ./radixsort -s 1 -n $n >> bench.csv
done
