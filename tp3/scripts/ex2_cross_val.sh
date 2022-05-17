#!/bin/bash

echo '' > ./scripts/results_accuracy.txt
echo '' > ./scripts/results_error.txt
echo '' > ./scripts/results_std.txt
for i in {1..5}
do
    echo -ne "$i"\\r
    python main.py -p 2 -f pics/ -ker linear -c 0.01 -mode dataset -k 20 > ./scripts/tmp.txt
    cat scripts/tmp.txt | grep "Accuracy average" | sed -e 's/.*Accuracy average --> \(.*\).*/\1/' >> ./scripts/results_accuracy.txt
    cat scripts/tmp.txt | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results_error.txt
    cat scripts/tmp.txt | grep "std" | sed -e 's/.*std --> \(.*\).*/\1/' >> ./scripts/results_std.txt
done