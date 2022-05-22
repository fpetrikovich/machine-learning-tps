#!/bin/bash

echo '' > ./scripts/results_accuracy.txt
echo '' > ./scripts/results_error.txt
echo '' > ./scripts/results_std.txt
echo "0.001 - Linear" $(date -u)
for i in {1..3}
do
    echo "$i" $(date -u)
    python main.py -p 2 -f pics/ -ker linear -c 0.001 -mode dataset -k 5 > ./scripts/tmp.txt
    cat scripts/tmp.txt | grep "Accuracy average" | sed -e 's/.*Accuracy average --> \(.*\).*/\1/' >> ./scripts/results_accuracy.txt
    cat scripts/tmp.txt | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results_error.txt
    cat scripts/tmp.txt | grep "std" | sed -e 's/.*std --> \(.*\).*/\1/' >> ./scripts/results_std.txt
done

echo "--------------------------------" >> ./scripts/results_accuracy.txt
echo "--------------------------------" >> ./scripts/results_error.txt
echo "--------------------------------" >> ./scripts/results_std.txt
echo "0.01 - Linear" $(date -u)
for i in {1..3}
do
    echo "$i" $(date -u)
    python main.py -p 2 -f pics/ -ker linear -c 0.01 -mode dataset -k 5 > ./scripts/tmp.txt
    cat scripts/tmp.txt | grep "Accuracy average" | sed -e 's/.*Accuracy average --> \(.*\).*/\1/' >> ./scripts/results_accuracy.txt
    cat scripts/tmp.txt | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results_error.txt
    cat scripts/tmp.txt | grep "std" | sed -e 's/.*std --> \(.*\).*/\1/' >> ./scripts/results_std.txt
done

echo "--------------------------------" >> ./scripts/results_accuracy.txt
echo "--------------------------------" >> ./scripts/results_error.txt
echo "--------------------------------" >> ./scripts/results_std.txt
echo "0.1 - Linear" $(date -u)
for i in {1..3}
do
    echo "$i" $(date -u)
    python main.py -p 2 -f pics/ -ker linear -c 0.1 -mode dataset -k 5 > ./scripts/tmp.txt
    cat scripts/tmp.txt | grep "Accuracy average" | sed -e 's/.*Accuracy average --> \(.*\).*/\1/' >> ./scripts/results_accuracy.txt
    cat scripts/tmp.txt | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results_error.txt
    cat scripts/tmp.txt | grep "std" | sed -e 's/.*std --> \(.*\).*/\1/' >> ./scripts/results_std.txt
done

echo "--------------------------------" >> ./scripts/results_accuracy.txt
echo "--------------------------------" >> ./scripts/results_error.txt
echo "--------------------------------" >> ./scripts/results_std.txt
echo "1 - Linear" $(date -u)
for i in {1..3}
do
    echo "$i" $(date -u)
    python main.py -p 2 -f pics/ -ker linear -c 1 -mode dataset -k 5 > ./scripts/tmp.txt
    cat scripts/tmp.txt | grep "Accuracy average" | sed -e 's/.*Accuracy average --> \(.*\).*/\1/' >> ./scripts/results_accuracy.txt
    cat scripts/tmp.txt | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results_error.txt
    cat scripts/tmp.txt | grep "std" | sed -e 's/.*std --> \(.*\).*/\1/' >> ./scripts/results_std.txt
done

echo "--------------------------------" >> ./scripts/results_accuracy.txt
echo "--------------------------------" >> ./scripts/results_error.txt
echo "--------------------------------" >> ./scripts/results_std.txt
echo "10 - Linear" $(date -u)
for i in {1..3}
do
    echo "$i" $(date -u)
    python main.py -p 2 -f pics/ -ker linear -c 10 -mode dataset -k 5 > ./scripts/tmp.txt
    cat scripts/tmp.txt | grep "Accuracy average" | sed -e 's/.*Accuracy average --> \(.*\).*/\1/' >> ./scripts/results_accuracy.txt
    cat scripts/tmp.txt | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results_error.txt
    cat scripts/tmp.txt | grep "std" | sed -e 's/.*std --> \(.*\).*/\1/' >> ./scripts/results_std.txt
done

echo "--------------------------------" >> ./scripts/results_accuracy.txt
echo "--------------------------------" >> ./scripts/results_error.txt
echo "--------------------------------" >> ./scripts/results_std.txt
echo "100 - Linear" $(date -u)
for i in {1..3}
do
    echo "$i" $(date -u)
    python main.py -p 2 -f pics/ -ker linear -c 100 -mode dataset -k 5 > ./scripts/tmp.txt
    cat scripts/tmp.txt | grep "Accuracy average" | sed -e 's/.*Accuracy average --> \(.*\).*/\1/' >> ./scripts/results_accuracy.txt
    cat scripts/tmp.txt | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results_error.txt
    cat scripts/tmp.txt | grep "std" | sed -e 's/.*std --> \(.*\).*/\1/' >> ./scripts/results_std.txt
done