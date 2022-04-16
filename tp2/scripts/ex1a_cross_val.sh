#!/bin/bash

echo '' > ./scripts/results.txt
for i in {1..25}
do
    echo -ne "$i"\\r
   python main.py -f input/german_credit.csv -p 1 -crossk 10 -m solve | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results.txt
done
