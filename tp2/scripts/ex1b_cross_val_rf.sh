#!/bin/bash

echo '' > ./scripts/results.txt
for i in {1..25}
do
    echo -ne "$i"\\r
    python main.py -f input/german_credit.csv -p 3 -m solve -eu 3 -gu 0.01 -hl 3 -sal 10 -ss 0.8 -ns 10 -crossk 7 | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results.txt
done
