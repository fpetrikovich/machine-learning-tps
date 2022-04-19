#!/bin/bash

echo '' > ./scripts/results15.txt
for i in {1..2}
do
    echo -ne "$i"\\r
    python main.py -f input/german_credit.csv -p 3 -m solve -sal 15 -ss 0.8 -ns 30 -crossk 12 | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results15.txt
done
