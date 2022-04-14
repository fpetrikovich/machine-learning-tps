#!/bin/bash

echo '' > ./scripts/results.txt
for i in {1..50}
do
    echo -ne "$i"\\r
   python main.py -f input/reviews_sentiment.csv -p 2 -k 5 -crossk 5 -m simple -sm solve | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results.txt
done