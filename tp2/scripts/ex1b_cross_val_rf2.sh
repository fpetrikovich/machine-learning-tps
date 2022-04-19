#!/bin/bash

# echo '' > ./scripts/results10.txt
# for i in {1..5}
# do
#     echo -ne "$i"\\r
#     python main.py -f input/german_credit.csv -p 3 -m solve -sal 5 -ss 0.5 -ns 10 -crossk 12 | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results10.txt
# done

# echo '' > ./scripts/results20.txt
# for i in {1..5}
# do
#     echo -ne "$i"\\r
#     python main.py -f input/german_credit.csv -p 3 -m solve -sal 5 -ss 0.5 -ns 20 -crossk 12 | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results20.txt
# done

# echo '' > ./scripts/results30.txt
# for i in {1..5}
# do
#     echo -ne "$i"\\r
#     python main.py -f input/german_credit.csv -p 3 -m solve -sal 5 -ss 0.5 -ns 30 -crossk 12 | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results30.txt
# done

echo '' > ./scripts/results40.txt
for i in {1..5}
do
    echo -ne "$i"\\r
    python main.py -f input/german_credit.csv -p 3 -m solve -sal 5 -ss 0.5 -ns 40 -crossk 12 | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results40.txt
done

echo '' > ./scripts/results50.txt
for i in {1..5}
do
    echo -ne "$i"\\r
    python main.py -f input/german_credit.csv -p 3 -m solve -sal 5 -ss 0.5 -ns 50 -crossk 12 | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results50.txt
done

echo '' > ./scripts/results60.txt
for i in {1..5}
do
    echo -ne "$i"\\r
    python main.py -f input/german_credit.csv -p 3 -m solve -sal 5 -ss 0.5 -ns 60 -crossk 12 | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results60.txt
done

echo '' > ./scripts/results70.txt
for i in {1..5}
do
    echo -ne "$i"\\r
    python main.py -f input/german_credit.csv -p 3 -m solve -sal 5 -ss 0.5 -ns 70 -crossk 12 | grep "Error average" | sed -e 's/.*Error average --> \(.*\).*/\1/' >> ./scripts/results70.txt
done