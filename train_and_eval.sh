#!/bin/sh

python baseline.py --train --model model_$1.pkz --data validation_train.csv --verbose --classifier $1
python baseline.py --evaluate --samples validation_test.csv --model model_$1.pkz