#!/bin/sh

python baseline.py --train --model model_$1.pkz --data train.csv --verbose --classifier $1
python baseline.py --predict --samples test.csv --model model_$1.pkz > to_upload.csv