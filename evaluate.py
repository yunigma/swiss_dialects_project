
import sys
import numpy as np
import operator
from pandas import read_csv

result_file = sys.argv[1]
actual_result_file = sys.argv[2]

results = np.array(read_csv(result_file))
actual_results = np.array(read_csv(actual_result_file))

accurate = 0

for idx in range(actual_results.shape[0]):
    if results[idx, 1] == actual_results[idx, 1]:
        accurate += 1

print("Accuracy: {:.2f}%".format(
    float(accurate) / actual_results.shape[0] * 100))

errors = dict()
for idx in range(actual_results.shape[0]):
    if results[idx, 1] != actual_results[idx, 1]:
        key = (results[idx, 1], actual_results[idx, 1])
        if key not in errors:
            errors[key] = 1
        else:
            errors[key] += 1

print

for key, value in sorted(errors.items(), key=operator.itemgetter(1), reverse=True):
    print("{} -> {}: {}".format(key[0], key[1], value))

with open("validation_test.csv") as val_set:
    val_set.readline()
    validation_set = np.array([s.replace('\t', ',').split(',')
                               for s in val_set])

for idx in range(actual_results.shape[0]):
    if results[idx, 1] == "LU" and actual_results[idx, 1] != "LU" or \
            actual_results[idx, 1] == "LU" and results[idx, 1] != "LU":
        print "Should: {}; Is: {} -- {}".format(actual_results[idx, 1], results[idx, 1], validation_set[idx, 1])
