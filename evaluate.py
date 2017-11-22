
import sys
import numpy as np
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
