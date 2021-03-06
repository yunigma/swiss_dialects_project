import numpy as np
from pandas import read_csv

train_set = np.array(read_csv('train.csv'))

indices = np.random.random_integers(
    0, train_set.shape[0], int(train_set.shape[0] * 0.2))

validation_set = []
test_set = []
count = 1

for idx in indices:
    test_set.append("{},{}\t{}".format(
        count, train_set[idx, 0], train_set[idx, 1]))
    validation_set.append("{},{}".format(count, train_set[idx, 1]))
    count += 1

train_set = [",".join(row.astype("str"))
             for idx, row in enumerate(train_set) if idx not in indices]

with open('validation_result.csv', 'w') as file:
    file.write("Id,Prediction\n")
    file.write("\n".join(validation_set))

with open('validation_test.csv', 'w') as file:
    file.write("Id,Text\tGoldlabel\n")
    file.write("\n".join(test_set))

with open('validation_train.csv', 'w') as file:
    file.write("Text,Label\n")
    file.write("\n".join(train_set))
