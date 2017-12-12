# swiss_dialects_project

This is the approach to the kaggle competition [Swiss Dialect Identication](https://www.kaggle.com/c/swiss-dialect-identification) from:

* Iuliia Nigmatulina
* Elias Bernhaut
* Li Tang.

We used 2-6 character n-grams up to word boundaries as features and reduced the set of the used n-grams to the ones which mean the most to the classification.

As a classifier, we made out MLP as the best classifier for this case, although the Support Vector Machines and Random Forest Classifier were also taken into consideration.

# Run it

For running a classification you have to first train and then predict the dataset. This can be simply done by using the predefined shell-script as follows:

```
sh train_and_predict.sh mlp
```

You give as the argument the classifier which you want to use for the prediction.
This call will save a submission as `to_upload.csv` into the current directory. This file can be used for the upload to kaggle.

## Evaluation

### Split the training set!

Before you can perform an evaluation, you have to split the training set (`train.csv`). For this, you can simply run the file `split_set.py`:

```
python2 split_set.py
```

This will create 3 files:

* `validation_result.csv` - Holds the real classes (Same layout as a submission)
* `validation_test.csv` - Holds the test set together with the real classes for proper evaluation
* `validation_train.csv` - Holds the training data

The `train.csv` set was split 80/20%, so the `validation_train.csv` set holds 80% of the previous `train.csv` set, the `validation_test.csv` holds 20% of it.

### Evaluate

If you only want to evaluate the prediction, you can use the shell script `train_and_eval.sh`:

```
sh train_and_eval.sh mlp
```

which will train on the `validation_train.csv` set and evaluate on the `validation_test.csv` set.
