import csv
from pprint import pprint
import matplotlib.pyplot as Plot
import seaborn as sns
import pandas as pd
from IPython.display import display, HTML

pd.set_option('display.max_columns', 50)
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LogisticRegression

def isOver(row):
    return 0 if row[len(learndf.columns)-1] == '- 50000.' else 1

def validate(df):
    assert isinstance(df, pd.DataFrame)
    for col in df:
        if df[col].isnull().sum() > 0:
            print("Error NaN detected for {}!".format(col))
            return False
    print("No NaNs.")
    return True

def GetPrediction(probabilities):
    size = len(probabilities)
    predictions = np.empty(size)
    for i in range(size):
        predictions[i] = 1 if probabilities[i] > 0.5 else 0
    return predictions


def print_scores(model, X_test, y_true, y_pred):
    if y_pred.dtype == np.float16 or y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        y_pred = GetPrediction(y_pred)
    acc_score_norm = metrics.accuracy_score(y_true, y_pred)
    acc_score_non_norm = metrics.accuracy_score(y_true, y_pred, normalize=False)
    print('Acc norm: {} Acc non-norm: {}'.format(acc_score_norm, acc_score_non_norm))
    ce_score_norm = metrics.log_loss(y_true, y_pred)
    ce_score_non_norm = metrics.log_loss(y_true, y_pred, normalize=False)
    print('CE norm: {} CE non-norm: {}'.format(ce_score_norm, ce_score_non_norm))
    matthews = metrics.matthews_corrcoef(y_true, y_pred)
    print('Matthews Cor. Coef: {}'.format(matthews))
    scores = get_roc_auc(model, X_test, y_true, y_pred)
    print('roc_auc: {} <- {}'.format(np.average(scores), scores))

def get_roc_auc(model, X_test, y_true):
    scores = cross_val_score(model, X_test, y=y_true, scoring='roc_auc', n_jobs=-1)
    return scores

learndf = pd.read_csv("census_income_learn.csv", header = None, skipinitialspace = True, na_values= "Not in universe")
testdf = pd.read_csv("census_income_test.csv", header = None, skipinitialspace = True, na_values= "Not in universe")
y_train = pd.DataFrame()
y_train['IsOver'] = learndf.apply(isOver, axis=1)
y_test = pd.DataFrame()
y_test['IsOver'] = testdf.apply(isOver, axis=1)


model = LogisticRegression()
X_train = pd.DataFrame()
X_test = pd.DataFrame()
roc_auc = 0

for col in learndf:
    if learndf[col].dtype.name == 'int64':
        col_train = pd.DataFrame({col : learndf[col]})
        col_test = pd.DataFrame({col : testdf[col]})
    elif learndf[col].dtype.name == 'object':
        col_train = pd.get_dummies(learndf[col])
        col_test = pd.get_dummies(testdf[col])
    else:
        print('bad type')
        () + 1
    assert isinstance(col_train, pd.DataFrame)
    assert isinstance(col_test, pd.DataFrame)
    for newcol in col_train:
        if newcol not in col_test.columns:
            col_test[newcol] = 0
    print(len(col_train))
    print(len(col_test))
    new_X_train = pd.concat([X_train, col_train], axis=1)
    new_X_test = pd.concat([X_test, col_test], axis=1)
    model.fit(new_X_train, y_train.IsOver.ravel())
    y_pred = model.predict(new_X_test)
    scores = get_roc_auc(model, new_X_test, y_test.IsOver.values, y_pred)
    new_roc_auc = np.average(scores)
    print('{} roc_auc: {} <- {}'.format(col, np.average(scores), scores))
    if new_roc_auc > roc_auc:
        X_train = new_X_train
        X_test = new_X_test
        roc_auc = new_roc_auc

print_scores(model, X_test, y_test.IsOver.values, y_pred)