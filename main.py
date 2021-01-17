import pandas as pd
import numpy as np

import xgboost as xgb
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, log_loss



train = pd.read_csv('1_name_train.csv')
test = pd.read_csv('1_name_test.csv')
target = train['gender']
train = train.values

def transform_test(cols):
    if isinstance(cols['first_name'], (str)):
        return cols['first_name'].lower().split()[0]
    else:
        return ''

test['first_name'] = test.apply(transform_test, axis=1)
test = test.values

print(train)
print(test)

TRAIN_SPLIT = 0.8

def features(name):
    name = name.lower()
    return {
        'first-letter': name[0],
        'first2-letters': name[0:2],
        'first3-letters': name[0:3],
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
        'all-letters': name[:]
    }
print("Example", features("joy"))

features = np.vectorize(features)
print("Example", features(["isaac", "ijeoma", "paul"]))

X = features(train[:, 0])
print(X)
y = train[:, 1] 
print(y)

X, y = shuffle(X, y)
X_train, X_val = X[:int(TRAIN_SPLIT * len(X))], X[int(TRAIN_SPLIT * len(X)):]
y_train, y_val = y[:int(TRAIN_SPLIT * len(y))], y[int(TRAIN_SPLIT * len(y)):]
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

vectorizer = DictVectorizer()
vectorizer.fit(X_train)

transformed = vectorizer.transform(features(["Martha", "John"]))
print("Example", transformed)

xgb_param_grid = {'learning_rate': [0.01,0.1],
                 'n_estimators':[500, 1500],
                 'subsample':[0.3, 0.7],
                 'colsample_bytree': [0.3, 0.7],
                 'max_depth': [2, 5],
                 'min_child_weight': [1, 5, 15],
                 'objective':['binary:logistic'],
                 'seed': [1337]}
xgb = xgb.XGBClassifier()
eval_set = [(vectorizer.transform(X_val), y_val)]
xgbm = GridSearchCV(estimator=xgb,param_grid=xgb_param_grid, scoring='neg_log_loss', cv=3, verbose=5,
                    n_jobs = -1)

xgbm.fit(vectorizer.transform(X_train),y_train, eval_metric="logloss", 
         early_stopping_rounds=5, eval_set=eval_set, verbose=5)

print(xgbm.best_score_, xgbm.best_estimator_, xgbm.best_params_)

for name in test:
    print(name[0], ':', xgbm.predict(vectorizer.transform(features(name)))[0], 
           xgbm.predict_proba(vectorizer.transform(features(name)))[0])
           

# save model and vectorizer
with open('gender_classifier.pickle', 'wb') as f:
    pickle.dump(xgbm, f)

with open('dict_vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
