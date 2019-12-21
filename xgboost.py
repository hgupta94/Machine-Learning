# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:36:47 2019

@author: hirsh
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:30:42 2019

@author: hirsh
"""


import pandas as pd
import numpy as np

train = pd.read_csv('MLProject_train.csv')
valid = pd.read_csv('MLProject_valid.csv')
test = pd.read_csv('MLProject_test.csv')

train = train.dropna()
valid = valid.dropna()
test = test.dropna()


#########################
#####    XG Boost   #####
#########################
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

train_sub = train.sample(n=100000, random_state=1)

# Separate predictors and target
X_train = train_sub.drop(['target1', 'target2'], axis=1)
X_valid = valid.drop(['target1', 'target2'], axis=1)

y1_train = train_sub['target1']
y2_train = train_sub['target2']

y1_valid = valid['target1']
y2_valid = valid['target2']


# Predict target1
data_matrix1 = xgb.DMatrix(data=X_train, label=y1_train)

xg_reg1 = xgb.XGBRegressor(objective = 'binary:logistic',
                          colsample_bytree = 0.85,
                          learning_rate = 0.15,
                          max_depth = 3,
                          n_estimators = 15)

xg_reg1.fit(X_train, y1_train)
preds1 = xg_reg1.predict(X_valid)


fpr1,tpr1,thresh1 = metrics.roc_curve(y1_valid, preds1)
roc_auc1 = metrics.auc(fpr1, tpr1)
roc_auc1


preds_test1 = xg_reg1.predict(test)



# Predict target2
data_matrix2 = xgb.DMatrix(data=X_train, label=y2_train)

xg_reg2 = xgb.XGBRegressor(objective = 'binary:logistic',
                          colsample_bytree = 0.75,
                          learning_rate = 0.25,
                          max_depth = 3,
                          n_estimators = 15)


xg_reg2.fit(X_train, y2_train)
preds2 = xg_reg2.predict(X_valid)


fpr2,tpr2,thresh2 = metrics.roc_curve(y2_valid, preds2)
roc_auc2 = metrics.auc(fpr2, tpr2)
roc_auc2

preds_test2 = xg_reg2.predict(test)


xgb_preds_final = pd.DataFrame(data = [preds_test1, preds_test2])
xgb_preds_final = np.transpose(xgb_preds_final)

xgb_preds_final.columns = ['target1', 'target2']
xgb_preds_final['row'] = xgb_preds_final.index + 1
xgb_preds_final = xgb_preds_final[['row', 'target1', 'target2']]
xgb_preds_final['target1'] = round(xgb_preds_final['target1'],2)
xgb_preds_final['target2'] = round(xgb_preds_final['target2'],2)
xgb_preds_final.to_csv('xgb_predictions.csv', index=False)






















