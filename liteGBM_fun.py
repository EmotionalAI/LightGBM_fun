# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:08:34 2020

@author: pkrzywkowski

Fun with lightGBM model from Microsoft
You can use it for your categorical heavy dataset
it handles them directly no need to one hot encode
You can also try CatBoost 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMRegressor
from lightgbm import plot_importance
#%% read data
df = pd.read_csv(r'your_data', sep=',')
df.info()
#%%
#filter based on number of X
low = df['column_you_want_to_filter_on'].value_counts()
dfl = df[df['column_you_want_to_filter_on'].isin(low[low > 9].index)]
#%%convert object columns into categorical
for c in dfl.columns:
    col_type = dfl[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        dfl[c] = dfl[c].astype('category')
dfl.info()

#%% split data into train/test
target = dfl['Your_target_column']
X = dfl.drop('Your_target_column', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, target, test_size=.25, random_state=2)

#%% training part
#%% early stopping
fit_params={"early_stopping_rounds":40, 
            "eval_metric" : 'l2_root', 
            "eval_set" : [(x_test,y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)], # you can define lerining rate dacay if you want to use it
            'verbose': 50,
            'categorical_feature': 'auto'}
#%%Hyperparameters search -> define parameters space
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 60), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'min_split_gain' : sp_uniform(loc=0.0, scale=0.2),
             'bagging_fraction' : sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
#%% parameters search
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
HP_points_to_test = 200
#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
gbm = lgb.LGBMRegressor(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=-1, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=gbm, param_distributions=param_test, 
    n_iter=HP_points_to_test,
    scoring='r2',
    cv=5,
    refit=True,
    random_state=314,
    verbose=True)
#%% run parameter search
gs.fit(x_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

#%% best parameters so far (dart is for dart boosting type) You should run search for your problem
opt_params = {'bagging_fraction': 0.6502403997538188, 'colsample_bytree': 0.48542760812934804, 'min_child_samples': 120, 'min_child_weight': 1, 'min_split_gain': 0.14056022471870622, 'num_leaves': 45, 'reg_alpha': 2, 'reg_lambda': 20, 'subsample': 0.986425399672787} 
opt_dart = {'bagging_fraction': 0.6728664524779502, 'colsample_bytree': 0.5724726792874275, 'min_child_samples': 149, 'min_child_weight': 1, 'min_split_gain': 0.1989813111606026, 'num_leaves': 36, 'reg_alpha': 0.1, 'reg_lambda': 1, 'subsample': 0.9145266399346188} 

gbm_final = lgb.LGBMRegressor(**gs.best_estimator_.get_params())
gbm_final.set_params(**opt_params)
gbm_final.fit(x_train, y_train, **fit_params)


'''This is when you already have HP ready and want to skip the search'''
#%%model with traditional tree boost
gbm = lgb.LGBMRegressor(n_jobs=-1, max_depth=-1, n_estimators=2000, **opt_params)
gbm.fit(x_train, y_train, **fit_params)

#%% model using dart boosting
gbmd = lgb.LGBMRegressor(n_jobs=-1, 
                           n_estimators=450,
                           boosting_type='dart',
                           **opt_dart)
gbmd.fit(x_train, y_train)

#%% performance metrics
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
y_pred = gbm_final.predict(x_test)
lgb_mse = mean_squared_error(y_test, y_pred)
print('MSE score: ', lgb_mse)
print('RMSE score: ', np.sqrt(lgb_mse))
lgb_r2 = r2_score(y_test, y_pred)
print('r2 score: ', lgb_r2)
lgb_MAE = median_absolute_error(y_test, y_pred)
print('MAE score: ', lgb_MAE)

#%% feature importances
feat_imp = pd.Series(gbm.feature_importances_, index=X.columns)
feat_imp.nlargest(35).plot(kind='barh', figsize=(8,10))

feat_imp.sort_values(inplace=True, ascending=False)
plt.figure(figsize=(24,16))
ax = sns.barplot(y=feat_imp.index, x=feat_imp, orient='h', color='#00A3E0')
ax.tick_params(labelsize=30)
sns.despine()

#%%scatterplot prediction vs reality
y_pred1 = pd.DataFrame(y_pred)
predictions = pd.DataFrame(pd.np.column_stack([y_pred1, y_test]), columns=['Prediction', 'Y'])

plt.figure(figsize=(10,10))
ax = sns.scatterplot(predictions['Prediction'], predictions['Y'])
ax.set(xlabel='Predictions', ylabel='Y', xlim=(0.95,6), ylim=(0.95,6))

#%% shapley values starter if you want to look deeper into your model 
import shap
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(x_train)
shap_feat = pd.DataFrame(shap_values, columns=x_train.columns)
shap_feat.info()
shap_feat['impact_on_mean'] = shap_feat.sum(axis=1)

shap.summary_plot(shap_values, x_train)
shap.summary_plot(shap_values, x_train, plot_type="bar")
shap.force_plot(explainer.expected_value, shap_values[269,:], x_train.columns, matplotlib=True)

#%% Save your model
gbm.booster_.save_model(r'path_and_name_of_model_file')
