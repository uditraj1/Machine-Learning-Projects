import pandas as pd
print(pd.__version__)

auto_data = pd.read_csv('C:/machine learning/imports-85.data',sep=r'\s*,\s*',engine='python')
auto_data

import numpy as np
auto_data = auto_data.replace('?',np.nan)
auto_data

auto_data.describe()

auto_data.describe(include='all')

auto_data['gas'].describe()

auto_data['gas'] = pd.to_numeric(auto_data['gas'],errors='coerce')
auto_data['gas'].describe()

auto_data = auto_data.drop('130',axis=1)
auto_data.head()

auto_data.describe()

auto_data['27'].describe()

two_dict = {'two':2,
         'four':4}
auto_data['two'].replace(two_dict,inplace=True)
auto_data.head()

auto_data = pd.get_dummies(auto_data,columns=['alfa-romero','gas','std','convertible','rwd','front','four','mpfi'])
auto_data.head()

auto_data = pd.get_dummies(auto_data,columns=['dohc'])
auto_data.head()

auto_data = auto_data.dropna()
auto_data

auto_data[auto_data.isnull().any(axis=1)]

from sklearn.model_selection import train_test_split
auto_data['13495'].describe()

auto_data['13495'] = pd.to_numeric(auto_data['13495'],errors='coerce')
auto_data.head()

X = auto_data.drop('13495',axis=1)
Y = auto_data['13495']
X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators':500,'max_depth':6,'min_samples_split':2,'learning_rate':0.01,'loss':'ls'}
gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train,Y_train)

gbr_model.score(X_train,Y_train)

y_predict = gbr_model.predict(x_test)
%pylab inline
pylab.rcParams['figure.figsize']=(15,6)
plt.plot(y_predict,label='Predicted')
plt.plot(y_test.values,label='Actual')
plt.ylabel('MPM')
plt.legend()
plt.show()

gbr_model.score(x_test,y_test)

from sklearn.metrics import mean_squared_error
gbr_model_mse = mean_squared_error(y_predict,y_test)
gbr_model_mse

import math 
math.sqrt(gbr_model_mse)

from sklearn.model_selection import GridSearchCV
learn_rate = [0.01,0.02,0.05,0.1]
max_depths = [4,6,8]
num_estimators = [100,200,500]
param_grid = {'n_estimators':num_estimators,
             'max_depth':max_depths,
             'learning_rate':learn_rate}
grid_search = GridSearchCV(GradientBoostingRegressor(min_samples_split=2,loss='ls'),
                           param_grid,cv=3,return_train_score=True)
grid_search.fit(X_train,Y_train)
grid_search.best_params_

grid_search.cv_results_

for i in range(36):
    print('Parameters: ',grid_search.cv_results_['params'][i])
    print('Mean Test Score: ',grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ',grid_search.cv_results_['rank_test_score'][i])
    print()

params = {'n_estimators':200,'max_depth':4,'min_samples_split':2,'learning_rate':0.1,'loss':'ls'}
gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train,Y_train)

y_predict = gbr_model.predict(x_test)
pylab.rcParams['figure.figsize']=(15,6)
plt.plot(y_predict,label='Predicted')
plt.plot(y_test,label='Actual')
plt.ylabel('MPM')
plt.legend()
plt.show()

gbr_model.score(x_test,y_test)

gbr_model_mse = mean_squared_error(y_predict,y_test)
math.sqrt(gbr_model_mse)
