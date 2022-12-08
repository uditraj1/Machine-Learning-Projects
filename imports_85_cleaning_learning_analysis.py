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
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train,Y_train)
linear_model.score(X_train,Y_train)
linear_model.coef_
predictors = X_train.columns
coef = pd.Series(linear_model.coef_,predictors).sort_values()
print(coef)
y_predict = linear_model.predict(x_test)
%pylab inline
pylab.rcParams['figure.figsize'] = (15,6)
plt.plot(y_predict,label='Predicted')
plt.plot(y_test.values,label='Actual')
plt.ylabel('Price')
plt.legend()
plt.show()
r_square = linear_model.score(x_test,y_test)
r_square
from sklearn.metrics import mean_squared_error
linear_model_mse = mean_squared_error(y_predict,y_test)
linear_model_mse
import math
math.sqrt(linear_model_mse)
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=5,normalize=True)
lasso_model.fit(X_train,Y_train)
lasso_model.score(X_train,Y_train)
coef = pd.Series(lasso_model.coef_,predictors).sort_values()
print(coef)
y_predict = lasso_model.predict(x_test)
pylab.rcParams['figure.figsize']=(15,6)
plt.plot(y_predict,label='Predicted')
plt.plot(y_test.values,label='Actual')
plt.ylabel('Price')
plt.legend()
plt.show()
r_square = lasso_model.score(x_test,y_test)
r_square
lasso_model_mse = mean_squared_error(y_predict,y_test)
math.sqrt(lasso_model_mse)
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=0.5,normalize=True)
ridge_model.fit(X_train,Y_train)
ridge_model.score(X_train,Y_train)
coef = pd.Series(ridge_model.coef_,predictors).sort_values()
print(coef)
y_predict = ridge_model.predict(x_test)
pylab.rcParams['figure.figsize']=(15,6)
plt.plot(y_predict,label='Predicted')
plt.plot(y_test.values,label='Actual')
plt.ylabel('Price')
plt.legend()
plt.show()
r_square = ridge_model.score(x_test,y_test)
r_square
ridge_model_mse = mean_squared_error(y_predict,y_test)
math.sqrt(ridge_model_mse)
