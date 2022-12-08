import pandas as pd
print(pd.__version__)
auto_data = pd.read_csv('C:/machine learning/auto-mpg.data',delim_whitespace=True,header=None,
                       names=['mpg',
                             'cylinders',
                             'displacement',
                             'horsepower',
                             'weight',
                             'acceleration',
                             'model',
                             'origin',
                             'car_name'])
auto_data.head()
len(auto_data['car_name'].unique())
len(auto_data)
len(auto_data['car_name'])
auto_data = auto_data.drop('car_name',axis=1)
auto_data.head()
auto_data['origin'] = auto_data['origin'].replace({1:'america',2:'europe',3:'asia'})
auto_data.head()
auto_data = pd.get_dummies(auto_data,columns=['origin'])
auto_data.head()
auto_data
import numpy as np
auto_data = auto_data.replace('?',np.nan)
auto_data = auto_data.dropna()
auto_data
from sklearn.model_selection import train_test_split
X = auto_data.drop('mpg',axis=1)
Y = auto_data['mpg']
X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.svm import SVR
regression_model = SVR(kernel='linear',C=0.5)
regression_model.fit(X_train,Y_train)
regression_model.coef_
regression_model.score(X_train,Y_train)
from pandas import Series
import matplotlib.pyplot as plt
%matplotlib inline
predictors = X_train.columns
coef = Series(regression_model.coef_[0],predictors).sort_values()
coef.plot(kind='bar',title='Model Coefficients')
y_predict = regression_model.predict(x_test)
%pylab inline
pylab.rcParams['figure.figsize']=(15,6)
plt.plot(y_predict,label='Predicted')
plt.plot(y_test.values,label='Actual')
plt.ylabel('MPG')
plt.legend()
plt.show()
regression_model.score(x_test,y_test)
from sklearn.metrics import mean_squared_error
regression_model_mse = mean_squared_error(y_predict,y_test)
regression_model_mse
import math
math.sqrt(regression_model_mse)
