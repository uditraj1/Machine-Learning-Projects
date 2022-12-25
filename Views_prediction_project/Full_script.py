#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#connecting to the data
from google.colab import drive 
drive.mount('/content/drive')

#importing log data file
log_data = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/ENPM808O/log.pickle')

#converting it to a data frame
df_log_data = pd.DataFrame(log_data)

#displaying initial values
print(df_log_data)

#importing test movies data
test_movies = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/ENPM808O/test_movi es.pickle')

df_test_movies = pd.DataFrame(test_movies) 
print(df_test_movies)

df_train_movies = pd.DataFrame(train_movies) 
print(df_train_movies)

#importing users data
users = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/ENPM808O/users.pickle')

df_users = pd.DataFrame(users)
print(df_users)

#checking for numeric columns
df_numeric_train_movies = df_train_movies.select_dtypes(include=[np.number]) 
numeric_cols = df_numeric_train_movies.columns.values

#checking for non-numeric columns
df_numeric_train_movies = df_train_movies.select_dtypes(include=[np.number]) 
numeric_cols = df_numeric_train_movies.columns.values
df_non_numeric_train_movies = df_train_movies.select_dtypes(exclude=[np.number]) 
non_numeric_cols = df_non_numeric_train_movies.columns.values

print(numeric_cols)

print(non_numeric_cols)

#checking for missing values in the columns
values_list = list() 
cols_list = list()
for col in df_train_movies.columns:
pct_missing = np.mean(df_train_movies[col].isnull())*100 
cols_list.append(col)
values_list.append(pct_missing) 
pct_missing_df = pd.DataFrame()
pct_missing_df['col'] = cols_list
pct_missing_df['pct_missing'] = values_list

print(pct_missing_df)
df_users.head()

print(df_users.T)

#transposing the data frame for the correct format
df_users = df_users.T
df_users

df_numeric_users = df_users.select_dtypes(include=[np.number]) 
numeric_cols_users = df_numeric_users.columns.values
df_non_numeric_users = df_users.select_dtypes(exclude=[np.number]) 
non_numeric_cols_users = df_non_numeric_users.columns.values

numeric_cols_users

print(numeric_cols_users)

print(non_numeric_cols_users)

values_list_users = list() 
cols_list_users = list()
for col in df_users.columns:
pct_missing_users = np.mean(df_users[col].isnull())*100 
cols_list_users.append(col)
values_list_users.append(pct_missing_users) 
pct_missing_df_users = pd.DataFrame()
pct_missing_df_users['col'] = cols_list_users
pct_missing_df_users['pct_missing_users'] = values_list_users

print(pct_missing_df_users)

df_numeric_test_movies = df_test_movies.select_dtypes(include=[np.number]) 
numeric_cols_test_movies = df_numeric_test_movies.columns.values
df_non_numeric_test_movies = df_test_movies.select_dtypes(exclude=[np.number]) 
non_numeric_cols_test_movies = df_non_numeric_test_movies.columns.values

print(numeric_cols_test_movies)

print(non_numeric_cols_test_movies)

values_list_test_movies = list() 
cols_list_test_movies = list()
for col in df_test_movies.columns:
pct_missing_test_movies = np.mean(df_test_movies[col].isnull())*100 
cols_list_test_movies.append(col)
values_list_test_movies.append(pct_missing_test_movies) 
pct_missing_df_test_movies = pd.DataFrame()
pct_missing_df_test_movies['col'] = cols_list_test_movies
pct_missing_df_test_movies['pct_missing_test_movies'] = values_list_test_movies

print(pct_missing_df_test_movies)

#importing the external data source
rating_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ENPM808O/title.rating s.tsv',sep='\t')

df_rating_data = pd.DataFrame(rating_data)

df_rating_data.head()

df_rating_data

df_numeric_rating_data = df_rating_data.select_dtypes(include=[np.number]) 
numeric_cols_rating_data = df_numeric_rating_data.columns.values
df_non_numeric_rating_data = df_rating_data.select_dtypes(exclude=[np.number]) 
non_numeric_cols_rating_data = df_non_numeric_rating_data.columns.values

print(numeric_cols_rating_data)

print(non_numeric_cols_rating_data)

values_list_rating_data = list() 
cols_list_rating_data = list()
for col in df_rating_data.columns:
pct_missing_rating_data = np.mean(df_rating_data[col].isnull())*100 
cols_list_rating_data.append(col)
values_list_rating_data.append(pct_missing_rating_data) 
pct_missing_df_rating_data = pd.DataFrame()
pct_missing_df_rating_data['col'] = cols_list_rating_data
pct_missing_df_rating_data['pct_missing_rating_data'] = values_list_rating_data

print(pct_missing_df_rating_data)

df_train_movies.head()

#checking for differences in values
df_rating_data.averageRating.describe()

#displaying box plots to find outliers
df_rating_data.averageRating.plot(kind='box', figsize=(12, 8)) 
plt.show()

df_rating_data.numVotes.describe()

df_rating_data["numVotes"].describe().apply(lambda x: format(x, 'f'))

df_rating_data.numVotes.plot(kind='box', figsize=(12, 8)) 
plt.show()

df_rating_data["averageRating"].describe().apply(lambda x: format(x, 'f'))

df_rating_data.averageRating.plot(kind='box', figsize=(12, 8)) 
plt.show()

df_train_movies['budget'] = pd.to_numeric(df_train_movies['budget'],errors='coerce') 
df_train_movies.dtypes

df_train_movies["budget"].describe().apply(lambda x: format(x, 'f'))

df_train_movies.dtypes

#converting object to numeric type
df_train_movies['budget'] = pd.to_numeric(df_train_movies['budget'],errors='coerce') 
df_train_movies.dtypes

df_train_movies["budget"].describe().apply(lambda x: format(x, 'f'))

df_train_movies.budget.plot(kind='box', figsize=(12, 8)) 
plt.show()

df_train_movies["runtime"].describe().apply(lambda x: format(x, 'f'))

df_train_movies.runtime.plot(kind='box', figsize=(12, 8))
plt.show()

df_test_movies['budget'] = pd.to_numeric(df_test_movies['budget'],errors='coerce')
df_test_movies.dtypes

df_test_movies["budget"].describe().apply(lambda x: format(x, 'f'))

df_train_movies.budget.plot(kind='box', figsize=(12, 8))
plt.show()

df_test_movies["runtime"].describe().apply(lambda x: format(x, 'f'))

df_train_movies.runtime.plot(kind='box', figsize=(12, 8)) 
plt.show()

df_users["age"].describe().apply(lambda x: format(x, 'f'))

df_users.age.plot(kind='box', figsize=(12, 8))
plt.show()

#removing the missing values from the columns
less_missing_values_cols_list_test = list(pct_missing_df_test_movies.loc[(pct_missing_d f_test_movies.pct_missing_test_movies < 0.5) & (pct_missing_df_test_movies.pct_missing_ test_movies > 0), 'col'].values)
df_test_movies.dropna(subset=less_missing_values_cols_list_test, inplace=True)

values_list_test_movies = list() 
cols_list_test_movies = list()
for col in df_test_movies.columns:
pct_missing_test_movies = np.mean(df_test_movies[col].isnull())*100 
cols_list_test_movies.append(col)
values_list_test_movies.append(pct_missing_test_movies) 
pct_missing_df_test_movies = pd.DataFrame()
pct_missing_df_test_movies['col'] = cols_list_test_movies
pct_missing_df_test_movies['pct_missing_test_movies'] = values_list_test_movies

#all missing values removed
print(pct_missing_df_test_movies)

less_missing_values_cols_list_train = list(pct_missing_df.loc[(pct_missing_df.pct_missi ng < 0.5) & (pct_missing_df.pct_missing > 0), 'col'].values)
df_train_movies.dropna(subset=less_missing_values_cols_list_train, inplace=True)

values_list_train_movies = list() 
cols_list_train_movies = list()
for col in df_train_movies.columns:
pct_missing_train_movies = np.mean(df_train_movies[col].isnull())*100 
cols_list_train_movies.append(col)
values_list_train_movies.append(pct_missing_train_movies)
pct_missing_df_train_movies = pd.DataFrame()
pct_missing_df_train_movies['col'] = cols_list_train_movies
pct_missing_df_train_movies['pct_missing_train_movies'] = values_list_train_movies

print(pct_missing_df_train_movies)

#checking for duplicate rows
df_users.duplicated()

df_rating_data.duplicated()

df_rating_data = df_rating_data.loc[df_rating_data.numVotes < 2000000]

df_rating_data["numVotes"].describe().apply(lambda x: format(x, 'f'))

df_rating_data.numVotes.plot(kind='box', figsize=(12, 8))
plt.show()

df_rating_data = df_rating_data.loc[df_rating_data.numVotes < 1500000]

df_rating_data.numVotes.plot(kind='box', figsize=(12, 8))
plt.show()

#removing outliers
df_train_movies = df_train_movies.loc[df_train_movies.budget < 250000000]

df_train_movies.budget.plot(kind='box', figsize=(12, 8)) 
plt.show()

df_train_movies = df_train_movies.loc[df_train_movies.runtime < 800] 
df_train_movies.runtime.plot(kind='box', figsize=(12, 8))
plt.show()

df_test_movies = df_test_movies.loc[df_test_movies.runtime < 300] 
df_test_movies.runtime.plot(kind='box', figsize=(12, 8))
plt.show()

df_test_movies = df_test_movies.loc[df_test_movies.budget < 180000000] 
df_test_movies.budget.plot(kind='box', figsize=(12, 8))
plt.show()

#removed the rows having no vote counts
df_train_movies = df_train_movies[df_train_movies.vote_count != '0'] 
df_train_movies

#keeping only those rows for which movie has been released
df_train_movies = df_train_movies[df_train_movies.status == 'Released'] 
df_train_movies

df_numeric_log_data = df_log_data.select_dtypes(include=[np.number]) 
numeric_cols_log_data = df_numeric_log_data.columns.values
df_non_numeric_log_data = df_log_data.select_dtypes(exclude=[np.number]) 
non_numeric_cols_log_data = df_non_numeric_log_data.columns.values

print(numeric_cols_log_data)

print(non_numeric_cols_log_data)

values_list_log_data = list() 
cols_list_log_data = list()
for col in df_log_data.columns:
pct_missing_log_data = np.mean(df_log_data[col].isnull())*100 
cols_list_log_data.append(col)
values_list_log_data.append(pct_missing_log_data) 
pct_missing_df_log_data = pd.DataFrame()
pct_missing_df_log_data['col'] = cols_list_log_data
pct_missing_df_log_data['pct_missing_log_data'] = values_list_log_data

print(pct_missing_df_log_data)

df_log_data.head()

df_log_data.duplicated()

df_log_data = df_log_data.drop_duplicates(subset=['userId']) 
df_log_data

#merging data frames together
merged_table = pd.merge(df_log_data,df_users,left_on='userId',right_on='user_id',how='l eft')

merged_table

#dropping irrelevant columns
merged_table = merged_table.drop(['time','occupation'],axis=1) 
merged_table

df_users = df_users.T

df_users.head()

merged_table.head()

merged_table2 = pd.merge(df_log_data,df_train_movies,left_on='title',right_on='id')

merged_table2

df_log_data.userId.duplicated()

#checking for column types
df_train_movies.dtypes

df_rating_data.head()

#merging other data frames together
merged_table2 = pd.merge(merged_table2,df_rating_data,left_on='imdb_id',right_on='tcons t')
merged_table2

merged_table2 = merged_table2.drop(['time','title_x','original_title','spoken_language s'],axis=1)
merged_table2

merged_table2.head()

merged_table2 = merged_table2.drop(['id','belongs_to_collection','genres','homepage'],a xis=1)
merged_table2

merged_table2 = merged_table2.drop(['poster_path'],axis=1) 
merged_table2

merged_table2 = merged_table2.drop(['production_companies'],axis=1) 
merged_table2

#removing tt from imdb_id to convert it to numeric form later
merged_table2["imdb_id"] = merged_table2["imdb_id"].str.replace("tt","")
merged_table2.head()

#encoding adult column to a usable model format
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
merged_table2['adult'] = le.fit_transform(merged_table2['adult'].astype(str))

#checking for unique values in a column
n = len(pd.unique(merged_table2['original_language'])) 
print("No.of.unique values :", n)
m = len(pd.unique(merged_table2['overview'])) 
print("No.of.unique values :", m)

merged_table2['userId'] = pd.to_numeric(merged_table2['userId'],errors='coerce') 
merged_table2['imdb_id'] = merged_table2['imdb_id'].astype("string")
merged_table2 = merged_table2.drop(['title_y'],axis=1) 
merged_table2.dtypes

merged_table2['imdb_id'].head()

merged_table2['imdb_id'] = pd.to_numeric(merged_table2['imdb_id'],errors='coerce') 
merged_table2 = merged_table2.drop(['original_language'],axis=1)
merged_table2 = merged_table2.drop(['overview'],axis=1)
merged_table2['popularity'].head()

#converting columns to correct format
merged_table2['popularity'] = pd.to_numeric(merged_table2['popularity'],errors='coerce')
merged_table2['revenue'] = pd.to_numeric(merged_table2['revenue'],errors='coerce') 
merged_table2['status'] = le.fit_transform(merged_table2['status'].astype(str))
merged_table2['vote_average'] = pd.to_numeric(merged_table2['vote_average'],errors='coe rce')
merged_table2['vote_count'] = pd.to_numeric(merged_table2['vote_count'],errors='coerce')
merged_table2 = merged_table2.drop(['tconst'],axis=1) 
merged_table2.dtypes

merged_table2.head()

merged_table.dtypes

merged_table.head()

merged_table2.dtypes

merged_table2 = merged_table2.drop(['date'],axis=1)
merged_table2 = merged_table2.drop(['release_date'],axis=1) 
merged_table2.head()

merged_table = merged_table.drop(['date'],axis=1) 
merged_table.head()

#making views_per_day as the dependent variable to be predicted along with data splitti ng
from sklearn.model_selection import train_test_split
X = merged_table2.drop('views_per_day',axis=1)
Y = merged_table2['views_per_day']
X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

merged_table2

#dropping any rows with NaN values
merged_table2 = merged_table2.dropna()

from sklearn.model_selection import train_test_split
X = merged_table2.drop('views_per_day',axis=1)
Y = merged_table2['views_per_day']
X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#training the ridge regression model from sklearn.linear_model import Ridge
regression_model = Ridge(alpha=0.1)
regression_model.fit(X_train,Y_train)

#getting the coefficients of the columns
regression_model.coef_

#checking for accuracy on training data
regression_model.score(X_train,Y_train)

y_predict = regression_model.predict(x_test)

#checking for accuracy in test data
regression_model.score(x_test,y_test)

#checking for error rate
from sklearn.metrics import mean_squared_error
regression_model_mse = mean_squared_error(y_predict,y_test)
regression_model_mse

import math
math.sqrt(regression_model_mse)

#training different models
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

lr.coef_

lr.score(X_train,Y_train)

y_predict = lr.predict(x_test)

lr.score(x_test,y_test)

from sklearn.metrics import mean_squared_error
regression_model_mse = mean_squared_error(y_predict,y_test) 
regression_model_mse

import math
math.sqrt(regression_model_mse)

from sklearn.linear_model import LinearRegression 
lr = LinearRegression()
lr.fit(X_train,Y_train)

y_predict = lr.predict(x_test)

lr.score(x_test,y_test)

from sklearn.linear_model import Lasso 
lasso = Lasso(alpha=0.1)
lasso.fit(X_train,Y_train)

lasso.coef_

lasso.score(X_train,Y_train)

y_predict = lasso.predict(x_test)

lasso.score(x_test,y_test)

regression_model = Ridge(alpha=0.5)
regression_model.fit(X_train,Y_train)

regression_model.score(X_train,Y_train)

y_predict = regression_model.predict(x_test)
regression_model.score(x_test,y_test)

lasso = Lasso(alpha=0.5)
lasso.fit(X_train,Y_train)
lasso.score(X_train,Y_train)
y_predict = lasso.predict(x_test) 
lasso.score(x_test,y_test)

lasso.score(X_train,Y_train)

X.head()

#comparing the model predictions visually
features = X.columns[0:14]
plt.figure(figsize = (10, 10))
plt.plot(features,regression_model.coef_,alpha=0.7,linestyle='none',marker='*',markersi ze=5,color='red',label=r'Ridge; $\alpha = 10$',zorder=7)
#plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',l abel=r'Ridge; $\alpha = 100$')
plt.plot(features,lasso.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color= 'green',label='Lasso Regression')
plt.xticks(rotation = 90) 
plt.legend()
plt.show()

df_test_movies = df_test_movies[df_test_movies.status == 'Released'] 
df_train_movies

df_test_movies = df_test_movies.drop(['id'],axis=1)
df_test_movies = df_test_movies.drop(['title'],axis=1)
df_test_movies = df_test_movies.drop(['original_title'],axis=1)
df_test_movies = df_test_movies.drop(['belongs_to_collection'],axis=1) 
df_test_movies = df_test_movies.drop(['genres'],axis=1)
df_test_movies = df_test_movies.drop(['homepage'],axis=1)
df_test_movies = df_test_movies.drop(['production_companies'],axis=1)
df_test_movies = df_test_movies.drop(['production_countries'],axis=1)
df_test_movies = df_test_movies.drop(['release_date'],axis=1)
df_test_movies

df_test_movies = df_test_movies.drop(['spoken_languages'],axis=1) 
df_test_movies

df_test_movies = df_test_movies.drop(['original_language'],axis=1) 
df_test_movies = df_test_movies.drop(['overview'],axis=1)
df_test_movies = df_test_movies.drop(['poster_path'],axis=1) 
df_test_movies

df_test_movies.dtypes

df_test_movies['adult'] = le.fit_transform(df_test_movies['adult'].astype(str))
df_test_movies['status'] = le.fit_transform(df_test_movies['status'].astype(str)) 
df_test_movies.head()

df_test_movies.dtypes

df_test_movies["imdb_id"] = df_test_movies["imdb_id"].str.replace("tt","")
df_test_movies.head()

df_test_movies['imdb_id'] = pd.to_numeric(df_test_movies['imdb_id'],errors='coerce') 
df_test_movies['revenue'] = pd.to_numeric(df_test_movies['revenue'],errors='coerce')
df_test_movies.dtypes

df_test_movies.head()

check_nan = df_test_movies['tmdb_id'].isnull().values.any()
print(check_nan)

check_nan = df_test_movies['imdb_id'].isnull().values.any() 
print(check_nan)

check_nan = df_test_movies['adult'].isnull().values.any() 
print(check_nan)

check_nan = df_test_movies['budget'].isnull().values.any() 
print(check_nan)

check_nan = df_test_movies['revenue'].isnull().values.any() 
print(check_nan)

check_nan = df_test_movies['runtime'].isnull().values.any() 
print(check_nan)

check_nan = df_test_movies['status'].isnull().values.any() 
print(check_nan)

#checking for prediction with test dataset
X = df_test_movies.drop('views_per_day',axis=1) 
Y = df_test_movies['views_per_day']
X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

lasso_new = Lasso(alpha=0.4)
lasso_new.fit(X_train,Y_train)

lasso_new.score(X_train,Y_train)

lasso_new = Lasso(alpha=0.1)
lasso_new.fit(X_train,Y_train)

lasso_new.score(X_train,Y_train)

lasso_new.score(x_test,y_test)

#changing hyperparameter values to check for accuracy changes
ridge_new = Ridge(alpha=0.4)
ridge_new.fit(X_train,Y_train)

ridge_new.score(X_train,Y_train)

ridge_new.score(x_test,y_test)

#splitting data into train,validate and test 
X = merged_table2.drop('views_per_day',axis=1)
Y = merged_table2['views_per_day']
x, x_test, y, y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8)
x_train, x_cv, y_train, y_cv = train_test_split(x,y,test_size = 0.25,train_size =0.75)

linear_regression = LinearRegression() 
linear_regression.fit(x_train,y_train) 
ridge_regressor = Ridge(alpha=0.2)
ridge_regressor.fit(x_train,y_train)

linear_regression.score(x_train,y_train)

ridge_regressor.score(x_train,y_train)

ridge_regressor = Ridge(alpha=0.8)
ridge_regressor.fit(x_train,y_train)

ridge_regressor.score(x_train,y_train)

linear_regression.score(x_cv,y_cv)

ridge_regressor.score(x_cv,y_cv)

linear_regression.score(x_test,y_test)

ridge_regressor.score(x_test,y_test)

features = X.columns[0:14]
plt.figure(figsize = (10, 10))
plt.plot(features,linear_regression.coef_,alpha=0.7,linestyle='none',marker='*',markers ize=5,color='red',label=r'Linear Regression; $\alpha = 10$',zorder=7)
#plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',l abel=r'Ridge; $\alpha = 100$')
plt.plot(features,ridge_regressor.coef_,alpha=0.4,linestyle='none',marker='o',markersiz e=7,color='green',label='Ridge Regression')
plt.xticks(rotation = 90) 
plt.legend()
plt.show()

!pip install lime

#importing lime for explainabiltiy
import lime
import lime.lime_tabular

#dividing the data
X = merged_table2[['entry','userId','tmdb_id','imdb_id','adult','budget','popularity', 'revenue','runtime','status','vote_average','vote_count','averageRating','numVotes']]
y = merged_table2['views_per_day']

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#training three models and checking for explainability
from sklearn.ensemble import RandomForestRegressor 
lasso_explain = Lasso(alpha=0.1)
lasso_explain.fit(X_train,y_train) 
ridge_explain = Ridge(alpha=0.1)
ridge_explain.fit(X_train,y_train)
rf_model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
rf_model.fit(X_train, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_trai n.columns.values.tolist(),class_names=['views_per_day'], verbose=True, mode='regression')

j = 5
exp = explainer.explain_instance(X_test.values[j], ridge_regressor.predict, num_feature s=6)

#for ridge
exp.show_in_notebook(show_table=True)

j = 5
exp = explainer.explain_instance(X_test.values[j], rf_model.predict, num_features=6)

#for random forest
exp.show_in_notebook(show_table=True)

j = 5
exp = explainer.explain_instance(X_test.values[j], lasso_explain.predict, num_features= 6)

#for lasso
exp.show_in_notebook(show_table=True)

j = 5
exp = explainer.explain_instance(X_test.values[j], ridge_explain.predict, num_features= 6)

#for ridge
exp.show_in_notebook(show_table=True)





