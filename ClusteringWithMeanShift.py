import pandas as pd
titanic_data = pd.read_csv('C:/machine learning/train1.csv')
titanic_data.head()

titanic_data.drop(['PassengerId','Name','Ticket','Cabin'],'columns',inplace=True)
titanic_data.head()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'].astype(str))
titanic_data.head()

titanic_data = pd.get_dummies(titanic_data,columns=['Embarked'])
titanic_data.head()

titanic_data[titanic_data.isnull().any(axis=1)]

titanic_data = titanic_data.dropna()
titanic_data

from sklearn.cluster import MeanShift
analyzer = MeanShift(bandwidth=30)
analyzer.fit(titanic_data)

from sklearn.cluster import estimate_bandwidth
estimate_bandwidth(titanic_data)

import numpy as np
np.unique(analyzer.labels_)

import numpy as np
titanic_data['cluster_group'] = np.nan
data_length = len(titanic_data)
for i in range(data_length):
    titanic_data.iloc[i,titanic_data.columns.get_loc('cluster_group')] = analyzer.labels_[i]
titanic_data.head()

titanic_data.describe()

titanic_cluster_data = titanic_data.groupby(['cluster_group']).mean()
titanic_cluster_data

titanic_cluster_data['Counts'] = pd.Series(titanic_data.groupby(['cluster_group']).size())
titanic_cluster_data

titanic_data[titanic_data['cluster_group']==1].describe()
