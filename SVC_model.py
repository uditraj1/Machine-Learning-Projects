from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',shuffle=True)
twenty_train.keys()
print(twenty_train.data[0])

twenty_train.target_names

twenty_train.data[0]

twenty_train.target

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

print(X_train_counts[0])
  
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

print(X_train_tfidf[0])
  
from sklearn.svm import LinearSVC
clf_svc = LinearSVC(penalty="l1",dual=False,tol=1e-3)
clf_svc.fit(X_train_tfidf,twenty_train.target)

from sklearn.pipeline import Pipeline
clf_svc_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC(penalty="l1",dual=False,tol=0.001))
])
clf_svc_pipeline.fit(twenty_train.data,twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test',shuffle=True)
predicted = clf_svc_pipeline.predict(twenty_test.data)
from sklearn.metrics import accuracy_score
acc_svm = accuracy_score(twenty_test.target,predicted)
acc_svm
