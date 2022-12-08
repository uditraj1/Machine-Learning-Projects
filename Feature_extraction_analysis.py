from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document.',
         'This is the second document.',
         'Third document. Document number three',
         'Number four. To repeat, number four.']
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)
bag_of_words
print(bag_of_words)
vectorizer.vocabulary_.get('document')
vectorizer.vocabulary_
import pandas as pd
print(pd.__version__)
pd.DataFrame(bag_of_words.toarray(),columns=vectorizer.get_feature_names())
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)
print(bag_of_words)
pd.DataFrame(bag_of_words.toarray(),columns=vectorizer.get_feature_names())
vectorizer.vocabulary_
from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer(n_features=8)
bag_of_words = vectorizer.fit_transform(corpus)
print(bag_of_words)
