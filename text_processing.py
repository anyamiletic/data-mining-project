import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("processed_data.csv", encoding='utf-8')

df = df.sample(frac=0.1)

col = ['variety','description']
df = df[col]
df = df[pd.notnull(df['variety'])]
df = df[pd.notnull(df['description'])]

#keep only the top ten varieties
to_keep = ['Pinot Noir','Chardonnay','Cabernet Sauvignon','Red Blend','Bordeaux-style Red Blend','Sauvignon Blanc','Syrah','Riesling','Merlot','Zinfandel']
df = df[df['variety'].isin(to_keep)]

df.columns = ['Variety','Description']
df['category_id'] = df['Variety'].factorize()[0]
category_id_df = df[['Variety', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Variety']].values)
print(df.head())

fig = plt.figure(figsize=(8,6))
df.groupby('Variety').Description.count().plot.bar(ylim=0)
plt.show()


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.Description).toarray()
labels = df.category_id
print(features.shape)
# each of ~5000 descripitons is 
# represented by ~5000 features, representing the 
# tf-idf (Term Frequency, Inverse Document Frequency) 
# score for different unigrams and bigrams

N = 2
for Variety, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Variety))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

X_train, X_test, y_train, y_test = train_test_split(df['Description'], df['Variety'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

print(clf.predict(count_vect.transform(["Tannins and acidity"])))
#we get pinot noir as the prediction

print(clf.predict(count_vect.transform(["A rich blend of blackberry, strong flavors"])))
#we get red blend 

