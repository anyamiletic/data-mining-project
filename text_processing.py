#thanks to Susan Li at towardsdatascience.com

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
import seaborn as sns

from sklearn.metrics import confusion_matrix

from IPython.display import display

from sklearn import metrics

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

X_train, X_test, y_train, y_test = train_test_split(df['Variety'], df['Description'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

print(clf.predict(count_vect.transform(["Red Blend"])))
# we get : A 60-40 blend of Cabernet Sauvignon and Merlot, Patrimonio opens with bright fruit 
# tones of cherry and blueberry with background notes of spice and leather. 
# Overall, it's easy and compact in style.

print(clf.predict(count_vect.transform(["Pinot Noir"])))
# A barrel selection from estate-grown grapes, this Pinot 
# succeeds in rich red raspberry and an offering of strawberry-rhubarb pie, 
# the texture velvety smooth. The oak builds through the finish, providing ample body and substance, 
# yet the wine stays light and lilting on the palate, inviting one in for more



# models = [
#     RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#     LinearSVC(),
#     MultinomialNB(),
#     LogisticRegression(random_state=0),
# ]
# CV = 5
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# for model in models:
#   model_name = model.__class__.__name__
#   accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
#   for fold_idx, accuracy in enumerate(accuracies):
#     entries.append((model_name, fold_idx, accuracy))
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])



# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()
# cv_df.groupby('model_name').accuracy.mean()

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Variety.values, yticklabels=category_id_df.Variety.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show() #prikazuje matricu konfuzije

# shows those misplaced descriptions

# for predicted in category_id_df.category_id:
#   for actual in category_id_df.category_id:
#     if predicted != actual and conf_mat[actual, predicted] >= 10:
#       print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
#       display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Variety', 'Description']])
#       print('')

print(metrics.classification_report(y_test, y_pred, target_names=df['Variety'].unique()))

