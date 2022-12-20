# %%
import importlib
from pathlib import Path
from math import sqrt

from fitter import Fitter, get_common_distributions
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression

from source.KNN import KNN, euclidean_distance
from source.PNN import PNN
from source.metrics import accuracy_score, plot_confusion_matrix, print_summary
from source.plot import plot_kde, plot_histrogram, plot_importance, plot_pairplot, plot_correlation, plot_pca
from source.fuzzy import trapezeL, trapezeR, triangle, tNorm, sNorm

pd.options.display.float_format = '{:.3f}'.format
pd.options.mode.chained_assignment = None

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# Data read

# %%
pure = pd.read_csv(Path('./source/data/water_potability.csv'))
pure.dropna(inplace=True)
pure.reset_index(inplace=True, drop=True)

x = pure.iloc[:, :-1]
y = pure.iloc[:, -1]
name = pure.iloc[:, 0].name

for col in x:
  if x[col].dtype == 'object':
    m = {old_v: new_v for new_v, old_v in enumerate(x[col].unique())}
    x[col] = x[col].map(m)

y = y.map({0: 'f', 1: 't'})

# %%
pure = pd.read_csv(Path('./source/data/bodyPerformance.csv'))
pure.dropna(inplace=True)
pure.reset_index(inplace=True, drop=True)

x = pure.loc[:, pure.columns != "gender"]
y = pure.loc[:, "gender"]
name = pure.iloc[:, 0].name

for col in x:
  if x[col].dtype == 'object':
    m = {old_v: new_v for new_v, old_v in enumerate(x[col].unique())}
    x[col] = x[col].map(m)

x.columns = [name.replace(' ', '_') for name in x.columns]
y.name = y.name.replace(' ', '_')

# %%
pure = pd.read_csv(Path('./source/data/exams.csv'))
pure.dropna(inplace=True)
pure.reset_index(inplace=True, drop=True)

x = pure.iloc[:, pure.columns != "gender"]
y = pure.loc[:, "gender"]
name = pure.iloc[:, 0].name

for col in x:
  if x[col].dtype == 'object':
    m = {old_v: new_v for new_v, old_v in enumerate(x[col].unique())}
    x[col] = x[col].map(m)

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# I Data visualization

# %%
plot_pairplot(x, y)

# %%
plot_correlation(x, y)

# %%
plot_pca(x)

# %%
plot_kde(x, y)

# %%
plot_histrogram(x, y)

# %%
df = pd.melt(pd.concat([x, y], axis=1), id_vars=[y.name])
ax = sns.boxplot(data=df, x='variable', y='value', hue=y.name)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

# %%
ax = sns.boxplot(data=x)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# Removing outliers

# %%
clf = IsolationForest(n_estimators=round(y.size/10))
pred = clf.fit_predict(x)

mask = [True if x == -1 else False for x in pred]
to_remove = x.loc[mask, name]

print(to_remove)

x.drop(to_remove.index, inplace=True)
y.drop(to_remove.index, inplace=True)
x.reset_index(inplace=True, drop=True)
y.reset_index(inplace=True, drop=True)

# %%
removed_elements = pd.Series(dtype='float')

for col in x:
  outliers = boxplot_stats(x.loc[:, col])[0]['fliers']
  mask = x.loc[:, col].isin(outliers)
  to_remove = x.loc[mask, :]
  
  removed_elements = pd.concat([to_remove, removed_elements], axis=0)

  x.drop(to_remove.index, inplace=True)
  y.drop(to_remove.index, inplace=True)
  x.reset_index(inplace=True, drop=True)
  y.reset_index(inplace=True, drop=True)

print(removed_elements)

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# Normalizers

# %%
normalizer = preprocessing.MinMaxScaler().fit(x)
x = pd.DataFrame(normalizer.transform(x), columns=x.columns)

# %%
normalizer = preprocessing.StandardScaler().fit(x)
x = pd.DataFrame(normalizer.transform(x), columns=x.columns)

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# PCA

# %%
explained_variance = 0.95

pca = PCA()
pca.fit(x)

total_sum = np.cumsum(pca.explained_variance_ratio_)
coponents = np.argmax(total_sum >= explained_variance) + 1

pca = PCA(n_components=coponents)
principalComponents = pca.fit_transform(x)

x = pd.DataFrame(data = principalComponents, columns=[f"PCA{i}" for i in range(coponents)])

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# Classification using KNN

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
y_pred = KNN(x_train, y_train, 3, x_test, euclidean_distance)
plot_confusion_matrix(y_test, y_pred)
print_summary(y_test, y_pred)

# %%
data = pd.DataFrame(columns=["train_acc", "test_acc"])

for i in range(2,20):
  KNN_keras = KNeighborsClassifier(n_neighbors=i)
  KNN_keras.fit(x_train, y_train)

  y_pred_train = KNN_keras.predict(x_train)
  y_pred_test = KNN_keras.predict(x_test)

  data.loc[i] = [
    accuracy_score(y_train, y_pred_train, True),
    accuracy_score(y_test, y_pred_test, True)
  ]

sns.lineplot(data=data)

# %%
KNN_keras = KNeighborsClassifier(n_neighbors=5)
KNN_keras.fit(x_train, y_train)

y_pred_train = KNN_keras.predict(x_train)
y_pred_test = KNN_keras.predict(x_test)

plot_confusion_matrix(y_train, y_pred_train)
print_summary(y_train, y_pred_train)

plot_confusion_matrix(y_test, y_pred_test)
print_summary(y_test, y_pred_test)

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# Classification using PNN

# %%
pnn = PNN()
pnn.fit(x_train, y_train)
y_pred = pnn.predict(x_test)

plot_confusion_matrix(y_test, y_pred)
print_summary(y_test, y_pred)

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# Classification decision tree

# %%
data = pd.DataFrame(columns=["train_acc", "test_acc"])

for i in range(2,20):
  tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=i)
  tree.fit(x_train, y_train)

  y_pred_train = tree.predict(x_train)
  y_pred_test = tree.predict(x_test)

  data.loc[i] = [
    accuracy_score(y_train, y_pred_train, True),
    accuracy_score(y_test, y_pred_test, True)
  ]

sns.lineplot(data=data)

# %%
tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=2)
tree.fit(x_train, y_train)

y_pred_train = tree.predict(x_train)
y_pred_test = tree.predict(x_test)

plot_confusion_matrix(y_train, y_pred_train)
print_summary(y_train, y_pred_train)

plot_confusion_matrix(y_test, y_pred_test)
print_summary(y_test, y_pred_test)

# plot_tree(tree)

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# Feature Importance

# %%
tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=2)
tree.fit(x_train, y_train)

plot_importance(x.columns, tree.feature_importances_)

# %%
results = permutation_importance(tree, x_train, y_train, scoring='accuracy')
importance = results.importances_mean

plot_importance(x.columns, importance, results.importances_std)

# %%
KNN_keras = KNeighborsClassifier(n_neighbors=5)
KNN_keras.fit(x_train, y_train)
results = permutation_importance(KNN_keras, x_train, y_train, scoring='accuracy')
importance = results.importances_mean

plot_importance(x.columns, importance, results.importances_std)

# %%
model = LogisticRegression()
model.fit(x_train, y_train)

plot_importance(x.columns, model.coef_[0])

# %%
model = LinearRegression()
model.fit(x_train, y_train.map({'f': 0, 't': 1}))

plot_importance(x.columns, model.coef_[0])

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# Fuzzy logic

# %%
class_name = "F"
feature = "height_cm"

filter = (y_train == class_name)
data = x_train.loc[filter, feature]
f = Fitter(data, distributions=get_common_distributions(), bins=round(sqrt(y_train.size)))
f.fit()

# %%
f.df_errors

# %%
f.fitted_param

# %%
short = trapezeR(0.3, 0.6)
tall = trapezeL(0.3, .6)

light = trapezeR(0.3, 0.6)
heavy = trapezeL(0.3, .6)

weak = trapezeR(0.3, 0.6)
strong = trapezeL(0.3, .6)

jump_low = trapezeR(0.3, 0.6)
jump_high = trapezeL(0.3, .6)

# %%
y_pred_train = []

for i, sample in x_train.iterrows():
  # m = sNorm(tall(sample.height_cm), sNorm(heavy(sample.weight_kg), sNorm(strong(sample.gripForce), jump_low(sample.broad_jump_cm))))
  m = tall(sample.height_cm) + heavy(sample.weight_kg) + strong(sample.gripForce) + jump_low(sample.broad_jump_cm)
  f = short(sample.height_cm) + light(sample.weight_kg) + weak(sample.gripForce) + jump_high(sample.broad_jump_cm)

  if m > f:
    y_pred_train.append("M")
  else:
    y_pred_train.append("F")

plot_confusion_matrix(y_train, y_pred_train)
print_summary(y_train, y_pred_train)

# %%
