# %%
import importlib
from pathlib import Path

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

from source.KNN import KNN, euclidean_distance
from source.PNN import PNN
from source.metrics import plot_confusion_matrix, print_summary
from source.plot import plot_pairplot, plot_correlation, plot_pca

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

# %%
x = pure.iloc[:, :-1]
y = pure.iloc[:, -1]
name = pure.iloc[:, 0].name

# %%
for col in x:
  if x[col].dtype == 'object':
    m = {old_v: new_v for new_v, old_v in enumerate(x[col].unique())}
    x[col] = x[col].map(m)

y = y.map({0: 'f', 1: 't'})

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
  to_remove = x.loc[mask, name]
  
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
y_pred = KNN(x_train, y_train, 6, x_test ,euclidean_distance)
plot_confusion_matrix(y_test, y_pred)
print_summary(y_test, y_pred)

# %%
KNN_keras = KNeighborsClassifier(n_neighbors=6)
KNN_keras.fit(x_train, y_train)
y_pred = KNN_keras.predict(x_test)

plot_confusion_matrix(y_test, y_pred)
print_summary(y_test, y_pred)

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
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

y_pred_train = tree.predict(x_train)
# y_pred_test = tree.predict(x_test)

plot_confusion_matrix(x_train, y_pred_train)
# print_summary(x_train, y_pred_train)

# plot_confusion_matrix(y_test, y_pred_test)
# print_summary(y_test, y_pred_test)

# plot_tree(tree)

# %%
y_train
# %%
