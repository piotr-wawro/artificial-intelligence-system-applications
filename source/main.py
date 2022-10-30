# %%
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest

from source.KNN import KNN, euclidean_distance
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
pure = pd.read_csv(Path('./source/data/happiness/2019.csv'))

x = pure.iloc[:, 3:]
y = pure.iloc[:, 2]
name = pure.iloc[:, 1].name

# %%
q1 = y.quantile(0.25)
q3 = y.quantile(0.75)

select_low = y <= q1
select_mid = (y > q1) & (y <= q3)
select_high = y > q3

y[select_low] = 'low'
y[select_mid] = 'mid'
y[select_high] = 'high'

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
# Kiedy usuwać wartości odstające z Score?
# Kiedy rozdzielać dane?

# %%
clf = IsolationForest(n_estimators=round(y.size/10))
pred = clf.fit_predict(x)

mask = [True if x == -1 else False for x in pred]
to_remove = pure.loc[mask, name]

print(to_remove)
x.drop(to_remove.index, inplace=True)
y.drop(to_remove.index, inplace=True)

# %%
outliers = boxplot_stats(pure.iloc[:, [8]].values)[0]['fliers']
selected_rows = pure.iloc[:, 8].isin(outliers)
pure.loc[selected_rows].iloc[:, 1]

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# PCA

# %%
pca = sklearnPCA()



# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# Classification using KNN

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
pred_y = KNN(x_train, y_train, 6, x_test ,euclidean_distance)
plot_confusion_matrix(y_test, pred_y)
print_summary(y_test, pred_y)

# %%
KNN_keras = KNeighborsClassifier(n_neighbors=6)
KNN_keras.fit(x_train, y_train)
pred_y = KNN_keras.predict(x_test)

plot_confusion_matrix(y_test, pred_y)
print_summary(y_test, pred_y)

# %%
