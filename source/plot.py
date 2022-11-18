import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def upper(x1, x2, **kwargs):
  y = kwargs['y']
  df = pd.concat([x1, x2], axis=1)
  for value in y.unique():
    sns.regplot(data=df[y == value], x=x1.name, y=x2.name, label=value)

def diag(x, **kwargs):
  y = kwargs['y']
  df = pd.concat([x, y], axis=1)
  sns.violinplot(data=df, x=x.name, y=y.name)

def lower(x1, x2, **kwargs):
  y = kwargs['y']
  df = pd.concat([x1, x2, y], axis=1)
  sns.kdeplot(data=df, x=x1.name, y=x2.name, hue=y.name, levels=1)
  sns.scatterplot(data=df, x=x1.name, y=x2.name, hue=y.name)

def plot_pairplot(x, y):
  grid = sns.PairGrid(x)
  grid.map_upper(upper, y=y)
  grid.map_diag(diag, y=y)
  grid.map_lower(lower, y=y)

def plot_correlation(x, y, model=True):
  df = pd.concat([x, y], axis=1)

  if model:
    corr = x.corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap="Blues")
  else:
    for value in y.unique():
      corr = x[y == value].corr(method='pearson')
      sns.heatmap(corr, annot=True, cmap="Blues")
      plt.figure()

def plot_pca(x):
  x_normalized = (x - x.mean()) / x.std()

  pca = PCA()
  pca.fit(x_normalized)

  total_sum = np.cumsum(pca.explained_variance_ratio_)

  plt.plot(total_sum)
  plt.xlabel('number of components')
  plt.ylabel('cumulative explained variance')
