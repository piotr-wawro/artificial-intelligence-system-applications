# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.multivariate.pca import PCA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

from source.KNN import KNN, euclidean_distance

# %%
pure = pd.read_csv(Path('./source/data/happiness/2019.csv'))
pure.columns = [x.replace(' ', '_') for x in pure.columns]

x = pure.iloc[:, 3:]
y = pure.iloc[:, 2]

data = pd.concat([x,y], axis=1)

# %%
grid = sns.pairplot(data)
grid.map_lower(sns.kdeplot, levels=1, color=".8")

# %%
sm_x = sm.add_constant(x)
model = sm.OLS(y, sm_x)
result = model.fit()
result.summary()

# %%
(distance, p_value) = OLSInfluence(result).cooks_distance
g = sns.histplot(distance)
g.set_xticks(np.linspace(0, 0.2, 11))
g.set_xticklabels(g.get_xticklabels(), rotation=30)

# %%
to_delete = []
for i, d in enumerate(distance):
    if d > 0.03:
        to_delete.append(i)

print(pure.iloc[to_delete, 1])

x = pure.iloc[:, 3:].drop(to_delete)
y = pure.iloc[:, 2].drop(to_delete)

data = pd.concat([x,y], axis=1)

# %%
pca_model = PCA(x, 3, gls=True)
pca_model.plot_scree(log_scale=False)

# %%
pca = sklearnPCA()
pca.fit(x)

total_sum = np.cumsum(pca.explained_variance_ratio_)

plt.plot(total_sum)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# %%
for col in data:
    sns.histplot(data[col])
    plt.show()

# %%
sns.boxplot(data=x)

# %%
outliers = boxplot_stats(pure.iloc[:, [8]].values)[0]['fliers']
selected_rows = pure.iloc[:, 8].isin(outliers)
pure.loc[selected_rows].iloc[:, 1]

# %%
correlation = data.corr(method='person')
sns.heatmap(correlation, annot=True)

# %%
q1 = y.quantile(0.25)
q2 = y.quantile(0.5)
q3 = y.quantile(0.75)

# %%
select_q1 = y < q1
select_q23 = (y >= q1) & (y <= q3)
select_q3 = y > q3

# %%
y.loc[select_q1] = 'low'
y.loc[select_q23] = 'mid'
y.loc[select_q3] = 'high'
# data[y.name] = y

# %%
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

# %%
pred_y = KNN(train_x, train_y, 6, test_x ,euclidean_distance)

# %%
p = 0
n = 0
for a,b in zip(test_y, pred_y):
    if a == b:
        p+=1
    else:
        n+=1

# %%

KNN_keras = KNeighborsClassifier(n_neighbors=6)
KNN_keras.fit(train_x, train_y)

# %%
pred_y = KNN_keras.predict(test_x)


# %%
p = 0
n = 0
for a,b in zip(test_y, pred_y):
    if a == b:
        p+=1
    else:
        n+=1

# %%
