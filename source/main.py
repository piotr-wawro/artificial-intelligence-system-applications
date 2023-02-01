# %%
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

# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# #### Wczytywanie danych
# 
# W pracy został wykorzystany zbiór danych dotyczący sprawności fizycznej.
# Znajduje się w domenie publicznej i jest dostępny na [kaggle.com][1]
# 
# W zbiorze znajduje się ponad 13 tys. rekordów. Każda próbka ma dwanaście cech.
# * age (wiek)
# * gender (płeć)
# * height_cm (wzrost)
# * weight_kg (waga)
# * body fat_% (procent tkanki tłuszczowej)
# * diastolic (rozkurczowe ciśnienie krwi)
# * systolic (skurczowe ciśnienie krwi)
# * gripForce (siła chwytu)
# * sit and bend forward_cm (skłon do przodu w pozycji siedzącej)
# * sit-ups counts (ilość przysiadów)
# * broad jump_cm (skok w dal)
# * class (klasa)
# 
# [1]: https://www.kaggle.com/datasets/kukuroo3/body-performance-data

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

# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# #### Wizualizacja danych

# %% [markdown]
# plot_pairplot pozwala na wizualizację pomiędzy parami cech z podziałem ze
# względu na płeć(niebieski - mężczyzna, pomarańczowy - kobieta). Na przekątnej
# znajdują się wykresy skrzypcowe dla każdej cechy. Górnotrójkątna część
# przedstawia wykres punktowy z regresją liniową, a dolnotrójkątna wykres
# punktowy z estymatorem jądrowym gęstości.

# %%
plot_pairplot(x, y)

# %% [markdown]
# plot_correlation zwraca współczynniki korelacji Pearsona oznaczające poziom
# zależności liniowej między zmiennymi. 1 oznacza dodatnią liniową zależność,
# -1 ujemną liniową zależność, a 0 brak liniowej zależność.

# %%
plot_correlation(x, y)

# %% [markdown]
# Aby zmniejszyć ilość cech możemy użyć analizy głównych składowych. plot_pca
# rysuje wykres przedstawiający procent wyjaśnionej wariancji względem ilości
# komponentów.

# %%
plot_pca(x)

# %% [markdown]
# Histogram pokazuje nam, ile obserwacji przypada na określony przedział
# wartości.

# %%
plot_histrogram(x, y)

# %% [markdown]
# Estymator jądrowy gęstości (ang. kernel density estimate, kde) to metoda
# wizualizacji rozkładu obserwacji w zbiorze danych, analogiczna do histogramu.

# %%
plot_kde(x, y)

# %% [markdown]
# Wykres pudełkowy przedstawia położenie, rozproszenie i kształt rozkładu
# empirycznego badanej cechy.

# %%
ax = sns.boxplot(data=x)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

# %% [markdown]
# Wykres pudełkowy ale z podziałem ze względu na płeć.

# %%
df = pd.melt(pd.concat([x, y], axis=1), id_vars=[y.name])
ax = sns.boxplot(data=df, x='variable', y='value', hue=y.name)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# #### Usuwanie wartości odstających
# 
# Obserwacje odstające są odległe od pozostałych elementów próby. Mogą być wynikiem
# błędnego pomiaru lub odzwierciedlać rzeczywisty przypadek, ale taki, który jest
# mało prawdopodobny. Takie obserwacje chcemy usuwać bo mogą negatywnie wpływać
# na jakość modelu. Poniżej znajdują się dwa algorytmy, którymi możemy je usunąć.
# 
# Drugi algorytm wykorzystuje odchylenie standardowe. Jeżeli obserwacja znajduje się
# poza przedziałem [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR], gdzie Q1 i Q3 to odpowiednio
# pierwszy i trzeci kwartyl, a IQR to rozstęp kwartylny, to taką obserwację zaliczamy
# jako odstającą.

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
# #### Normalizacja i standaryzacja
# 
# Niektóre algorytmy wymagają, aby dane zostały podjęte normalizacji lub
# standaryzacji. Normalizacja polega na przeskalowaniu danych zazwyczaj
# do wartości od 0 do 1. Standaryzacja przekształca rozkład do
# standardowego rozkładu normalnego czyli o wartości średniej 0 oraz odchyleniu
# standardowym 1.

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
# #### Analizy głównych składowych
# 
# Za pomocą analizy głównych składowych jesteśmy w stanie przekształcić układ
# współrzędnych w taki sposób, aby zmaksymalizować wariancję.
# 
# Algorytm poniżej dobierze minimalną ilość komponentów, która jest potrzebna
# do wyjaśnienia 95% wariancji.

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
# #### Klasyfikacja za pomocą KNN
# 
# KNN klasyfikuje nowe próbki biorąc pod uwagę k najbliższych sąsiadów.

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
y_pred = KNN(x_train, y_train, 3, x_test, euclidean_distance)
plot_confusion_matrix(y_test, y_pred)
print_summary(y_test, y_pred)

# %%
KNN_keras = KNeighborsClassifier(n_neighbors=5)
KNN_keras.fit(x_train, y_train)

y_pred_train = KNN_keras.predict(x_train)
y_pred_test = KNN_keras.predict(x_test)

plot_confusion_matrix(y_train, y_pred_train)
print_summary(y_train, y_pred_train)

plot_confusion_matrix(y_test, y_pred_test)
print_summary(y_test, y_pred_test)

# %% [markdown]
# Na podstawie wykresu widać, że potrzeba co najmniej 3 sąsiadów, aby uzyskać
# zadowalające wyniki. Zwiększanie liczby sąsiadów zmniejsza dokładność.
# Jeżeli korzystamy z PCA zwiększanie liczby sąsiadów nie ma większego wpływu.

# %%
data = pd.DataFrame(columns=["train_acc", "test_acc"], dtype='float')

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

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# #### Klasyfikacja za pomocą PNN
# 
# Probabilistyczne sieci neuronowe wykorzystują rozkład gęstości i prawdopodobieństwo
# w celu przypisania próbki do odpowiedniej klasy.

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
# #### Klasyfikacja za pomocą drzew decyzyjnych
# 
# Drzewa decyzyjne opierają swoje działanie na prostych warunkach logicznych.
# Wartości parametrów każdej próbki są sprawdzane, czy mieszczą się w odpowiednich
# przedziałach i na tej podstawie wybierana jest klasa.

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
# #### Istotność cech
# 
# Nie każda cecha w równym stopniu przykłada się do zwiększenia jakości modelu.
# Wybranie najistotniejszych cech pozwoli nam zmniejszyć ilość wymiarów przy
# jednoczesnym zachowaniu wysokiej sprawności modelu.
# 
# Problem ten można rozwiązać przy pomocy PCA jednak tracimy informacje
# o poszczególnych cechach.
# 
# Poniżej znajduje się kilka algorytmów, które służą do wybrania najistotniejszych
# cech. W scikit-learn implementacja drzew decyzyjnych ma zmienną feature_importances_
# dzięki czemu "za darmo" możemy dostać potrzebne informacje.
# 
# Drugi sposób polega na permutacji cechy. Mieszamy w ten sposób dane przez co
# dana cecha staje się bezużyteczna. Na wcześniej wytrenowanym modelu sprawdzamy
# jak bardzo miało to wpływ na dokładność.

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
model.fit(x_train, y_train.map({'M': 0, 'F': 1}))

plot_importance(x.columns, model.coef_[0])

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

# %% [markdown]
# #### Logika rozmyta
# 
# Metoda ta polega na rozmyciu wartości za pomocą funkcji przynależności, które
# ustala ekspert. Następnie na podstawie wartości rozmytych przeprowadzamy
# wnioskowanie wykorzystując sNormy i tNormy.

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
  m = tall(sample.height_cm) + heavy(sample.weight_kg) + strong(sample.gripForce) + jump_low(sample.broad_jump_cm)
  f = short(sample.height_cm) + light(sample.weight_kg) + weak(sample.gripForce) + jump_high(sample.broad_jump_cm)

  if m > f:
    y_pred_train.append("M")
  else:
    y_pred_train.append("F")

plot_confusion_matrix(y_train, y_pred_train)
print_summary(y_train, y_pred_train)

# %% [markdown]
# Rules
# short AND weak AND jump_low => F
# short AND weak AND jump_high => F
# short AND strong AND jump_low => F
# short AND strong AND jump_high => M
# tall AND weak AND jump_low => F
# tall AND weak AND jump_high => M
# tall AND strong AND jump_low => M
# tall AND strong AND jump_high => M

# %%
y_pred_train = []

for i, sample in x_train.iterrows():
  f = sNorm([
    tNorm([short(sample.height_cm), weak(sample.gripForce), jump_low(sample.broad_jump_cm)]),
    tNorm([short(sample.height_cm), weak(sample.gripForce), jump_high(sample.broad_jump_cm)]),
    tNorm([short(sample.height_cm), strong(sample.gripForce), jump_low(sample.broad_jump_cm)]),
    tNorm([tall(sample.height_cm), weak(sample.gripForce), jump_low(sample.broad_jump_cm)]),
  ])

  m = sNorm([
    tNorm([short(sample.height_cm), strong(sample.gripForce), jump_high(sample.broad_jump_cm)]),
    tNorm([tall(sample.height_cm), weak(sample.gripForce), jump_high(sample.broad_jump_cm)]),
    tNorm([tall(sample.height_cm), strong(sample.gripForce), jump_low(sample.broad_jump_cm)]),
    tNorm([tall(sample.height_cm), strong(sample.gripForce), jump_high(sample.broad_jump_cm)]),
  ])

  if m > f:
    y_pred_train.append("M")
  else:
    y_pred_train.append("F")

plot_confusion_matrix(y_train, y_pred_train)
print_summary(y_train, y_pred_train)

# %%
y_pred_test = []

for i, sample in x_test.iterrows():
  f = sNorm([
    tNorm([short(sample.height_cm), weak(sample.gripForce), jump_low(sample.broad_jump_cm)]),
    tNorm([short(sample.height_cm), weak(sample.gripForce), jump_high(sample.broad_jump_cm)]),
    tNorm([short(sample.height_cm), strong(sample.gripForce), jump_low(sample.broad_jump_cm)]),
    tNorm([tall(sample.height_cm), weak(sample.gripForce), jump_low(sample.broad_jump_cm)]),
  ])

  m = sNorm([
    tNorm([short(sample.height_cm), strong(sample.gripForce), jump_high(sample.broad_jump_cm)]),
    tNorm([tall(sample.height_cm), weak(sample.gripForce), jump_high(sample.broad_jump_cm)]),
    tNorm([tall(sample.height_cm), strong(sample.gripForce), jump_low(sample.broad_jump_cm)]),
    tNorm([tall(sample.height_cm), strong(sample.gripForce), jump_high(sample.broad_jump_cm)]),
  ])

  if m > f:
    y_pred_test.append("M")
  else:
    y_pred_test.append("F")

plot_confusion_matrix(y_test, y_pred_test)
print_summary(y_test, y_pred_test)
# %%
