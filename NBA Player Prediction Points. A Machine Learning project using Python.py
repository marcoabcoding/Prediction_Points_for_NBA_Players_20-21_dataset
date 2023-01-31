#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[290]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


# ## Request players Stats table

# The data was read through the url using the 'requests' import.

# In[291]:


import requests
req = requests.get("https://www.basketball-reference.com/leagues/NBA_2021_totals.html").content


#                        All meanings of the variables can be found in the data table on the website.

# # Data without repeating players

# In[292]:


soup2 = BeautifulSoup(req, 'html.parser')


# This module defines an HTMLParser class that serves as a basis for parsing text files formatted in HTML (HyperText Mark-up Language) and XHTML.

# In[293]:


### Data without repetition

dados_sr = soup2.findAll('tr', class_= "full_table")
len(dados_sr)


# Separating the data found on the site, ' tr ' refers to the beginning of the table and ' full_table ' to non-repeating data.

# In[294]:


for tabela in dados_sr:
  for character in tabela:
    if(character.string == None):
      character.string = "0"


# A 'for' was created to fill the NA values found in the table with 0.

# Verifying if NA's now are zero with one sample.

# In[295]:


dados_sr[12].get_text(',')


# In[296]:


dados_sr = [character.get_text(',').split(',') for character in dados_sr ]


# Using the '.get.text' and '.split' commands, selecting the website texts and separating them respectively. Now it is possible to create a DataFrame.

#  Extrating titles

# In[297]:


titles = soup2.find('tr')
header = [id for id in titles.stripped_strings if id != '\n']
print(header)


# Another ' for ' was created to select and separate each value from the table

# In[298]:


dados_sr = pd.DataFrame(dados_sr,  columns= header)
dados_sr.drop("Rk", axis=1, inplace = True)
dados_sr[0:5]


# 
# # Data repeating players

# In[299]:


soup = BeautifulSoup(req, 'html.parser')
#soup.prettify()


# In[300]:


# Data repeating players

dados_cr = soup.findAll('tr', attrs={"class": ["italic_text partial_table", "full_table"]})
len(dados_cr)


# With repetition it was necessary to insert two classes. It is also noted that the size of the table was 23.4% larger than the first.

# In[301]:


for tabela in dados_cr:
  for character in tabela:
    if(character.string == None):
      character.string = "0"


# In[302]:


dados_finais_cr = [ character.get_text(',').split(',') for character in dados_cr]


# In[303]:


titles = soup.find('tr')
header = [id for id in titles.stripped_strings]
print(header)


# In[304]:


dados_cr = pd.DataFrame(dados_finais_cr, columns= header)
#dados_cr.drop("Rk", axis=1, inplace = True)
dados_cr[0:5]


# # Exploratory Analysis - without repeating players

# In[305]:


names = dados_sr.columns
names


# In[306]:


dados_sr.info()


# In[307]:


#dados_srp['Age'] = dados_srp['Age'].apply(np.int64)
#dados_srp['PTS'] = dados_srp['PTS'].apply(np.int64)
#dados_srp['PTS'] = dados_srp['PTS'].apply(np.int64)
#dados_srp['MP'] = dados_srp['MP'].apply(np.int64)...


# In[308]:


#import re
#for ind in dados_srp:
#  if re.match('^\d+$', dados_srp[ind][0]):
#    dados_srp[ind] = pd.to_numeric(dados_srp[ind])
#  elif re.match('^.\d+$', dados_srp[ind][0]):
#    print(2)
#    dados_srp[ind] = pd.to_numeric(dados_srp[ind])
#dados_srp.info()


# ## Transforming Dtypes for analysis

# In[309]:


import re
for ind in dados_sr:
  if re.match('^\d+$', dados_sr[ind][0]):
    dados_sr[ind] = pd.to_numeric(dados_sr[ind])
  elif re.match('^.\d+$', dados_sr[ind][0]):
    dados_sr[ind] = pd.to_numeric(dados_sr[ind])
dados_sr.info()


# ## Players Age

# In[403]:


sns.displot(dados_sr['Age'], kde=True,alpha = 0.5,bins=21)


# It is possible to identify by the histogram that most players are between 21 and 25 years old.

# ## Position x Points

# In[311]:


sns.boxplot(data = dados_sr, x = 'Pos', y = 'PTS');


# ## Minutes Played x Points

# In[312]:


sns.lmplot(x = 'MP' ,y='PTS', data = dados_sr);


# ## Field Goals x Points

# In[313]:


sns.lmplot(data = dados_sr, x = 'FG', y = 'PTS');


# ## Field Goals Attempts x Points 

# In[314]:


sns.lmplot(data = dados_sr, x = 'FGA', y = 'PTS');


# ## Turnovers x Points

# In[409]:


sns.scatterplot(data = dados_sr, x = 'TOV', y = 'PTS');


# ## Minutes played per games X Points

# In[408]:


dados_sr['MP/G'] = dados_sr['MP'] / dados_sr['G']
sns.scatterplot(data = dados_sr, x = 'MP/G', y = 'PTS');


# ## Points per Team

# In[318]:


plt.figure(figsize = (14,7))
sns.boxplot(data = dados_sr, x = 'Tm', y = 'PTS');


# ## Points per Age

# In[413]:


plt.figure(figsize = (14,7))
sns.boxplot(data = dados_sr, x = 'Age', y ='PTS');


# ## Variables correlation with Points

# In[320]:


plt.figure(figsize = (10,10))
sns.heatmap(dados_sr.corr()['PTS'].to_frame(), annot = True,cmap="Spectral");


# ## Variables Heatmap

# In[321]:


plt.figure(figsize = (18,18))
sns.heatmap(dados_sr.corr(), annot = True,cmap ="YlOrBr");


# ## Total points per team

# In[322]:


times = dados_sr['Tm'].unique()
times


# In[323]:


sep_times = dados_sr.groupby('Tm').groups
for time in sep_times:
  sep_times[time] = sum(sep_times[time])

valores = sep_times.values()
indice = sep_times.keys()
pt_por_time = pd.DataFrame(data = valores, index = indice)
pt_por_time.reset_index(inplace=True)
pt_por_time.columns = ['Team','Points']


# In[324]:


import matplotlib.pyplot as plt
plt.figure(figsize = (20,5))
sns.barplot(x = pt_por_time['Team'],y=pt_por_time['Points']);


# In[325]:


cont = 0
for i in dados_sr['Tm']:
  if i == 'TOT':
    cont+=1
cont


# ## Players with highest score points

# In[326]:


maiores_pontuadores = dados_sr['PTS'].nlargest(10)
maiores_pontuadores = maiores_pontuadores.reset_index()
lista = maiores_pontuadores['index'].to_list()
maiores_frame = dados_sr.loc[lista]
maiores_frame['PTP'] = str('nan')
maiores_frame.reset_index(inplace= True)
maiores_frame
for player in range( 0,len(maiores_frame)):
  maiores_frame.at[player,'PTP'] = maiores_frame.iloc[player]['Player'] + '\n' + maiores_frame.iloc[player]['Pos'] + '\n' + maiores_frame.iloc[player]['Tm']


# In[327]:


plt.figure(figsize = (20,5))
sns.barplot(data = maiores_frame, x = 'PTP', y = 'PTS');


# ## Total Points per Position

# In[328]:


dois_times = dados_sr[dados_sr['Tm'] == 'TOT']
dados_sr.drop(dois_times.reset_index()['index'], inplace = True)
plt.figure(figsize = (20,5))
# sns.barplot(x = pt_por_time['Time'],y=pt_por_time['Pontos']);


# In[329]:


sep_pos = dados_sr.groupby('Pos').groups
for pos in sep_pos:
  sep_pos[pos] = sum(sep_pos[pos])

valores = sep_pos.values()
indice = sep_pos.keys()
pt_por_pos = pd.DataFrame(data = valores, index = indice)
pt_por_pos.reset_index(inplace=True)
pt_por_pos.columns = ['Pos','Pontos']

plt.figure(figsize = (10,5))
sns.barplot(x = pt_por_pos['Pos'],y=pt_por_pos['Pontos']);


# # Data Manipulation to create a Supervised Machine Learning Model

# Was use the non-repetiton Players data.

# In[330]:


#dados_sr.drop('FGA',inplace = True, axis='columns')
dados_sr.drop('3PA',inplace = True, axis='columns')
dados_sr.drop('3P',inplace = True, axis='columns')
dados_sr.drop('2P',inplace = True, axis='columns')
dados_sr.drop('2PA',inplace = True, axis='columns')
dados_sr.drop('FT',inplace = True, axis='columns')
dados_sr.drop('FTA',inplace = True, axis='columns')
#dados_sr.drop('Rk',inplace = True, axis='columns')
dados_sr.drop('FG',inplace = True, axis='columns')


# In[331]:


dados_sr.drop('FG/MIN',inplace = True, axis='columns')
dados_sr.drop('MP/G',inplace = True, axis='columns')


# ## A Different Numerical Heatmap

# In[332]:


dados_sr.corr().style.background_gradient(cmap='BrBG').set_precision(2)
# 'RdBu_r' & 'coolwarm' are other good diverging colormaps


# In[333]:


dados_sr.info()


# In[334]:


dados_sr


# In[335]:


dados_sr.drop('Player',inplace = True, axis='columns')
dados_sr.drop('Pos',inplace = True, axis='columns')
dados_sr.drop('Tm',inplace = True, axis='columns')


# In[336]:


dados_sr.info()


# In[337]:


dados_sr


# # Training, testing and comparing different models - without fine-tuning

# In[338]:


X = dados_sr.iloc[:, :-1].values
y = dados_sr.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                     test_size = 0.3)


# In[339]:


X_train.shape


# In[340]:


X_test.shape


# In[341]:


from sklearn import svm
model = svm.SVC(kernel='rbf',C=30,gamma='auto')
model.fit(X_train,y_train)
model.score(X_test, y_test)


# In[342]:


from sklearn.ensemble import RandomForestClassifier

rdclassifier = RandomForestClassifier()
rdclassifier.fit(X_train, y_train)
rdclassifier.score(X_test, y_test)


# In[343]:


print(rdclassifier.feature_importances_)


# In[344]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
reg.score(X_test, y_test)


# In[345]:


from sklearn.linear_model import LogisticRegression
LRclassifier = LogisticRegression(solver='liblinear')
LRclassifier.fit(X_train, y_train)
LRclassifier.score(X_test, y_test)


# In[346]:


from sklearn.neighbors import KNeighborsClassifier

KNclassifier = KNeighborsClassifier(n_neighbors=8)
KNclassifier.fit(X_train, y_train)
KNclassifier.score(X_test, y_test)


# In[347]:


from sklearn.tree import DecisionTreeClassifier

DTclassifier = DecisionTreeClassifier()
DTclassifier.fit(X_train, y_train)
DTclassifier.score(X_test, y_test)


# In[348]:


from sklearn.naive_bayes import GaussianNB
NBclassifier = GaussianNB()
NBclassifier.fit(X_train, y_train)
NBclassifier.score(X_test, y_test)


# # Gridsearch used for fine-tuning
# 

# In[349]:


from sklearn.model_selection import GridSearchCV


#  # Random forest 

# In[350]:


# Number of trees in random forest - generate 10 numbers from 1 to 80.
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[351]:


param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)


# In[352]:


rf_Grid = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
rf_Grid.fit(X, y)
rf_Grid.best_params_


# In[353]:


print (f'Train Accuracy - : {rf_Grid.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(X_test,y_test):.3f}')


# In[415]:


rf_Grid.score(X_test, y_test)


# # KNeighborsRegressor

# In[354]:


X = dados_sr.iloc[:, :-1].values
y = dados_sr.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                     test_size = 0.3)


# In[355]:


from sklearn.neighbors import KNeighborsRegressor
KNclassifier = KNeighborsRegressor()
KNclassifier.fit(X_train, y_train)
KNclassifier.score(X_test, y_test)


# In[356]:


parametros = {'n_neighbors': [3,5,7,12,20,50,100]}

modelo = GridSearchCV(KNeighborsRegressor(),parametros)
modelo.fit(X, y)


# In[357]:


# modelo.cv_results_
modelo.best_estimator_


# In[358]:


# Parametros do grid 
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(param_grid)


# In[359]:


KNclassifier = KNeighborsRegressor(n_neighbors=3)
KNclassifier.fit(X_train, y_train)
KNclassifier.score(X_test, y_test)


# In[360]:


X_train.shape


# # The model used was the KneighborsRegressor because it has the highest score after fine-tuning, now let's test the model's prediction using an example from the table

# ## Prediction test

# In[361]:


dados_sr.head(1)


# In[362]:


print(modelo.predict([[ 21,  61,   4,  737,  124,  0.544,  0.0,  0.546,  0.544,  0.509,   73,  135,  208,   29, 20,   28,   43,  91 ]]))


# A 26.34 points difference, or 8.66% error.
