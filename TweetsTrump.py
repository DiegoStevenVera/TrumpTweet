# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:52:32 2020

@author: diego
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sns.set(style="ticks", color_codes=True)

""" PREPARACIÓN DE LOS DATOS DE LOS TWEETS """

tweets = pd.read_excel('./tweets_trump.xlsx', header=0)

# Eliminar todas las filas sin ningún valor
tweets = tweets.dropna(how='all')

# Algoritmo para poner valores en los bloques NA que poseían un Tweet
for col in tweets.columns:
    if tweets[col].isnull().any():
        for i,null in zip(tweets[col].isnull().index,tweets[col].isnull()):
            if not null:
                last_var = tweets[col][i]
            else:
                tweets[col][i] = last_var

# Con los datos que se pudo recopilar en Yahoo Finance, no se encontraron
# de las empresas Aetna y andeavor, estas empresas serán eliminadas del dataset

index_to_elim = pd.concat([tweets[tweets['Company name'] == 'Aetna'],
                           tweets[tweets['Company name'] == 'Andeavor']]).index
tweets = tweets.drop(index_to_elim)

# Resetear index debido a la eliminación de filas sin valores, se agregará 
# una columna 'index' con los valores anteriores del index, que se eliminarán
tweets = tweets.reset_index()

# Normalizar los valores de la característica "Opening/Closing", debido 
# a que algunos tenían espacios en blancos y estaban en mayusculas y otros
# en minusculas
tweets['Opening/Closing'] = tweets['Opening/Closing'] \
                                    .apply(lambda x: x.lower().strip())

# Eliminación de columnas
drop_col = ['Event Number', 'SentiStrength', 'Code', 'index']
tweets = tweets.drop(drop_col, axis=1)

# Características
print(tweets.columns)

""" 
ANÁLISIS DE SENTIMIENTOS CON TEXT BLOB

 Polárity rango: [-1 : 1] = [negativo : positivo]
 Subjectivity rango: [0 : 1] = [objetivo : subjetivo]

"""

tweets['Polarity'] = tweets['Tweet'] \
                    .apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    
tweets['Subjectivity'] = tweets['Tweet'] \
                        .apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                        
# Hist de los datos
plt.hist(tweets['Polarity'], 10, alpha=0.5, label='Polarity')
plt.hist(tweets['Subjectivity'], 10, alpha=0.5, label='Subjectivity')
plt.legend(loc='upper right')
plt.show()

tweets = tweets.sort_values(['Polarity'])

""" PREPARACIÓN DE LOS DATOS DE LA BOLSA DE VALORES """

# Obtención de los datos de los índices de las empresas
# Nombres de las empresas que se usará como key para el dict que se hará
business_name = ['Alibaba', 'Apple', 'American Airline', 'Amazon', 'AT&T', \
                'Bayer AG', 'Boeing', 'Broadcom', 'Charter Communications inc', \
                'Corning', 'Delta Air Lines', 'Disney', 'Exxon Mobil', \
                'Facebook', 'Fiat Chrysler', 'Ford Motor', 'General Motors', \
                'Goldman Sachs', 'Harley Davidson', 'Humana', 'Intel', \
                'JP Morgan Chase', 'Kroger', 'Lockheed Martin', 'Morgan Stanley', \
                'Merck', 'Nike', 'Nordstrom', 'Novartis', 'Pfizer', 'Rexnord',\
                'Samsung Electronics', 'Softbank', 'Soutwest Airlines', 'Sprint', \
                'Target', 'Toyota Motor', 'Transcanada', 'Twitter', \
                'United Technologies', 'Wells Fargo', 'Walmart']
    
business = {}
    
for company in business_name:
    business[company] = {'data': pd.read_csv('./empresas/'+company+'.csv', 
                                     header=0)}
    #, usecols=['Date','Open']
    date = business[company]['data']['Date']
    business[company]['data'].drop('Date', axis='columns')
    business[company]['data']['Date'] = pd.to_datetime(date)
    business[company]['Std open'] = business[company]['data'] \
                           [business[company]['data']['Date'] < \
                           tweets[tweets['Company name'] == company]['Date'].min()] \
                           ['Open'].std()
    

"""
Ahora todas las empresas con la información que se necesita está en el 
diccionario, están con la estructura siguiente:
    business = { <Nombre de la empresa>: {
                    'data' : <Dataframe con datos de la empresa>,
                    'Std Open': <Valor de la desviación estándar del valor 
                                    del índice de apertura, 'Open'>}
                }
    
"""

""" CONSTRUCCIÓN DE LA MATRIZ FINAL """

# Función para obtener datos de los precios según:
# Input = name_company: Nombre de la empresa del que se quiere los datos, 
#         date: Fecha de la cual se quiere los datos
# Output = un Dataframe que contiene los precios de la empresa indicada
#           del día anterior, el día actual y el día posterior de la fecha
#           indicada

def data_of(name_company, date):
    data_company = business[name_company]['data']
    before = data_company[data_company['Date'] <= date] \
                            .sort_values('Date', ascending = False).head(2)
    after = data_company[data_company['Date'] > date].sort_values('Date').head(1)
    return pd.concat([after,before])


# Ahora con la función se puede obtener la diferencia de los precios con respecto
# al día posterior con el actual, esto para conocer si el precio subió o bajó
# el día posterior, se colocará todo en variations

variations = []

for company, date in tweets[['Company name', 'Date']].values:
    price_company_date = data_of(company, date).head(2)
    # Variación del incremento o decremento del precio de apertura del 
    # día posterior con respecto al actual
    prices = price_company_date['Open']
    variation = prices.iloc[0] - prices.iloc[1]
    variations.append(1 if variation > 0 else 0)

# La lista creada se agregará al dataframe de tweets para obtener el label Y
# que necesitamos para la predicción, este nos dirá si el precio subió o bajó
tweets['Variation'] = variations

# Gráfico, muestra la negatividad o positividad del tweet con respecto
# si el precio bajó o subió
plot = sns.catplot(x="Variation", y="Polarity", kind="swarm", data=tweets)
plot.set(xticklabels=['Descent', 'Ascent'])

# Se agrega la desviación estándar a la matriz final "tweets"
std_open = [business[i]['Std open'] for i in tweets['Company name']]
tweets['Std open'] = std_open

Y = tweets['Variation']
X = tweets[['Company name', 'Opening/Closing', 'Polarity', 'Subjectivity', 'Std open']]

""" CONSTRUCCIÓN DEL MODELO """

""" MODELO CON TEXT BLOB """

X_label = X.drop(['Company name', 'Opening/Closing'], axis='columns')

label_encoder = LabelEncoder()

for col in ['Company name', 'Opening/Closing']:
    label_encoder.fit(X[col])
    X_label[col] = label_encoder.transform(X[col])

train_X, val_X, train_y, val_y = train_test_split(X_label, Y, random_state = 0,
                                                  test_size = 0.2)

# Sin cross validation
model = RandomForestClassifier(n_estimators=360, max_depth=3, random_state=0)
model.fit(train_X, train_y)
model.score(val_X, val_y)

# con cross validation

list_scores = []

for i in range(100, 620, 20):
    print(i)
    model = RandomForestClassifier(n_estimators=i, max_depth=3, random_state=0)
    scores = cross_val_score(model, X_label, Y, cv=5)    
    list_scores.append(scores.mean())
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

RF_TB = plt.subplot()
RF_TB.set_xlabel('n_estimators')
RF_TB.set_ylim(0.58, 0.72)
RF_TB.set_ylabel('Precisión')
RF_TB.plot(list(range(100, 620, 20)), list_scores, label='Gini')

list_scores = []

for i in range(100, 620, 20):
    print(i)
    model = RandomForestClassifier(n_estimators=i, criterion='entropy',\
                                   max_depth=3, random_state=0)
    scores = cross_val_score(model, X_label, Y, cv=5)    
    list_scores.append(scores.mean())
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
RF_TB.plot(list(range(100, 620, 20)), list_scores, label='Entropy')
RF_TB.legend(loc='upper rigth', shadow=True)

""" MODELO CON VADER CON VALORES PUROS """

analizer = SentimentIntensityAnalyzer()
tweet_Vader = tweets.drop(['Polarity', 'Subjectivity'], axis='columns')

tweet_Vader['Negative'] = tweet_Vader['Tweet'] \
                    .apply(lambda x: analizer.polarity_scores(x)['neg'])

tweet_Vader['Neutro'] = tweet_Vader['Tweet'] \
                    .apply(lambda x: analizer.polarity_scores(x)['neu'])

tweet_Vader['Positive'] = tweet_Vader['Tweet'] \
                    .apply(lambda x: analizer.polarity_scores(x)['pos'])

tweet_Vader['Compound'] = tweet_Vader['Tweet'] \
                    .apply(lambda x: analizer.polarity_scores(x)['compound'])

Yv = Y.copy()
Xv = tweet_Vader[['Std open', 'Negative', 'Positive', 'Neutro', 'Compound']]

label_encoder = LabelEncoder()

for col in ['Company name', 'Opening/Closing']:
    label_encoder.fit(tweet_Vader[col])
    Xv[col] = label_encoder.transform(tweet_Vader[col])

train_X, val_X, train_y, val_y = train_test_split(Xv, Yv, random_state = 0,
                                                  test_size = 0.2)

# Sin cross validation
modelV = RandomForestClassifier(n_estimators=360, max_depth=3, random_state=0)
modelV.fit(train_X, train_y)
modelV.score(val_X, val_y)

# Con cross validation

list_scoresV = []

for i in range(100, 620, 20):
    print(i)
    modelV = RandomForestClassifier(n_estimators=i, max_depth=3, random_state=0)
    scores = cross_val_score(modelV, Xv, Yv, cv=5)    
    list_scoresV.append(scores.mean())
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

RF_VS = plt.subplot()
RF_VS.set_xlabel('n_estimators')
RF_VS.set_ylim(0.58, 0.72)
RF_VS.set_ylabel('Precisión')
RF_VS.plot(list(range(100, 620, 20)), list_scoresV, label='Gini')

list_scoresV = []

for i in range(100, 620, 20):
    print(i)
    modelV = RandomForestClassifier(n_estimators=i, criterion='entropy',\
                                    max_depth=3, random_state=0)
    scores = cross_val_score(modelV, Xv, Yv, cv=5)    
    list_scoresV.append(scores.mean())
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

RF_VS.plot(list(range(100, 620, 20)), list_scoresV, label='Entropy')
RF_VS.legend(loc='upper rigth', shadow=True)

""" MODELO CON VADER CON SOLO COMPOUND """

tweet_VaderC = tweets.drop(['Polarity', 'Subjectivity'], axis='columns')

# Si compound >= 0.05 es positivo
# Si 0.05 > compound > -0.05 es neutro
# Si -0.05 >= compound es negativo

tweet_VaderC['Sentiment'] = tweet_Vader['Tweet'] \
                    .apply(lambda x: "pos" \
                           if analizer.polarity_scores(x)['compound'] >= 0.05 \
                               else "neu" \
                                   if analizer.polarity_scores(x)['compound'] > -0.05 \
                                       else "neg")

YvC = tweet_VaderC['Variation']
XvC = tweet_VaderC.drop(['Company name', 'Date', 'Time (EST)', 'Tweet', \
                         'Opening/Closing', 'Variation', 'Sentiment'], axis='columns')

label_encoder = LabelEncoder()

for col in ['Company name', 'Opening/Closing', 'Sentiment']:
    label_encoder.fit(tweet_VaderC[col])
    XvC[col] = label_encoder.transform(tweet_VaderC[col])
    
train_X, val_X, train_y, val_y = train_test_split(XvC, YvC, random_state = 0,
                                                  test_size = 0.2)

# Sin cross validation
modelVC = RandomForestClassifier(n_estimators=360, max_depth=3, random_state=0)
modelVC.fit(train_X, train_y)
modelVC.score(val_X, val_y)

# Con cross validation

list_scoresVC = []

for i in range(100, 620, 20):
    print(i)
    modelVC = RandomForestClassifier(n_estimators=i, max_depth=3, random_state=0)
    scores = cross_val_score(modelVC, XvC, YvC, cv=5)    
    list_scoresVC.append(scores.mean())
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

RF_VSC = plt.subplot()
RF_VSC.set_xlabel('n_estimators')
RF_VSC.set_ylim(0.58, 0.72)
RF_VSC.set_ylabel('Precisión')
RF_VSC.plot(list(range(100, 620, 20)), list_scoresVC, label='Gini')

list_scoresVC = []

for i in range(100, 620, 20):
    print(i)
    modelVC = RandomForestClassifier(n_estimators=i, criterion='entropy', \
                                     max_depth=3, random_state=0)
    scores = cross_val_score(modelVC, XvC, YvC, cv=5)    
    list_scoresVC.append(scores.mean())
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    
RF_VSC.plot(list(range(100, 620, 20)), list_scoresVC, label='Entropy')
RF_VSC.legend(loc='upper rigth', shadow=True)
