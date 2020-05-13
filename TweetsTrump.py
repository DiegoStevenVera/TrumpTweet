# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:52:32 2020

@author: diego
"""

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

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

# Resetear index debido a la eliminación de filas sin valores, se agregará 
# una columna 'index' con los valores anteriores del index, que se eliminarán
tweets = tweets.reset_index()

# Eliminación de columnas
drop_col = ['Event Number', 'SentiStrength', 'Code', 'index']
tweets = tweets.drop(drop_col, axis=1)

# Características
print(tweets.columns)

# Análisis de sentimientos
# Polárity rango: [-1 : 1] = [negativo : positivo]
# Subjectivity rango: [0 : 1] = [objetivo : subjetivo]
tweets['Polarity'] = tweets['Tweet'] \
                    .apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
                    
tweets['Subjectivity'] = tweets['Tweet'] \
                        .apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
                        
# Hist de los datos
plt.hist(tweets['Polarity'], 10, alpha=0.5, label='Polarity')
plt.hist(tweets['Subjectivity'], 10, alpha=0.5, label='Subjectivity')
plt.legend(loc='upper right')
plt.show()

tweets = tweets.sort_values(['Company name'])

# Empresas que no tiene: Aetna, andeavor

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
                                     header=0, usecols=['Date','Open'])}
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
                    <Data> : <Dataframe con datos de la empresa>,
                    <Std Open>: <Valor de la desviación estándar del valor 
                                    del índice de apertura, 'Open'>}
                }
    
"""
