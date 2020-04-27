# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:52:32 2020

@author: diego
"""

import pandas as pd

tweets = pd.read_excel('C:\\Users\\diego\\OneDrive\\Documentos\\UNAM\\'+
                       'Cursos\\Análisis y Procesamiento de textos\\'+
                           'proyecto\\tweets_trump.xlsx', header=0)

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

tweets
