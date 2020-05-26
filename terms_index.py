# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:42:14 2020

@author: diego
"""

import pprint

from collections import defaultdict
from gensim import models
from gensim import corpora


# Index terms

corpus = ["\subsection{Motivación}\
\IEEEPARstart{A}{portar} con las investigaciones sobre variaciones en la \
bolsa de valores para ayudar en la construcción de otros modelos que se van \
desarrollando hoy en día.", \
"\subsection{Justificación} \
Conocer las variaciones de la bolsa de valores con información pública que \
se tenga a la mano en base a tweets publicados por personas influyentes, \
como en la presente investigación Donald Trump. Esto debido a que la mayor \
parte de la información con relación a los movimientos de la bolsa de valores \
es privada. Por esta razón cualquier información que sea libre y ayude a \
predecir el movimiento de la bolsa es importante ya que “valen oro” en este ámbito.",\
"\subsection{Pregunta del problema}\
¿Cómo realizar un modelo que pronostique las variaciones en la bolsa de \
valores de una empresa con respecto a un tweet, realizado por Donald Trump, \
de boycott o de beneficio para la empresa?",\
"\subsection{Solución propuesta}\
Hacer un análisis de textos con el tweet con el fin de conocer sobre qué \
empresa se está hablando, si es de boycott o beneficio para esta y el grado \
de este tweet. Toda esta información, de varios tweets junto a los movimientos \
de la bolsa de valores de esta empresa después de este tweet, se usará para \
entrenar un modelo de machine learning con el fin de que este modelo pueda \
predecir los movimientos en la bolsa de valores con el siguiente tweet que \
se le ingrese como input.",\
"\section{Marco teórico-metodológico}\
\subsection{Antecedentes}\
En \cite{torres_2019} se habla sobre poder pronosticar los precios de \
acciones utilizando datos de los anteriores valores en las acciones de \
mercado de Apple inc. Hizo 2 modelos, de árboles aleatorios y perceptrón \
multicapa, comparándolos con el fin de conocer cuál es mejor. Los datos \
fueron recopilados por Google Finance. \
Las variables obtenidas de Google Finance fueron la fecha, el precio de \
apertura de la sesión de mercado, el precio más alto al que llegó, el más \
bajo, el precio con el que cerró la sesión de mercado y el volumen, siendo \
este último el total de operaciones realizadas. Los resultados obtenidos \
fueron que el mejor modelo para la predicción fue el de árboles aleatorios \
en comparación al perceptrón multicapa, las medidas que se utilizaron para \
la validación fueron el coeficiente de correlación, el error absoluto medio, \
el error medio cuadrático, error absoluto relativo y error cuadrático \
relativo de raíz.", \
"En \cite{patel_2015} se investigó sobre la predicción de los valores futuros \
de los índices bursátiles de CNX Nifty y S\&P BSE Sensex, los cuales están \
compuestos por las acciones más grandes y líquidas que se encuentran en la \
bolsa de valores de India. Utilizó un total de 10 años (2003 – 2012) de datos \
históricos sobre el movimiento de los índices de la bolsa.\
Esta investigación propone un enfoque de 2 etapas, en la primera utilizó \
el modelo de Regresión de vectores de soporte (SVR) y en la segunda usó \
una fusión de una red neuronal artificial (ANN), bosques aleatorios (RF) \
y SVR. Utilizó un total de 10 variables de entrada que fueron usadas en la \
primera etapa como preparación para la red neuronal en la segunda etapa. Se \
pronosticó valores de 1 día a 30 de anticipación.  \
Las medidas de validación usadas fueron el error porcentual absoluto medio \
(MAPE), error absoluto medio (MAE), error cuadrático medio relativo (rMSE) \
y error cuadrático medio (MSE).  Los resultados mostraron que mientras \
mayores sean los días de anticipación para el pronóstico, mayores serán las \
medidas de error. La tabla \ref{table_patel_a} muestra los valores de error \
para CNX Nifty y la tabla \ref{table_patel_b} de S\&P BSE Sensex.\
Como conclusión se llegó a que los modelos híbridos de 2 etapas funcionan \
mejor que los de una sola etapa.",\
"En \cite{chiong_2018} se investiga sobre la predicción en el mercado \
financiero usando un análisis de sentimientos en las noticias financieras. \
Se usó el modelo de Máquina de Vectores de Soporte (SVM), enjambre de \
partículas (PSO), búsqueda en cuadrícula (GS) y TextBlob, librería de Python, \
para el análisis de sentimientos.\
Los resultados son mostrados en la tabla \ref{table_chiong}. Se puede ver \
que los modelos resultantes al usar algoritmos de optimización de \
parámetros (SVM\_senti\_GS y SVM\_senti\_PSO) como GS o PSO, ayudaron a la \
exactitud de este, pues la mejoró en 0.1 y 0.2 puntos, respectivamente, al \
modelo original (SVM\_senti). El modelo propuesto (SVM\_senti\_PSO) fue \
mejor que otros modelos desarrollados en base a Aprendizaje profundo con \
TFIDF (DL\_tfidf) y a modelos con SVM y TFIDF (SVM\_tfidf).",\
"En la investigación hecha en \cite{kim_2014} presenta un análisis en las \
noticias financieras en Corea con el fin de ser una herramienta de apoyo en \
la toma de decisiones de inversión en base a los pronósticos de los aumentos \
y caídas en el KOSPI (Índice compuesto de precios de acciones de Corea). \
Aparte del modelo inferencial que presentaron, otro de sus aportes fue el \
de la construcción de un diccionario de palabras orientadas al análisis de \
sentimiento en el mercado de valores de Corea.\
Los resultados en el set de entrenamiento y de validación son mostrados en \
las tablas \ref{table_kim_a} y \ref{table_kim_b} respectivamente, mostrando \
que el análisis de las noticias puede ayudar a la predicción de los \
movimientos del mercado de valores y que estas noticias pueden variar la \
exactitud del modelo dependiendo del medio de comunicación del que fueron \
recopiladas, como se presenta en las diferentes medidas de exactitud que \
presentan el medio M y H.",\
"En \cite{li_2011} se centran en la predicción de los índices del HSI \
(Hang Seng Index) el cual es el principal índice bursátil chino de Hong Kong.\
En esta investigación usan información sobre noticias del mercado y data \
histórica sobre los precios de las acciones para usarlos en un sistema de \
aprendizaje multikernel. El proceso usado para la construcción del modelo fue:\
Los resultados están expresados en la tabla \ref{table_li}, donde nos \
muestra que el modelo MKL fue el mejor a comparación de los demás modelos \
que usaron datos de los precios de las acciones, sobre noticias y la \
combinación de ambos, la predicción se hizo a razón de 5 a 30 minutos de \
espera con respecto a la noticia mostrando la exactitud de cada uno de ellos.",\
"En \cite{vargas_2017} se investigó sobre la predicción del precio de las \
acciones a través de las noticias financieras y datos de los precios de las \
acciones del índice Standard \& Poor’s 500 (S\&P500). Se usaron 3 componentes \
principales para esto, el primero fue la definición del horizonte de \
predicción, en este caso se usó diaria, el segundo el intervalo de tiempo \
en la cual las noticias influyen en el precio de las acciones y el último \
es el tipo de representación de la información.\
El modelo que desarrollaron fue una Red neuronal convolucional recurrente \
(RCNN) el cual consta de 4 capas:\
Para mostrar los resultados se hicieron distintos experimentos usando \
varios modelos como SVM, redes neuronales, redes neuronales convolucionales \
y redes neuronales convolucionales recurrentes, estas 2 últimas poseen la \
misma arquitectura. También con estos modelos se usaron distintas técnicas \
como la incrustación de palabras, sentencias, eventos, el uso de indicadores \
y el uso de bolsa de palabras. En la tabla \ref{table_vargas} se muestra \
la exactitud de estos modelos con respecto al set de entrenamiento y prueba, \
siendo el más eficaz el modelo EB-CNN.\
La presente investigación tiene como finalidad construir un modelo \
capaz de inferir el movimiento del precio de apertura de las cotizaciones \
de una empresa usando estos datos junto a tweets que hayan sido publicados \
por Trump donde se hablen de estas empresas, es decir se inferirá el alza \
o la bajada del precio de apertura del día siguiente de haber publicado \
el tweet. Para poder realizar lo anterior descrito, se hará un análisis \
de sentimientos a un conjunto de tweets publicados por Trump donde mencione \
a estas empresas, esta información se agrupará con medidas obtenidas de las \
cotizaciones de estas empresas para poder usarlas en un modelo de Random \
Forest para clasificación."]

# Create a set of frequent words
stoplist = set('a \
acá \
ahí \
al \
algo \
algún \
alguna \
algunas \
alguno \
algunos \
ambos \
ampleamos \
ante \
antes \
aquel \
aquella \
aquellas \
aquello \
aquellos \
asi \
atras \
aun \
aunque \
bajo \
bastante \
bien \
cabe \
cada \
casi \
cierta \
ciertas \
cierto \
ciertos \
como \
cómo \
con \
conmigo \
conseguimos \
conseguir \
consigo \
consigue \
consiguen \
consigues \
contigo \
contra \
cual \
cuales \
cuan \
cuán \
cuando \
cuanta \
cuánta \
cuantas \
cuántas \
cuanto \
cuánto \
cuantos \
cuántos \
de \
dejar \
del \
demás \
demas \
demasiada \
demasiadas \
demasiado \
demasiados \
dentro \
desde \
donde \
dos \
el \
él \
ella \
ellas \
ello \
ellos \
empleais \
emplean \
emplear \
empleas \
empleo \
en \
encima \
entonces \
entre \
era \
eramos \
eran \
eras \
eres \
es \
esa \
esas \
ese \
eso \
esos \
esta \
estaba \
estado \
estais \
estamos \
estan \
estar \
estas \
este \
esto \
estos \
estoy \
etc \
fin \
fue \
fueron \
fui \
fuimos \
gueno \
ha \
hace \
haceis \
hacemos \
hacen \
hacer \
haces \
hacia \
hago \
hasta \
incluso \
intenta \
intentais \
intentamos \
intentan \
intentar \
intentas \
intento \
ir \
jamás \
junto \
juntos \
la \
largo \
las \
lo \
los \
mas \
más \
me \
menos \
mi \
mía \
mia \
mias \
mientras \
mio \
mío \
mios \
mis \
misma \
mismas \
mismo \
mismos \
modo \
mucha \
muchas \
muchísima \
muchísimas \
muchísimo \
muchísimos \
mucho \
muchos \
muy \
nada \
ni \
ningun \
ninguna \
ningunas \
ninguno \
ningunos \
no \
nos \
nosotras \
nosotros \
nuestra \
nuestras \
nuestro \
nuestros \
nunca \
os \
otra \
otras \
otro \
otros \
para \
parecer \
pero \
poca \
pocas \
poco \
pocos \
podeis \
podemos \
poder \
podria \
podriais \
podriamos \
podrian \
podrias \
por \
por qué \
porque \
primero \
puede \
pueden \
puedo \
pues \
que \
qué \
querer \
quien \
quién \
quienes \
quienesquiera \
quienquiera \
quiza \
quizas \
sabe \
sabeis \
sabemos \
saben \
saber \
sabes \
se \
segun \
ser \
si \
sí \
siempre \
siendo \
sin \
sín \
sino \
so \
sobre \
sois \
solamente \
solo \
somos \
soy \
sr \
sra \
sres \
sta \
su \
sus \
suya \
suyas \
suyo \
suyos \
tal \
tales \
también \
tambien \
tampoco \
tan \
tanta \
tantas \
tanto \
tantos \
te \
teneis \
tenemos \
tener \
tengo \
ti \
tiempo \
tiene \
tienen \
toda \
todas \
todo \
todos \
tomar \
trabaja \
trabajais \
trabajamos \
trabajan \
trabajar \
trabajas \
trabajo \
tras \
tú \
tu \
tus \
tuya \
tuyo \
tuyos \
ultimo \
un \
una \
unas \
uno \
unos \
usa \
usais \
usamos \
usan \
usar \
usas \
uso \
usted \
ustedes \
va \
vais \
valor \
vamos \
van \
varias \
varios \
vaya \
verdad \
verdadera \
vosotras \
vosotros \
voy \
vuestra \
vuestras \
vuestro \
vuestros \
y \
ya \
yo'.split(' '))

# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist] \
         for document in corpus]
    
# Count word frequencies
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

ids = dictionary.token2id
pprint.pprint(ids)

bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)

# train the model
tfidf = models.TfidfModel(bow_corpus)
words = "análisis de sentimientos modelo de predicción bolsa de valores donald \
trump tweet índice de apertura random forest movimiento".lower().split()
index_terms=tfidf[dictionary.doc2bow(words)]
key_terms=[term[0] for term in index_terms if term[1] > 0.3]

terms = [word for word, index in ids.items() if index in key_terms]
print(terms)
