#!/usr/bin/python3

# pip3 install pandas nltk xlrd sklearn tweepy textblob nltk numpy pytest odfpy matplotlib
import numpy as np
import pytest
from pandas.tests.groupby.test_value_counts import df
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
# from emoticons import (positive_emoticons, negative_emoticons, positive_sentiment, negative_sentiment)
from nltk.tokenize import word_tokenize
import string
from operator import itemgetter
import xlrd
import pandas as pd
from sklearn.svm import SVC
import tweepy
import os
# os.system('pip3 install wxpython')
# import wxpython

# Função pra pegar horario atual utc 3
def horario():
    from datetime import datetime, timezone, timedelta, timezone
    global momento

    data_e_hora_atuais = datetime.now() # Pega o horario
    data_e_hora_em_texto = data_e_hora_atuais.strftime("%d/%m/%Y-horas-%H:%M") # Ajuta pra ficar mais bonito

    diferenca = timedelta(hours= -3) # Define a diferança de acordo com São Paulo
    fuso_horario = timezone(diferenca) # Ajustando o horario para o horario de São Paulo

    data_e_hora_sao_paulo = data_e_hora_atuais.astimezone(fuso_horario)
    momento = data_e_hora_sao_paulo.strftime("%d-%m-%Y-horas-%H:%M") # A varivel momento fica com o horario atual fomratado


text_positivo = []

print('Hellow word')

search = "rock"  # A variavel search recebe o input de pesquisa e já é concatenado para não receber retweets
print(search)

# Credenciais
consumer_key = ''
consumer_secret = ''

access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

Tfrase = []
frase = ""
str(Tfrase)
i = 1

from textblob import TextBlob

# Aqui foi criado um loop para pesquisar e imprimir os tweets
for tweet in tweepy.Cursor(api.search, q=search + ' -filter:retweets', count=100,
                           lang='pt',
                           since='2020-06-19').items(10):

    tweett = tweet.text
    frase = TextBlob(tweet.text)

    # print('Tweet', i, ': ', tweet.text)  # Mostra o tweets
    Tfrase.append(tweet.text)

    i = i + 1
    
    # quando executar a primeira vez vai preciesar digitar a linha com a ! no terminal
    # import nltk !
    # nltk.download()
    # nltk.download('punkt')
    # nltk.download('sotpwords') !
    
    classificator = SVC(kernel='linear')

    # Base de dados
    training_set = pd.read_excel(r'/home/luciano/scripts/Informado/Dataset.ods')
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords

    # Limpa os dados deixando mais facil de ser analisado
    tf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'),
                                    analyzer='word', ngram_range=(1, 1),
                                    lowercase=True, use_idf=True,
                                    strip_accents='unicode')
    
    # Nessas proximas duas linha a I.A. aprende a ler e analisar
    feature = tf_vectorizer.fit_transform(training_set['Text'])  # Neste ela le a base de dados (Está apresentando um warning)
    classificator.fit(feature, training_set['Classificator'])  # Neste ela classifica o sentimento

    vector_tweets = tf_vectorizer.transform(Tfrase)  # Aqui passo o que ela analisar (tweets)
    predictions = classificator.predict(vector_tweets)  # Aqui ela classifica
    
    print('----------------------------------------------------------------------')
# Essas variaveis servem pra contar quantos de cada sentimento tem
positivo = 0
neutro = 0
negativo = 0

print('-------Tweets-------')
#Tfrase.drop_duplicates([Tfrase], inplace=True)
for prediction, tweet in zip(predictions, Tfrase):  # Neste lopp mostra o texto e a classificação
    # print(f'{tweet} -> {prediction}')
    print('')
    if prediction == 'negativo':
        print(f'{tweet} -> {prediction}')
        negativo = negativo + 1
    if prediction == 'positivo':
        
        text_positivo.append(tweet)

        print(f'{tweet} -> {prediction}')
        positivo = positivo + 1
    if prediction == 'neutro':
        print(f'{tweet} -> {prediction}')
        neutro = neutro + 1
print('')

# Nessas proximas linha contam o total de textos analisados e mostra quantidade de cada sentimento
soma = positivo + neutro + negativo
print("Positivo:", positivo, "|Neutro: ", neutro, "|Negativo: ", negativo)
print("Total: ", soma)

quantidade = [positivo, neutro, negativo]

# Essas ultimas linhas mostram o grafico
# import matplotlib.pyplot as plt

# labels_list = ['Positivo', 'Neutro', 'Negativo']

# plt.figure(figsize=(7, 7))
# plt.pie(x=quantidade, labels=labels_list, autopct='%1.1f%%')

# plt.title('Frequencias')
# plt.show()

print('Apenas tweets positivos')

resultado1= ''
resultado_final=''

i = 0

total_text_positivo = len(text_positivo)

while i < total_text_positivo:
    resultado1 = text_positivo[i]
    resultado_final = resultado_final + resultado1 + '\n'
    
    i = i + 1


resultado_final = "".join([str(_) for _ in resultado_final])

txt_positivo = open('/home/luciano/scripts/Informado/Informado-zabbix.txt','w')
txt_positivo.write(resultado_final)


txt_positivo.close()

print(resultado_final)

horario()

texto_log = 'Script executado as: ', momento, '\n'
texto_log = "".join([str(_) for _ in texto_log])

log = open('/home/luciano/scripts/Informado/Informado.log', 'a')
log.write(texto_log)
log.close()
