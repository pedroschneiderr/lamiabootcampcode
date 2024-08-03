"""
Neste script, um modelo de Regressão Polinomial será treinado e testado com um conjunto de dados.
"""

import numpy as np
from numpy.polynomial.polynomial import Polynomial #importa a classe de polinômios da Numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

"""
Dados fictícios de tempo de carregamento de páginas web e compras realizadas nessas páginas são criados e uma relação inversamente proporcional é estabelecida entre os conjuntos
"""
np.random.seed(2) #gera sempre os mesmo número pseudo-aleatórios, para reproducibilidade do código
pageSpeeds = np.random.normal(3.0, 1.0, 100) #100 amostras de uma distribuição normal com média 3 e desvio padrão 1 para representar velocidades de carregamento de páginas web
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds #100 amostras de uma dist normal com méd 50 e desv pad 30, dividas pelas amostras de pageSpeeds (estabelecendo uma relação inversamente proporcional entre as 2 variáveis)

""" plt.scatter(pageSpeeds, purchaseAmount)
plt.show() """

"""
Agora, 20% dos dados serão separados para teste e 80% para treino
"""

trainX = pageSpeeds[:80] #pega as amostras de índice 0 até 79
testX = pageSpeeds[80:] #pega as amostras de índice 80 até 99

trainY = purchaseAmount[:80] #pega as amostras de índice 0 até 79
testY = purchaseAmount[80:] #pega as amostras de índice 80 até 99

""" plt.scatter(trainX, trainY) #plota os dados de treino
plt.show()
plt.scatter(testX, testY) #plota os dados de teste
plt.show() """

"""
Agora, um polinômio de grau 8 será ajustado aos dados de treino.
Dado que há uma relação simples inversamente proporcional entre as variáveis, é quase certo que haverá overfitting.
"""

p8 = Polynomial.fit(trainX, trainY, 8) #ajusta um polinômio de grau 8 ao conjunto de treino

""" xp = np.linspace(0, 7, 100) #valores para o eixo x (segundos de carregamento da página)
axes = plt.axes() #cria uma instância de eixos
axes.set_ylim([0, 200]) #fixa o eixo y entre 0 e 200
plt.scatter(testX, testY) #plota o conjunto de teste
plt.plot(xp, p8(xp), c='r') #plota a curva de p8 sobre os dados de teste
plt.show() #na plotagem é possível ver que o polinômio representa muito bem os dados de teste até onde os dados de treino iam, mas quando vai além deles há uma verticalização sem sentido
print(r2_score(testY, p8(testX))) #R² de 0.30018168612349405, muito ruim """

"""
Activity
Try measuring the error on the test data using different degree polynomial fits. What degree works best?
"""

"""
Ajusta e testa polinômios de graus 1 a 15, depois imprime qual foi o mais preciso:
"""

r2_scores = []
for i in range(1, 16):
    p = Polynomial.fit(trainX, trainY, i)
    r2 = r2_score(testY, p(testX))
    print(f'Grau {i}: {r2}')
    r2_scores.append(r2)

print(f'O estimador mais preciso foi o de grau {r2_scores.index(max(r2_scores)) + 1}')