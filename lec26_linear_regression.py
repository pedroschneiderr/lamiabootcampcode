"""
Nesse script, são criados dados fictícios de velocidade de carregamento de páginas web e compras feitas nessas páginas
Esses dados têm uma relação aproximadamente linear e regressões lineares são feitas a partir deles
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb' #se fez necessário para o funcionamento do matplotlib
import numpy as np
import matplotlib.pyplot as plt #biblioteca usada para plotagem de dados num estilo MATLAB
from scipy import stats #modulo com funções de estatística

"""
Primeiramente, os dados fictícios são criados:
pageSpeeds, que contém os dados de carregamento das páginas, é um array que recebe 1000 amostras retiradas aleatoriamente de uma distribuição normal com média 3 e desvio padrão 1.
purchaseAmount, que contém os dados de quantidade de compras feitas na página em diferentes ocasiões, é um array que:
    - recebe 1000 amostras
    - cada amostra é a subtração de 100 de um valor amplificado por um fator de 3
    - cada valor é a soma de uma amostra de pageSpeeds com uma amostra retirada aleatóriamente de uma distribuição normal com média 0 e desvio padrão 0.1
essa definição de purchaseAmount garante uma relação linear com pageSpeeds junto a um componente de aleatoriedade que se espera em uma situação real.
"""
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3

#plt.scatter(pageSpeeds, purchaseAmount) #plota a relação entre purchaseAmount (variável dependente) e pageSpeeds (variável dependente)
#plt.show() #exibe o gráfico

"""
Agora, uma regressão linear é realizada na relação das amostras de purchaseAmount em função das amostras de pageSpeeds.
A função linregress é usada, e retorna:
    - slope: inclinação da reta final
    - intercept: intercepto-y da reta (valor de purchaseAmount quando pageSpeeds == 0)
    - r_value: coeficiente de determinação de Pearson da reta
    - p_value: valor-p
    - std_err: erro padrão
"""
slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)

#print(r_value**2) #imprime o Coeficiente de Determinação (R²), que neste caso é 0.9903750925028195

"""
Agora, utilizando-se dos dados da regressão, define-se um estimador:
"""
#função que retorna o y aproximado (purchaseAmount) dado um x (pageSpeed) com os parâmetros da reta da regressão
def predict(x):
    return slope * x + intercept

fitLine = predict(pageSpeeds) #vai aproximar um valor para purchaseAmounts para cada valor de pageSpeeds

"""
As 3 linhas abaixo plotam os pontos "reais" com os previstos pela reta da regressão linear
"""

"""
plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()
"""

"""
=============================ACTIVITY===============================
Try increasing the random variation in the test data, and see what effect it has on the r-squared error value.
"""

purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.5, 1000)) * 3 #aumentando em 5x o desvio padrão
slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)
#print(r_value**2) #caiu para 0.8125440983766514, uma redução de ~18%

purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 1.0, 1000)) * 3 #aumentando em 10x o desvio padrão
slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)
print(r_value ** 2) #caiu para 0.49293485265925246, uma redução de ~50%



