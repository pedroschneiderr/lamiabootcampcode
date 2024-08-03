"""
Dados fictícios de velocidade de carregamento de páginas online e de compras feitas nessas páginas são criados
Esses dados possuem uma relação não linear e regressões polinomias são feitas a partir deles
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score #função que calcula o R² de um modelo polinomial

"""
Primeiramente, os dados fictícios são criados:
pageSpeeds, que contém os dados de carregamento das páginas, é um array que recebe 1000 amostras retiradas aleatoriamente de uma distribuição normal com média 3 e desvio padrão 1.
purchaseAmount, que contém os dados de quantidade de compras feitas na página em diferentes ocasiões, é um array que:
    - recebe 1000 amostras
    - cada amostra é um valor divido por um valor de pageSpeeds
    - cada valor é uma amostra retirada aleatóriamente de uma distribuição normal com média 50 e desvio padrão 10
essa definição de purchaseAmount garante uma relação não linear, de proporção inversa, com pageSpeeds.
"""
np.random.seed(2) #estabelece uma seed específica na função random, que fará que os números pseudo-aleatórios gerados no script sejam os mesmos sempre
pageSpeeds = np.random.normal(3.0, 1.0, 1000) #gera um array com 1000 amostras retiradas de uma distribuição normal de média 3 e desvio padrão 1
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

""" 
plt.scatter(pageSpeeds, purchaseAmount) #cria a plotagem da relação dos dados de pageSpeeds (eixo x, variável independente) com os de purchaseAmount (eixo y, variável dependente)
plt.show() #plota
"""

"""
Agora, um polinômio de grau 4 é ajustado aos dados:
"""
p4 = np.polynomial.polynomial.Polynomial.fit(pageSpeeds, purchaseAmount, 4) #método retorna um polinomio de grau 4 que modela a relação entre pageSpeeds e purchaseAmount a partir de uma regressão

#as linhas abaixo plotam: pageSpeeds x purchaseAmount junto com as previsões para purchaseAmount por p4 a partir de pageSpeeds
"""
x = np.array(sorted(pageSpeeds)) #cria uma versão em ordem crescente de pageSpeeds para plotagem
plt.scatter(pageSpeeds, purchaseAmount) #cria a plotagem da relação pageSpeeds e purchaseAmount
plt.plot(x, p4(x), c='r') #adiciona a plotagem de x (eixo x) e os valores previstos para purchaseAmount (eixo y) pelo polinômio p4
plt.show()
"""

r2 = r2_score(purchaseAmount, p4(pageSpeeds)) #calcula o R² para p4

#print(r2) #imprime o R², que neste caso é 0.8293766396303073

"""
Activity
Try different polynomial orders. Can you get a better fit with higher orders? Do you start to see overfitting, even though the r-squared score looks good for this particular data set?
"""

"""
Testando um polinômio de grau 5
"""
""" 
#criando o polinômio
p5 = np.polynomial.polynomial.Polynomial.fit(pageSpeeds, purchaseAmount, 5)

#plotando os dados originais
plt.scatter(pageSpeeds, purchaseAmount)

#plotando a curva gerada por pageSpeeds e os dados previstos por p5 para purchaseAmount
x = np.array(sorted(pageSpeeds)) #cria uma versão em ordem crescente de pageSpeeds para plotagem
plt.plot(x, p5(x), c = 'r')

#mostrando o plot final
plt.show()

#caculando o R² para p5
r2 = r2_score(purchaseAmount, p5(pageSpeeds))

print(r2) #imprime R² para p5, que neste caso é 0.8553884386186104, ~3% maior que o de p4
"""

"""
Testando um polinômio de grau 6
"""
""" 
#criando o polinômio
p6 = np.polynomial.polynomial.Polynomial.fit(pageSpeeds, purchaseAmount, 6)

#plotando os dados originais
plt.scatter(pageSpeeds, purchaseAmount)

#plotando a curva de p6
x = np.array(sorted(pageSpeeds)) #cria uma versão em ordem crescente de pageSpeeds
plt.plot(x, p6(x), c = 'r')
plt.show()

#calculando R² para p6
r2 = r2_score(purchaseAmount, p6(pageSpeeds))

print(r2) #neste caso, é 0.8700733999871759, ~5% maior que o de p4
"""

"""
Testando um polinômio de grau 7
"""
""" 
p7 = np.polynomial.polynomial.Polynomial.fit(pageSpeeds, purchaseAmount, 7)

plt.scatter(pageSpeeds, purchaseAmount)
x = np.array(sorted(pageSpeeds))
plt.plot(x, p7(x), c = 'r')
plt.show()

r2 = r2_score(purchaseAmount, p7(pageSpeeds))
print(r2) #0.8778369781712094, ~6% maior que de p4
"""

"""
Testando com um polinômio de ordem 10
"""
""" 
p10 = np.polynomial.polynomial.Polynomial.fit(pageSpeeds, purchaseAmount, 10)

plt.scatter(pageSpeeds, purchaseAmount)
x = np.array(sorted(pageSpeeds))
plt.plot(x, p10(x), c = 'r')
plt.show()

r2 = r2_score(purchaseAmount, p10(pageSpeeds))
print(r2) #0.883217235625733, ~6.5% maior que o de p4
"""

"""
Testando um polinômio de grau 3
"""
"""
p3 = np.polynomial.polynomial.Polynomial.fit(pageSpeeds, purchaseAmount, 3)

plt.scatter(pageSpeeds, purchaseAmount)
x = np.array(sorted(pageSpeeds))
plt.plot(x, p3(x), c = 'r')
plt.show()

r2 = r2_score(purchaseAmount, p3(pageSpeeds))
print(r2) #0.7842354030823269, ~6% menor que o de p4
"""

"""
Conclusão:
Quando a ordem do polinômio cai pra 3, há uma grande queda do R²
Quando a ordem do polinômio sobe para 6, há um aumento considerativo de R²
Quando a ordem continua subindo, os aumentos passam a ser muito pequenos
Considerando custo computacional e perigo de overfitting, o ideal para o modelo parece ser o polinômio de grau 6
"""