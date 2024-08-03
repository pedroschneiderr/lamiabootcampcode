"""
Neste script, são feitas regressões múltiplas num conjunto de dados real sobre atributos de carros
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import statsmodels.api as sm #API com classes de modelos matemáricos, com Ordinary Least Squares, que é usado
from sklearn.preprocessing import StandardScaler #classe que ajusta e aplica a dados uma escala de normalzação

#importa os dados
df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls') #converte o documento excel da URL para DataFrame

"""
Agora, os dados são extraídos da estrutura importada, agrupados por milhagem e a média de cada grupo é calculada.
"""
df1 = df[['Mileage', 'Price']].copy(deep=True) #extrai as colunas Mileage e Price de df para df1
bins = np.arange(0, 50000, 10000) #array com os limites para categorização dos dados da coluna Mileage de df1

categorized_milages = pd.cut(df1['Mileage'], bins) #retorna uma Series com as categorias dos dados da coluna Mileage
groups = df1.groupby(categorized_milages, observed=False).mean() #agrupa as linhas de df1 pela categoria da milhagem e agrega pela média
""" 
groups['Price'].plot.line() #plota os preços de acordo com seus indexes (que são as categorias de milhagem, graças aos groupby)
plt.show()
 """

"""
Agora, um modelo de Regressão Múltipla para a previsão de preços de carros a partir de número de cilindros, portas e milhagem utilizando-se a API statsmodel
"""
X = df[['Mileage', 'Cylinder', 'Doors']].copy(deep=True) #variáveis independentes
y = df['Price'].copy(deep=True) #variáveis dependentes

scale = StandardScaler() #cria uma nova instância de escala
X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].values) #cria uma escala para normalização dos atributos (fit) e a aplica nos conjuntos de dados (transform)
X = sm.add_constant(X) #adiciona-se uma coluna de constantes (para o intercepto y) no DataFrame X

est = sm.OLS(y, X).fit() #treina o modelo de Regressão Múltipla

#print(est.summary()) #gera e imprime um resumo com as principais informações sobre o estimador (modelo) final

"""
No resumo gerado em est.summary() temos que os coeficientes são:
const: 2.134e+04
Mileage: -1272.3412
Cylinder: 5587.4472
Doors: -1404.5513

Ou seja, o número de cilindros influencia no preço consideravelmente mais do que os outros atributos.
Uma forma mais simples de ver que, por exemplo, mais portas não implica em maior preço, seria observando as médias:
"""

#print(y.groupby(df['Doors'], observed=False).mean()) #agrupa os preços pelo número de portas e agrega pela média
"""
O resultado mostra que a média de preço de carros com 4 portas é um pouco menor que a média dos de 2 portas.
Isso é um exemplo de como a solução pode ser mais simples do que parece.
"""

"""
Usando o estimador gerado pela Regressão Múltipla para prever o preço de um carro fictício
"""
scaled = scale.transform([[45000, 8, 4]]) #normaliza os dados de um carro com 45000 milhas corridas, 8 cilindros e 4 portas com a escala já criada antes
scaled = np.insert(scaled[0], 0, 1) #insere uma coluna com a linha igual a 1 na posição 0, que serão os interceptos y
predicted = est.predict(scaled) 
#print(predicted) #preço previsto == 27658.15707316

"""
Activity¶
Mess around with the fake input data, and see if you can create a measurable influence of number of doors on price. Have some fun with it - why stop at 4 doors?
"""

df2 = df[['Price', 'Doors', 'Cylinder', 'Mileage']].copy(deep=True) #pega as dimensões pertinentes
"""
aumenta os preços dos carros de 4 portas:
"""
for i, val in df2['Doors'].items():
    if val == 4:
        df2.loc[i, 'Price'] = df2.loc[i, 'Price'] * (1 + (np.random.rand() * (0.5 - 0.1) + 0.1)) #aumenta o preço dos carros de 4 portas entre 10% e 50%

"""
cria 200 carros de 6 portas mais caros que os de 2 e de 4:
"""
biggerMean = df2.groupby('Doors', observed=False).mean()['Price'].max() #pega a maior média de preço entre os grupos de 2 e 4 portas
mileageMean = df2['Mileage'].mean() #calcula a milhagem média de todo o dataset

for i in range(200):
    if np.random.randint(4) > 0: #80% de chance pro primerio bloco, 20% pro segundo
        price = biggerMean * (1 + (np.random.rand() * (0.3 - 0.1) + 0.1)) #preço é de 10% a 30% maior que biggerMean
    else:
        price = biggerMean * (np.random.rand() * (1 - 0.7) + 0.7) #preço é de 0% a 30% menor que biggerMean
    new_car = {'Price': price, 'Doors': 6, 'Cylinder': np.random.choice([4, 6, 8]), 'Mileage': mileageMean} #cria uma representação de um carro com: 6 portas; preço == price; 4, 6 ou 8 cilindros; milhagem média
    df2.loc[len(df2)]= new_car #insere o carro ao fim do dataframe df2

"""
Normaliza os dados de df2:
"""
scale2 = StandardScaler()
X = df2[['Doors', 'Cylinder', 'Mileage']].copy(deep=True)
y = df2['Price'].copy(deep=True)
X[['Doors', 'Cylinder', 'Mileage']] = scale2.fit_transform(X[['Doors', 'Cylinder', 'Mileage']].values)
X = sm.add_constant(X)

est2 = sm.OLS(y, X).fit()

#print(est2.summary())
"""
const: 2.669e+04    Aumento de ~25% em relação ao dataset anterior, sinal mantido
Doors: 1080.1302    Diminuição de ~23% no módulo, mas agora o sinal é positivo
Cylinder: 5229.0477 Diminuição de ~7%, sinal mantido
Mileage: -1438.1461 Aumento de ~2% no módulo, sinal mantido

Ou seja, a influência do número de portas no preço aumentou enquanto a das outras variáveis foi pouco alterada.
"""







