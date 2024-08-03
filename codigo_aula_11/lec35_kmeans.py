"""
Neste script, dados fictícios clusterizados de uma relação entre renda e idade são criados.
Depois, usa-se o método de K-Means Clustering para tentar recuperar os clusters desses dados, conhecidos do estágio de criação dos dados fictícios.
"""
import numpy as np
from sklearn.preprocessing import scale #função de normalização por variância unitária
from sklearn.cluster import KMeans #classe de um modelo de K-Means Clustering
import matplotlib.pyplot as plt

"""
A função createClusteredData:
    - recebe dois argumentos: número total de pontos (N) e número de clusters (k)
    - cria ~(N/k) pontos por cluster, estabelendo um centroide aleatório para cada um e determinando os pontos em uma distribuição normal ao redor desse centroide
    - retorna um array de listas, onde cada cada lista representa um ponto, com 2 floats representando as coordenadas x (renda) e y (idade)
"""
def createClusteredData(N, k):
    np.random.seed(10) #seed para reproducibilidade dos dados
    pointsPerCluster = int(N/k) #determina o número aproximado de pontos por cluster
    X = [] #lista que irá conter os pontos
    for i in range(k): #uma iteração para cada cluster
        incomeCentroid = np.random.uniform(20000.0, 200000.0) #escolhe uma coordenada para o centroide
        ageCentroid = np.random.uniform(20.0, 70.0) #escolhe a outra coordenada do centroide
        for j in range(pointsPerCluster): #uma iteração para cada ponto do cluster
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)]) #cria um ponto aleatório, com desvio padrão de 10000.0 na abscissa e 2.0 na ordenada em relação ao centroide
    X = np.array(X)
    return X

"""
Agora, cria-se e normaliza-se os dados fictícios:
"""
data = createClusteredData(100, 5) #cria um conjunto de dados fictício com 100 amostras e 5 clusters
normData = scale(data) #normaliza os dados, algo necessário dada a grande diferença entre as escalas de idade (casa das dezenas) e renda (casa das dezenas de milhares)

"""
Em seguida, o modelo é treinado e usado para classificar cada ponto por cluster:
"""
model = KMeans(n_clusters=5, n_init='auto') #cria um modelo de K-Means Clustering com 5 clusters
model = model.fit(normData) #treina o modelo

"""
Agora, é feita a plotagem dos dados com a coloração de cada ponto dada pela classificação de cluster:
"""
""" #print(model.labels_) #imprime a classificação de cada ponto
plt.scatter(x=data[:, 0], y=data[:, 1], c=model.labels_.astype(float)) #nas abscissas vão as coordenadas de income, nas ordenadas vão as de age, e a cor de cada ponto é dado pelo float que representa o respectivo cluster
plt.show() """

"""
Activity
Things to play with: what happens if you don't scale the data? What happens if you choose different values of K? In the real world, you won't know the "right" value of K to start with - you'll need to converge on it yourself.
"""

"""
O que acontece se os dados não forem normalizados?
"""

model2 = KMeans(n_clusters=5, n_init='auto')
model2 = model2.fit(data) #treina o modelo com os dados não normalizados
plt.scatter(x=data[:, 0], y=data[:,1], c=model2.labels_.astype(float))
plt.show()
"""
Conclusão: a diferença mais gritante foi a de um cluster que englobou amostras com uma variação muito grande de idade.
Isso faz sentido, dado que, sem a normalização, diferenças de idade se tornam insignificantes perto de diferenças de renda.
"""

"""
O que acontece se diferentes valores de k forem escolhidos?
"""

"""
O loop abaixo aplica KMeans com valores de k variando de 1 a 7 ao mesmo conjunto de dados e plota cada um
"""
""" for i in range(1, 8):
    model = KMeans(n_clusters=i, n_init='auto')
    model = model.fit(normData)
    plt.scatter(x=data[:, 0], y=data[:,1], c=model.labels_.astype(float))
    plt.show() """
"""
Conclusão: visualmente, 1 a 3 parece insuficiente, 4 parece o ideal, 5 também faz sentido, 6 e 7 parece demais
"""




