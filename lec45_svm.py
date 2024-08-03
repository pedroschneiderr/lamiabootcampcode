"""
Nesse script, um modelo de Support Vector Machine (SVM) é utilizado para classificar pessoas em clusters em dados fictícios de renda e idade.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler # módulo de normalização de dados
import matplotlib.pyplot as plt
from sklearn import svm

"""
Criação dos dados fictícios:
"""

"""
A função createClusteredData:
    - divide N pontos em K clusters de tamanhos parecidos em uma distribuição de idade e renda
    - cria uma lista de arrays, onde cada um representa um ponto
    - cria uma lista contendo o cluster ao qual cada ponto pertence
    - converte ambas as listas para array e os retorna
"""
def createClusteredData(N, k):
    np.random.seed(1234) # para reproducibilidade
    pointsPerCluster = int(N/k) # determina o número de pontos que cada cluster terá
    X = [] # lista que irá conter os pontos em array
    y = [] # lista que irá conter o cluster ao qual cada ponto pertence
    for i in range(k): # uma iteração pra cada cluster
        incomeCentroid = np.random.uniform(20000.0, 200000.0) # escolhe a coordenada de renda para o centroide do clusters
        ageCentroid = np.random.uniform(20.0, 70.0) # escolhe a coordenada de idade para o centroide do cluster
        for j in range(pointsPerCluster): # uma iteração para cada ponto/amostra
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)]) # cria um ponto aleatório em volta do centroide, com desvio padrão de 10000 em uma coordenada e 2.0 em outro
            y.append(i) # registra a qual cluster o ponto criado pertence
    X = np.array(X)
    y = np.array(y)
    return X, y

(X, y) = createClusteredData(100, 5) # cria um conjunto de dados com 100 pessoas distribuídas em 5 clusters

"""
Normalizando e visualizando os dados
"""
#plotagem dos dados originais:
""" plt.scatter(x=X[:, 0], y=X[:, 1], c=y.astype(float)) # plota um gráfico de dispersão com os dados de X colorindo cada ponto de acordo com a classe correspondente em y
plt.show() """

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X) # cria uma escala de normalização para os dados de X entre -1 e 1
X = scaling.transform(X) # aplica a escala de normalização aos dados

#plotagem dos dados normalizados:
""" plt.scatter(x=X[:, 0], y=X[:, 1], c=y.astype(float))
plt.show() """


"""
Treinamento do modelo:
"""
C = 1.0 # hiperparâmetro de regularização do modelo
svc = svm.SVC(kernel='linear', C=C).fit(X, y) # cria um modelo SVM com kernel linear e o treina com os dados de X e y

"""
Visualizando as divisões feitas pelos SVCs:
"""

def plotPredictions(clf):
    # retorna arrays com as coordenadas de um grid com todos os pontos possíveis dados os arrays de coordenadas passados de argumento:
    xx, yy = np.meshgrid(np.arange(-1, 1, 0.001), np.arange(-1, 1, 0.001))

    # converte os arrays 2-D em 1-D:
    npx = xx.ravel()
    npy = yy.ravel()

    # cria um array de pontos com as coordenadas de mesmo index de npx e npy
    samplePoints = np.c_[npx, npy]

    # gera as predições (classificações) para cada ponto no grid
    Z = clf.predict(samplePoints)

    # plota as regiões dos clusters com diferentes cores:
    Z = Z.reshape(xx.shape) # coloca o array de classificações no mesmo formato dos de coordenada
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8) # desenha as regiões
    plt.scatter(X[:, 0], X[:, 1], c=y.astype(float)) # plota os dados de treino
    plt.show()

#plotPredictions(svc)

""" Activity
"Linear" is one of many kernels scikit-learn supports on SVC. Look up the documentation for scikit-learn online to find out what the other possible kernel options are. Do any of them work well for this data set? """

svc_poly = svm.SVC(kernel='poly', C=C).fit(X, y) # cria um modelo com kernel polinomial
#plotPredictions(svc_poly)
"""
Conclusão: não parece ter funcionado bem.
"""

svc_radial = svm.SVC(kernel='rbf', C=C).fit(X, y) # cria um modelo com kernel radial
#plotPredictions(svc_radial)
"""
Conclusão: não parece ter funcionado bem para 1 dos clusters.
"""