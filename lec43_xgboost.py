"""
Neste script, um conjunto de árvores de decisão é criado, com o algoritmo de XGBoost, para classificar uma flor de Íris por espécie através do comprimento e espessura de suas sépalas e pétalas.
"""

import xgboost as xgb #biblioteca com classes e funções para utilizar o algoritmo XGBoost
from sklearn.model_selection import train_test_split #função para divisão de conjuntos de teste e treino
from sklearn.datasets import load_iris #função que carrega um dataset com amostras de flores, dimensionadas por seus valores de largura e comprimento de pétalas e sépalas
from sklearn.metrics import accuracy_score #função para medir a precisão de um classificador
import numpy as np

"""
Preparação dos dados:
"""
iris = load_iris() #carrega o dataset
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0) #X=iris.data; y=iris.target; 20% dos dados para teste; random_state fixo para reproducibilidade
train = xgb.DMatrix(X_train, label=y_train) #cria uma DMatrix com atributos iguais aos valores de X_train e classes dadas por y_train
test = xgb.DMatrix(X_test, label=y_test) #cria uma DMatrix com atributos iguais aos valores de X_train e classes dadas por y_train


"""
Definição dos hiperparâmetros:
"""
param = {
    'objective': 'multi:softmax', #múltiplas classes possíves, retorna apenas a mais provável
    'max_depth': 4, #profundidade máxima de cada árvore
    'eta': 0.3, #taxa de aprendizado (learning rate)
    'num_class': 3 #número classes (possíveis classificações)
}
epochs = 10 #número de árvores que serão criadas 

"""
Treinamento do modelo:
"""
model = xgb.train(param, train, epochs) #treina e retorna um modelo XGBoost

"""
Teste do modelo:
"""
predictions = model.predict(test) #classifica as amostras do conjunto de teste
#print(predictions)
print(accuracy_score(y_test, predictions)) #mede e imprime a precisão do modelo em relação ao conjunto de teste, o resultado é 1.0 (perfeito)

"""
ACTIVITY
See what it takes to make the results worse! How few epochs (iterations) can I get away with? How low can I set the max_depth? Basically try to optimize the simplicity and performance of the model, now that you already have perfect accuracy.
"""

#testando o efeito da diminuição do número de estimadores (árvores, epochs) no modelo:
""" for i in range(9, 0, -1): #countdown de 9 até 1, de 1 em 1
    model = xgb.train(param, train, i)
    accuracy = accuracy_score(y_test, model.predict(test))
    print(f'Precisão com {i} estimadores: {accuracy}.') """
"""
Conclusão: não houve mudança, a precisão foi de 1.0 para todas as quantidades de estimadores testados.
"""

#testando o efeito da diminuição da profundidade máxima da árvore:
""" for i in range(3, 0, -1): #countdown de 3 até 1, de 1 em 1
    param1 = {
        'objective':'multi:softmax',
        'max_depth':i,
        'eta': 0.3,
        'num_class': 3
    }
    model = xgb.train(param1, train, epochs)
    accuracy = accuracy_score(y_test, model.predict(test))
    print(f'Precisão com max_depth == {i}: {accuracy}') """
"""
Conclusão: nos 3 casos, a precisão caiu para 0.9666666666666667. Ou seja, max_depth == 4 é o ideal (mínimo para precisão perfeita) dados esses valores para os outros hiperparâmetros.
"""

#testando o efeita da taxa de aprendizado (eta) na precisão:
for i in np.arange(1, 11, 1)/10: #vai de 0.1 até 1, de 0.1 em 0.1
    param1 = {
        'objective':'multi:softmax',
        'eta':i,
        'max_depth':4,
        'num_class':3
    }
    model = xgb.train(param1, train, epochs)
    accuracy = accuracy_score(y_test, model.predict(test))
    print(f'Precisão com eta de {i}: {accuracy}')
"""
Conclusão: a precisão só caiu (para 0.9666666666666667) quando a taxa de aprendizado chegou no valor máximo de 1.0.
"""

