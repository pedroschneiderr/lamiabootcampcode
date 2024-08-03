"""
Neste script, árvores de decisões são criadas para amostras cujas variáveis independentes são atributos de um candidato a um emprego e a dependente é a decisão final de contratar ou não.
É feito, também, o pré-processamento dos dados e Random Forests.
"""
import pandas as pd
from sklearn import tree #importa um módulo com modelos de árvore de decisão
from io import StringIO #classe que contrói objetos que se comportam como arquivos de texto, mas em memória
import pydotplus #biblioteca que permite ao Python interagir com o Graphviz para gerar e importar grafos
from IPython.display import Image #função e classe com funcionalidades para visualização de imagens a partir do Python
import os #biblioteca com funções para executar comandos de SO de dentro do script
from sklearn.ensemble import RandomForestClassifier


"""
Importação dos dados:
"""
input_file = 'PastHires.csv' #nome do arquivo csv com os dados sobre candidatos e contratações
df = pd.read_csv(input_file, header=0) #transforma o csv em um DF, assumindo a primeira linha como header (que contém o nome das colunas)
#print(df.head())

"""
Pré-processamento dos dados:
"""
#os modelos da scikit-learn precisam que todos os dados sejam numéricos, por isso, strings serão mapeadas para inteiros:
d = {'Y': 1, 'N': 0} #mapeia "sim" e "não" para 1 e 0
df['Employed?'] = df['Employed?'].map(d) #método map converte toda string em escalar dadas as relações no dicionário do argumento
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
df['Hired'] = df['Hired'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2} #mapeia os níveis de educação de 0 a 2
df['Level of Education'] = df['Level of Education'].map(d)

#agora, é necessária a separação dos atributos da classificação (coluna 'Hired'):
features = list(df.columns[:6]) #colunas de 0 a 5 são atributos
y = df['Hired'] #pega a coluna 'Header' como sendo a variável dependente
X = df[features] #pega todas as colunas cujos nomes estão na lista features como sendo as variáveis independentes

"""
Treinando o modelo:
"""
clf = tree.DecisionTreeClassifier() #cria uma instância de Árvore de Decisão
clf = clf.fit(X, y) #treina o modelo

"""
Visualizando a árvore:
"""
dot_data = StringIO() #inicializa uma instância de StringIO
tree.export_graphviz(clf, out_file=dot_data, feature_names=features) #cria um arquivo DOT (que pode ser lido e processado pelo Graphviz) a partir do modelo clf e o coloca em dot_data

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) #função que cria uma representação da árvore descrita em dot_data, com o Graphviz, e a retorna
#Image(graph.create_png()) #para Jupyper Notebook e ambientes IPython
#para salvar a imagem e abrir no linux:
graph.write_png('decision_tree.png')
os.system('xdg-open decision_tree.png')

"""
Aplicando o método de Random Forests e utilizando o modelo resultante para prever o resultado de dois candidatos.
"""

clf = RandomForestClassifier(n_estimators=10) #cria um modelo de Random Forest com 10 árvores
clf = clf.fit(X, y) #treina o modelo

#print(clf.predict([[10, 1, 4, 0, 0, 0], [10, 0, 4, 0, 0, 0]])) #imprime a classificação de dois candidatos fictícios. Resultado: [1 1]

"""
Activity
Modify the test data to create an alternate universe where everyone I hire everyone I normally wouldn't have, and vice versa. Compare the resulting decision tree to the one from the original data.
"""

"""
Modificação dos dados:
"""
transf = {0:1 , 1: 0} #dicionário de transformação de 0 em 1 e 1 e 0
df['Hired'] = df['Hired'].map(transf) #aplica à coluna de classficação 'Hired'
y = df['Hired']

"""
Treinando a nova árvore:
"""
clf2 = tree.DecisionTreeClassifier() #cria uma instância de Árvore de Decisão de classificação
clf2 = clf2.fit(X, y) #treina o modelo

"""
Gera a representação visual:
"""
dot_data2 = StringIO() #cria uma instância de StringIO
tree.export_graphviz(clf2, out_file=dot_data2, feature_names=features) #gera o arquivo DOT da árvore do modelo clf2
graph = pydotplus.graph_from_dot_data(dot_data2.getvalue()) #gera o objeto de grafo a partir do arquivo DOT
#Image(graph.create_png()) #gerando e imprimindo o png para ambientes IPython
#em linux:
graph.write_png('decision_tree2.png') #salva o png
os.system('xdg-open decision_tree2.png') #abre o png

"""
comparação:
Ao inverter as classificações: 
    - os atributos 'Interned' e 'Employed?' inverteram de papel
    - o atributo 'Year of experience' assumiu o lugar de 'Previous employers'
    - o atributo 'Top-tier school' assumiu o lugar de 'Level of education'
    - a entropia em cada passo permaneceu a mesma, assim como o tamanho da árvore. Algo esperado, já que a inverção das classificações não altera a entropia total.
"""