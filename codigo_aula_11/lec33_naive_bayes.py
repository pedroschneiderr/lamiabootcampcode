"""
Neste script, é criado e treinado um modelo classificador de e-mails (em spam e ham) utilizando-se do Método Naive Bayes.
"""

import os
import io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer #classe de tokenização de texto de contagem de ocorrência para criação de uma matriz esparsa
from sklearn.naive_bayes import MultinomialNB #classe de modelo classificador que é treinado por Naive Bayes
from sklearn.model_selection import train_test_split #função para partição de um DataFrame entre conjuntos de treino e de teste
from sklearn.metrics import f1_score

"""
A função readFiles é um gerador que entrega o caminho e o conteúdo de cada e-mail no diretório indicado por path.
"""
def readFiles(path):
    for root, dirnames, filenames in os.walk(path): #itera por todos os subdiretórios e arquivos em path
        for filename in filenames: #itera sobre todos os nomes de arquivos (e-mails)
            path = os.path.join(root, filename) #gera um caminho, acessível a esse script, para cada e-mail

            inBody = False #inicializa uma booleana que vai indicar se a leitura do e-mail já passou do header e chegou no corpo
            lines = [] #lista que irá conter as linhas do corpo do e-mail
            f = io.open(path, 'r', encoding='latin1') #abre o arquivo de e-mail no modo leitura e com codificação latin1
            for line in f: #itera por todas as linhas do e-mail
                if inBody: #caso a leitura esteja no corpo do e-mail
                    lines.append(line) #adiciona a linha do e-mail à lista
                elif line == '\n': #se a linha é uma quebra quer dizer que o header do e-mail acabou e o corpo vai começar a ser lido
                    inBody = True #a booleana indica que chegou no corpo do e-mail
            f.close() #fecha o arquivo
            message = '\n'.join(lines) #cria a mensagem final do e-mail restabelendo a estrutura original com a quebra de linha
            yield path, message #entrega o caminho e o conteúdo do e-mail em um tupla

"""
A função dataFrameFromDirectory cria e retorna um data frame com o conteúdo e classe de cada e-mail indexados pelo caminho do arquivo que contém o e-mail.
O argumento do parâmetro path é usado como argumento da função readFiles, que extrai o conteúdo de cada arquivo de e-mail em path.
O argumento classification contém a informação de se o e-mail é spam ou ham
"""
def dataFrameFromDirectory(path, classification):
    rows = [] #lista que receberá o conteúdo e a classificação de cada e-mail em forma de dicionário, para depois ser convertida em uma linha de DataFrame
    index = [] #lista que receberá o caminho para o arquivo de cada e-mail, que serão os índices do DataFrame
    for filePath, message in readFiles(path): #pega um caminho e o conteúdo de um e-mail de cada vez pela função geradora readFiles
        rows.append({'message': message, 'class': classification}) #adiciona um dicionário com o conteúdo e a classificação do e-mail ao fim da lista rows
        index.append(filePath) #adiciona o caminho do arquivo do e-mail ao fim da lista index
    
    return pd.DataFrame(rows, index=index) #retorna um DataFrame com dados (linhas) de rows e index da lista index

"""
Um DataFrame com conteúdo de cada e-mail e sua classifcação, indexados por seu caminho, é criado.
"""
data = pd.DataFrame({'message': [], 'class': []}) #cria um DataFrame vazio que irá conter os dados dos e-mails

data = pd.concat([data, dataFrameFromDirectory('emails/spam', 'spam')]) #concatena, por linhas, o DataFrame data com o retornado pela função, que conterá os dados dos e-mails de spam
data = pd.concat([data, dataFrameFromDirectory('emails/ham', 'ham')]) #concatena, por linhas, o DataFrame data com o retornado pela função, que irá conter os dados dos e-mails ham

"""
Agora, é necessário criar uma representação numérica do texto dos e-mails para que se possa treinar um modelo de ML com base neles.
Isso é feito por meio da tokenização do texto e da contagem dos tokens, que será a representação númerica.
"""
vectorizer = CountVectorizer() #cria uma instância da classe CountVectorizer
counts = vectorizer.fit_transform(data['message'].values) #tokeniza o conteúdos de texto em cada e-mail, gera uma matriz esparsa com uma linha para cada e-mail, uma coluna para cada token e os valores sendo a contagem dos tokens

"""
Agora, uma instância de um modelo classificador é criado e treinado pelo Método Naive Bayes.
"""
classifier = MultinomialNB() #cria uma instância da classe MultinomialNB, que é um classificador inicialmente não treinado
targets = data['class'].values #extrai as classes de cada e-mail para um array
classifier.fit(counts, targets) #treina o classificador, com cada documento (linha de count) correspondendo à sua classficação correta no array targets

"""
Com o modelo treinado, é possível classificar novos e-mails fictícios:
"""
examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
examples_counts = vectorizer.transform(examples) #utiliza o vocabulário de tokenição ajustado posteriormente
predictions = classifier.predict(examples_counts) #entra com a matriz esparsa examples_counts no classificador e retorna as predições em um array
#print(predictions) #imprime as predições, que são ['spam', 'ham']

"""
Activity
Our data set is small, so our spam classifier isn't actually very good. Try running some different test emails through it and see if you get the results you expect.
"""

"""
Testando o classificador com exemplos reais de e-mails recebidos por mim:
"""
examples2 = ['Fill your summer with more smiles, more focus, and better sleep with 40% \off an annual Headspace membership for a limited time. Sign up today and let the sunshine in.',
              "Your order's been processed. You're all set to start learning. Ready to jump in?"]
examples_counts2 = vectorizer.transform(examples2)
predictions = classifier.predict(examples_counts2)
#print(predictions) #imprime as classificações, que são ['ham', 'ham'], o que bate com a classificação feita pelo Outlook na minha caixa de entrada.

"""
Dividindo o DataFrame data em dados de treino e teste, treinando outro classificador e testando-o:
"""

"""
Primeramente, prepara-se os dados:
"""
X = data['message'].copy(deep=True) #copia os valores da variável independente (message)
y = data['class'].copy(deep=True) #copia os valores da variável dependente, que se quer prever (class)

"""
Agora, cria-se conjuntos de treino e teste com a função train_test_split:
    - test_size=0.2: proporção dos dados que serão usados para teste, neste caso é 20%
    - random_state=0: garante que a divisão pseudo-aleatória dos dados seja sempre a mesma em diferentes execuções do script, para reproducibilidade dos resultados
    - stratify=y: garante que a proporção de cada tipo de classe nos dados de treino e teste sejam os mesmos que no dataset original y
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

"""
Agora, são feitas a tokenização e a extração de atributo (criação da matriz esparse de contagem de token) dos conteúdos de texto em X.
"""
vectorizer2 = CountVectorizer() #cria uma instância da classe CountVectorizer
counts2 = vectorizer2.fit_transform(X_train.values) #faz a tokenização e a extração de atributo do dataset de treino

"""
Agora, os dados de teste são extraidos e o modelo classificador é treinado:
"""
classifier2 = MultinomialNB() #cria uma instância do modelo classificador
targets2 = y_train.values #extrai os dados treino da variável dependente 
classifier.fit(counts2, targets2) #treina o modelo

"""
Agora, o modelo é testado.
Como é um modelo de classificação desbalanceada (hams são muito mais comuns que spams), usa-se a métrica F-Score para avaliar a qualidade do estimador.
"""
test_counts = vectorizer2.transform(X_test.values) #aplica o vocabulário de tokenização, ajustado posteriormente, aos dados de teste
predictions = classifier.predict(test_counts) #faz as previsões
f1 = f1_score(y_test.values, predictions, average='binary', pos_label='spam') #calcula o f1-score para o classificador, tomando 'spam' como classe positiva
print(f1) #0.8636363636363636



