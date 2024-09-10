"""
Neste script, algumas funções de ativação são implementadas.
"""

import numpy as np

#função degrau
def degrau(x):
    if x >= 0:
        return 1
    return 0

#sigmoide
def sigmoide(x):
    return 1/(1 + np.exp(-x))

#hiperbólica tangente
def hiperb_tan(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

#ReLU
def reLU(x):
    if x >= 0:
        return x
    return 0

#softmax
def softmax(x): #recebe uma lista x
    ex = np.exp(x) #cria uma lista com os valores exponenciais
    return ex/ex.sum() #divide cada elemento da lista pela soma deles

#linear
def linear(x):
    return x

functions = [sigmoide, hiperb_tan, reLU, linear]
for i in functions:
    print(i(2.1))