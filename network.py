from random import random
import numpy as np

def sigmoid(x):
    """bijection de R vers ]-1,1["""
    return 1/(1+np.exp(-x))

def elu(x):
   """bijection de R dans R, continue dérivable strictement croissante"""
    if x <0:
        return exp(x)-1
    return x
def d_elu(x):
    """dérivée de la fonction elu"""
    if x<0:
        return exp(x)
    return 1

def produitListes(L1,L2):
    """entrer deux listes de taille n pour obtenir une liste de taille n dont chaque terme est le produit des termes de meme rang des listes d'entrée"""
    sum = 0
    for i in range(len(L1)):
        sum += L1[i]*L2[i]
    return sum

class Layer(n,p):
    """entrer le nombre de neurones sur cette couche et sur la couche suivante"""
    def __init__(self,n):
        self.neurons = [random() for i in range(n)]
        self.coefs = [[random() for i in range(n)] for j in range(p)]
        self.biases = [random() for i in range(n)]

class Neural_Network(L):
    """entrer une liste comportant autant de termes qu'il y a de couches dans le réseau, et dont chaque terme correspond au nombre de neurones sur la couche associée à ce terme"""
    def __init__(self,L):
        self.layers = [] #liste contenant les couches (des objets)
        L.append(0) #Couche finale sans poids
        for i in range(len(L)-1): #Ne pas deborder dans la ligne suivante
            self.layers.append(Layer(L[i],L[i+1])) #création de chaque couche du réseau

    def compute(self,X): #calcule la sortie en fonction de l'entrée
        for j in range(len(X)):
            self.layers[0].neurons[j] = sigmoid(X[j])
        forward(0)

    def forward(self,i): #transfert des données des neurones d'une couche i vers la couche suivante
        if i > len(self.layers): #arrête la récursivité
            return
        for k in range(len(self.layers[i+1].neurons):
            self.layers[i+1].neurons[k] = sigmoid(produitListes(self.layers[i].neurons,self.layers[i].coefs[k])+self.layers[i+1].biases[k])
        forward(i+1) #récursivité pour transférer les données de la première couche à la dernière

    def cost(self,X,Y):
        self.cost = 0
        compute(self,X)
        for i in range(len(self.layers[-1].neurons)):
            self.cost += (self.layers[-1].neurons[i]-sigmoid(Y[i]))**2

    def grad(self,X,Y):
        cost(X,Y)
