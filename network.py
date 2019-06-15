from random import random,randint
import numpy as np
  
def elu(x):
    """bijection de R dans ]-1,+inf[, continue dérivable strictement croissante"""
    if x <0:
       return np.exp(x)-1
    return x
  
def dElu(x):
    """dérivée de la fonction elu"""
    if x<0:
        return np.exp(x)
    return 1

def invElu(x):
    if x<0:
       return np.log(x+1)
    return x

def produitListes(L1,L2):
    """entrer deux listes de taille n pour obtenir une liste de taille n dont chaque terme est le produit des termes de meme rang des listes d'entrée"""
    L = []
    for i in range(len(L1)):
        L.append(L1[i]*L2[i])
    return L

def scalaireListe(x,L):
    """multiplication de tous les termes d'une liste par un scalaire"""
    for i in range(len(L)):
        L[i] = x*L[i]
        
def sommeListe(L):
    somme = 0
    for x in L:
        somme += x
    return somme

def reverse(L):
    return L[::-1]

class layer():
    """entrer le nombre de neurones sur cette couche et sur la couche précédente"""
    def __init__(self,n,p):
        self.neurons = [random() for i in range(n)]
        self.coefs = [[random() for i in range(p)] for j in range(n)]
        self.biases = [random() for i in range(n)]

class neuralNetwork():
    """entrer une liste comportant autant de termes qu'il y a de couches dans le réseau, et dont chaque terme correspond au nombre de neurones sur la couche associée à ce terme"""
    def __init__(self,L):
        self.layers = [layer(L[0],0)]#liste contenant les couches (des objets)
        for i in range(1,len(L)): #Ne pas deborder dans la ligne suivante
            self.layers.append(layer(L[i],L[i-1])) #création de chaque couche du réseau
 
    def compute(self,X): #calcule la sortie en fonction de l'entrée
        for j in range(len(X)):
            self.layers[0].neurons[j] = elu(X[j])
        self.forward(0,len(self.layers))
 
    def forward(self,i,n): #transfert des données des neurones d'une couche i vers la couche suivante
        if i >= n-1: #arrête la récursivité
            return
        for k in range(len(self.layers[i+1].neurons)):
            valeur =  elu(sommeListe(produitListes(self.layers[i].neurons,self.layers[i+1].coefs[k]))+self.layers[i+1].biases[k])
            self.layers[i+1].neurons[k] = valeur
        self.forward(i+1,n)#récursivité pour transférer les données de la première couche à la dernière
 
    def cost(self,X,Y):
        self.cost = 0
        compute(self,X)
        for i in range(len(self.layers[-1].neurons)):
            self.cost += (self.layers[-1].neurons[i]-elu(Y[i]))**2

    def grad(self,theor):
        Y = self.layers[-1].neurons #Couche de sortie/reponse
        self.e = [[dElu(invElu(self.layers[-1].neurons[i]))*(Y[i] - theor[i]) for i in range(len(Y))]]#La liste accueillant les erreurs e_i^k <-> erreur entre experimental et desiree pour le neurone i de la couche k
        for k in range(len(self.layers)-1): #Pour chaque couche du reseau moins la derniere (deja effectue)
            self.e.append([dElu(invElu(self.layers[-1-k-1].neurons[i]))*sommeListe(produitListes([X[i] for X in self.layers[-1-k].coefs],self.e[-1])) for i in range(len(self.layers[-1-k-1].neurons))])
        self.e = reverse(self.e)
                
    def modifWeights(self):
        for k in range(1,len(self.layers)):
            for i in range(len(self.layers[k].coefs)):
                for j in range(len(self.layers[k].coefs[i])):
                    self.layers[k].coefs[i][j] -= self.e[k][i]*0.01*self.layers[k-1].neurons[j]

    def modifBiases(self):
        for i in range(1,len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                self.layers[i].biases[j] -= self.gradientBias[i-1][j]*0.01*self.layers[i].neurons[j]

    def train(self,nbr,n):
        n = len(self.layers[0].neurons)
        p = len(self.layers[-1].neurons)
        result = [[random() for i in range(p)] for i in range(n)]
        food = [[random() for i in range(n)] for i in range(n)]
        for k in range(nbr):
            a = randint(0,n-1)
            self.compute(food[a])
            self.grad(result[a])
            self.modifWeights()
            #self.modifBiases()
            print(self.layers[-1].neurons)
        print(food,result)
