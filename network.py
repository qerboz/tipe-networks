from random import random
import numpy as np

def sigmoid(x):
    """bijection de R vers ]-1,1["""
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    """Dérivée de la fonction sigmoide"""
    return np.exp(-x)/(1+np.exp(-x))**2
  
def elu(x):
    """bijection de R dans R, continue dérivable strictement croissante"""
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
       return np.ln(x+1)
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

class layer():
    """entrer le nombre de neurones sur cette couche et sur la couche suivante"""
    def __init__(self,n,p):
        self.neurons = [random() for i in range(n)]
        self.coefs = [[random() for i in range(n)] for j in range(p)]
        self.biases = [random() for i in range(n)]

class neuralNetwork():
    """entrer une liste comportant autant de termes qu'il y a de couches dans le réseau, et dont chaque terme correspond au nombre de neurones sur la couche associée à ce terme"""
    def __init__(self,L):
        self.layers = []#liste contenant les couches (des objets)
        self.cost = []
        self.listeDer = []
        self.sommeDer = []
        self.gradient = []
        L.append(0) #Couche finale sans poids
        for i in range(len(L)-1): #Ne pas deborder dans la ligne suivante
            self.layers.append(layer(L[i],L[i+1])) #création de chaque couche du réseau
 
    def compute(self,X): #calcule la sortie en fonction de l'entrée
        for j in range(len(X)):
            self.layers[0].neurons[j] = elu(X[j])
            self.forward(0,len(self.layers))
 
    def forward(self,i,n): #transfert des données des neurones d'une couche i vers la couche suivante
        if i >= n-1: #arrête la récursivité
            return
        for k in range(len(self.layers[i+1].neurons)):
            valeur =  elu(sommeListe(produitListes(self.layers[i].neurons,self.layers[i].coefs[k]))+self.layers[i+1].biases[k])
            self.layers[i+1].neurons[k] = valeur
        self.forward(i+1,n)#récursivité pour transférer les données de la première couche à la dernière
        print('OKAY')
        print(len(test.layers))
 
    def cost(self,X,Y):
        self.cost = 0
        compute(self,X)
        for i in range(len(self.layers[-1].neurons)):
            self.cost += (self.layers[-1].neurons[i]-elu(Y[i]))**2

    def partialDerivative(self,Y):
        """Dérivée des neurones de chacune des couches par rapport aux neurones de la couche précédente"""
        self.listeDer = []
        for i in range(len(self.layers)-1):
            L = []
            for j in range(len(self.layers[i+1].neurons)):
                L.append([])
                prime = dElu(invElu(self.layers[i+1].neurons[j]))
                for k in range(len(self.layers[i].neurons)):
                    L[j].append(self.layers[i].coefs[j][k]*prime)
            self.listeDer.append(L)
        self.listeDer.append([2*(self.layers[-1].neurons[k]-Y[k]) for k in range(len(self.layers[-1].neurons))]) #dérivée de C par rapport à la dernière couche
        self.sommeDer = listeDer[-1:]
        for i in range(len(self.layers)-1):
            self.sommeDer.append([sommeListe(produitListes(listeDer[-i-1][j],sommeDer[-1])) for j in range(len(self.layers[-i]))])
                            
    def grad(self,theor):
        ''' Entree: valeurs theoriques attendues
            Sortie: Le vecteur gradient correspondant, dirigé vers l'augmentation la plus rapide.'''
        self.gradient = []
        self.gradientBias = []
        partialDerivative(theor)
        partDer = self.sommeDer
        for i in range(len(self.layers)-1):
            for k in range(len(self.layers[i+1])):
                for j in range(len(self.layers[i])):
                    self.gradient.append(self.layers[i].neurons[j]*dElu(invElu(self.layers[i+1].neurons[k])) * partDer[i+1][k])
                self.gradientBias.append(sommeListe([dElu(invElu(self.layers[i].neurons[j])) * partDer[i+1][k] for k in range(len(self.layers[i+1]))]))
