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
    sum = 0
    for i in range(len(L1)):
        sum += L1[i]*L2[i]
    return sum

def scalaireListe(x,L):
    """multiplication de tous les termes d'une liste par un scalaire"""
    for i in range(len(L)):
        L[i] = x*L[i]
        
def sommeListe(L):
    somme = 0
    for x in L:
        somme += x
    return x

class Layer():
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
        self.gradient
        L.append(0) #Couche finale sans poids
        for i in range(len(L)-1): #Ne pas deborder dans la ligne suivante
            self.layers.append(Layer(L[i],L[i+1])) #création de chaque couche du réseau
 
    def compute(self,X): #calcule la sortie en fonction de l'entrée
        for j in range(len(X)):
            self.layers[0].neurons[j] = elu(X[j])
            forward(0)
 
    def forward(self,i): #transfert des données des neurones d'une couche i vers la couche suivante
        if i > len(self.layers): #arrête la récursivité
            return
        for k in range(len(self.layers[i+1].neurons)):
            self.layers[i+1].neurons[k] = elu(produitListes(self.layers[i].neurons,self.layers[i].coefs[k])+self.layers[i+1].biases[k])
        forward(i+1) #récursivité pour transférer les données de la première couche à la dernière
 
    def cost(self,X,Y):
        self.cost = 0
        compute(self,X)
        for i in range(len(self.layers[-1].neurons)):
            self.cost += (self.layers[-1].neurons[i]-elu(Y[i]))**2

    def partialDerivative(self):
        """Dérivée des neurones de chacune des couches par rapport aux neurones de la couche précédente"""
        self.listeDer = []
        for i in range(len(self.layers)):
            L = []
            for j in range(len(self.layers[i+1].neurons)):
                L.append([])
                prime = dElu(invElu(self.layers[i+1].neurons[j]))
                for k in range(len(self.layers[i].neurons)):
                    L[j].append(self.layers[i].coefs[j][k]*prime)
            self.listeDer.append(L)
        self.sommeDer = [[sommeListe(self.listeDer[0][i] for i in range(len(self.layers[-2].neurons))]
        for i in range(len(self.listeDer)):
            sommeDer.append(produitListe(sommeDer[-1],sommeListe(self.listeDer[0][i] for i in range(len(self.layers[-2].neurons)
                                     
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                derPar = produit([sommeListe(self.listeDer[i] #A finir/retravailler
     
    def deriveeRec(self,i):
         ''' renvoie '''
         return "oui"
                            
    def grad(self,exper,theor):
        self.gradient = []
        for i in range(len(self.layers)):
            partDer = partialDerivative(self,i)
            for j in range(len(self.layers[i])):
                self.gradient.append(sommeListe(self.layers[i].neurons) * dElu(invElu(self.layers[i].neurons[j])) * partDer[i][j])
        return 'oui'
         
