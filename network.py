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
        #print('OKAY')
        #print(len(test.layers))
 
    def cost(self,X,Y):
        self.cost = 0
        compute(self,X)
        for i in range(len(self.layers[-1].neurons)):
            self.cost += (self.layers[-1].neurons[i]-elu(Y[i]))**2

    def partialDerivative(self,Y):
        """Dérivée des neurones de chacune des couches par rapport aux neurones de la couche précédente"""
        self.listeDer = []
        for i in range(len(self.layers)-1):#Pour chaque couche (-1 car on utilise la couche suivante)
            prime = [dElu(invElu(self.layers[i+1].neurons[j])) for j in range(len(self.layers[i+1].neurons))]
            L = []
            for k in range(len(self.layers[i].neurons)):#Pour chaque neurone dans la couche actuelle
                L.append([])
                for j in range(len(self.layers[i+1].neurons)):#Pour chaque neurone dans la couche suivante
                    L[k].append(self.layers[i].coefs[j][k]*prime[j])#On calcule la dérivée de a^(i+1)_(k,j) par rapport à a^(i)_(k,j) (c.f. maths)
            self.listeDer.append(L)
        #self.listeDer.append([[2*(self.layers[-1].neurons[k]-Y[k])]for k in range(len(self.layers[-1].neurons))]) #dérivée de C par rapport à la dernière couche
        self.sommeDer = [[2*(self.layers[-1].neurons[k]-elu(Y[k]))for k in range(len(self.layers[-1].neurons))]]
        for i in range(0,len(self.layers)-1):
            self.sommeDer.append([sommeListe(produitListes(self.listeDer[-i-1][j],self.sommeDer[-1])) for j in range(len(self.listeDer[-i-1]))])
            #print('TYPE',type(self.listeDer[-i-1][j]),type(self.sommeDer[-1]))
    
    def grad(self,theor):
        ''' Entree: valeurs theoriques attendues
            Sortie: Le vecteur gradient correspondant, dirige vers l'augmentation la plus rapide.'''
        self.gradient = []
        self.gradientBias = []
        self.partialDerivative(theor)
        for i in range(len(self.layers)-1):
            self.gradient.append([])
            self.gradientBias.append([])
            for j in range(len(self.layers[i].neurons)):
                #for k in range(len(self.layers[i+1].neurons)):
                    #print('yay?',i,j,k)
                self.gradient[i].append([self.layers[i].neurons[j]*dElu(invElu(self.layers[i+1].neurons[k]))*self.sommeDer[-i-1][j] for k in range(len(self.layers[i+1].neurons))])
                    #print('yay',i,j,k)
                #print(len(self.sommeDer[-i-1]),len(self.layers[i+1].neurons))
                self.gradientBias[i].append(sommeListe([dElu(invElu(self.layers[i].neurons[j]))*self.sommeDer[-i-1][l] for l in range(len(self.sommeDer[-i-1]))]))

    def modifWeights(self):
        for k in range(len(self.layers)):
            for i in range(len(self.layers[k].coefs)):
                n = len(self.layers[k].coefs)
                for j in range(len(self.layers[k].coefs[i])):
                    self.layers[k].coefs[i][j] -= self.gradient[k][j][i]*0.05*self.layers[k].neurons[j]

    def train(self,nbr):
        n = len(self.layers[0].neurons)
        p = len(self.layers[-1].neurons)
        food = [1]*n
        for k in range(nbr):
            self.compute(food)
            self.grad([0.9]*p)
            self.modifWeights()
            print(self.layers[-1].neurons)
