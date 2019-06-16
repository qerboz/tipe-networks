########
#Imports
########

from random import random,randint
import numpy as np
import matplotlib.image as img
import os
import glob #gestion de fichier


##########
#Fonctions
##########

###Générales
def enleverDoublons(liste):
    ''' Entrée: Une liste
        Sortie: La liste sans les doublons
        note: Préserve l'ordre
        '''
    res = []
    for x in liste:
        if x not in res:
            res.append(x)
    return res

###Réseau de neurones  
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

def train(reseau,base,nbr):
    n = len(data)
    for i in range(nbr):
        foodIndex = randint(0,n-1)
        reseau.compute( data[foodIndex][0] )
        reseau.grad(sorties[data[foodIndex][1] ])
        reseau.modifPoids()
        reseau.modifBiais()
        

###Traitement d'images
def pixelValue(pix):
    '''Entrée: Un tableau de trois valeurs représentant un pixel
        Sortie: Ces trois valeurs concaténées.
    '''
    return ('00'+str(pix[0]))[-3::] + ('00'+str(pix[1]))[-3::] + ('00'+str(pix[2]))[-3::]

def imgEnNombres(image):
    '''Entrée: Une image de taille nxp sous la forme d'un array numpy contenant les valeurs RGB de chaque pixel
        Sortie: Une liste de même taille que le nombre de pixels de l'image contenant des valeurs représentant les pixels''' 
    n= len(image)
    p= len(image[0])
    return [int(pixelValue(image[i//p][i%p])) for i in range(n*p)]

########
#Classes
########


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
        return self.layers[-1].neurons
 
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
        self.ecart = [[dElu(invElu(self.layers[-1].neurons[i]))*(Y[i] - theor[i]) for i in range(len(Y))]]#La liste accueillant ecarts du coût par rapport à chaque poids
        for k in range(len(self.layers)-1): #Pour chaque couche du reseau moins la derniere (deja effectue)
            self.ecart.append([dElu(invElu(self.layers[-1-k-1].neurons[i]))*sommeListe(produitListes([X[i] for X in self.layers[-1-k].coefs],self.ecart[-1])) for i in range(len(self.layers[-1-k-1].neurons))])
        self.ecart = reverse(self.ecart)
                
    def modifPoids(self):
        for k in range(1,len(self.layers)):
            for i in range(len(self.layers[k].coefs)):
                for j in range(len(self.layers[k].coefs[i])):
                    self.layers[k].coefs[i][j] -= self.ecart[k][i]*0.05*self.layers[k-1].neurons[j]

    def modifBiais(self):
        for i in range(1,len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                self.layers[i].biases[j] -= self.ecart[i][j]*0.05

    def train(self,nbr,v):
        n = len(self.layers[0].neurons)
        p = len(self.layers[-1].neurons)
        result = [[random() for i in range(p)] for i in range(v)]
        food = [[random() for i in range(n)] for i in range(v)]
        for k in range(nbr):
            a = randint(0,v-1)
            self.compute(food[a])
            self.grad(result[a])
            self.modifPoids()
            self.modifBiais()
        for i in range(v):
            print(food[i] , 'doit renvoyer' , result[i])

#####
#Main
#####
reconnaisseur = neuralNetwork([25,20,17,10,6])

###obtention des donnees d'entrainement
os.chdir(".\TrainDB")
trainData_name = glob.glob("*.png")
especes = enleverDoublons([name.split("_")[0] for name in trainData_name])

trainData = np.array([ np.floor(img.imread(trainData_name[i]) * 255).astype(np.uint8) for i in range(len(trainData_name))])
trainData = [( imgEnNombres(trainData[i]), trainData_name[i].split("_")[0] ) for i in range(len(trainData))]


###obtention des donnees de test
os.chdir("..\TestDB")
testData_name = glob.glob("*.png")
testData = np.array([ np.floor(img.imread(trainData_name[i]) * 255).astype(np.uint8) for i in range(len(trainData_name))])
testData = [( imgEnNombres(trainData[i]), trainData_name[i].split("_")[0] ) for i in range(len(trainData))]

###Sorties théoriques à partir du nom de l'espèce
sorties = [ ( especes[i] , [ 1*(i==j) for j in range(len(especes))] ) for i in range(len(especes))]
sorties = dict(sorties)








