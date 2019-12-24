import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import glob #gestion de fichier
import random
from scipy.misc import face,ascent

flou = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]).reshape(3,3,1)

os.chdir("TestDBP")
im = img.imread("Blob_0.png")
im = im.reshape(im.shape[0],im.shape[1],3)

###################
#Fonctions diverses
###################

def lectImg(nom):
    Img = img.imread(nom)
    if Img.shape[2] > 3:
        Img = Img[:,:,0:3]
    Img = np.reshape(Img,(-1,1))
    return Img

def normaliser(L):
    if L.shape == (1,1):
        sortie = np.array([1*(L[0,0] > 0.5)])
    else:
        sortie = np.array([1*(X == np.amax(L)) for X in L])
    sortie.shape = (-1,1)
    return sortie
    
def convolution(X,Y):
    return np.sum(X*Y)
    
def conv3D(entree,masque):
    m,n,p = entree.shape
    r,s,t = masque.shape
    sortie = np.zeros((m-r+1,n-s+1,p-t+1))
    for i in range(0,m-r+1):
        for j in range(0,n-s+1):
            for k in range(0, p-t+1):
                extract = entree[i:i+r,j:j+s,k:k+t]
                sortie[i,j,k] = convolution(extract,masque)
    return sortie

def reduction(entree,filtre,liste):
    m,n,p = entree.shape
    r,s,t = filtre
    sortie = np.zeros((m//r,n//s,p//t))
    liste.append([])
    for i in range(0,m-r,r):
        for j in range(0,n-s,s):
            for k in range(0, p-t,t):
                sortie[i//r,j//s,k//t] = np.max(entree[i:i+r,j:j+s,k:k+t])
                coord = np.where(entree[i:i+r,j:j+s,k:k+t] == sortie[i//r,j//s,k//t])
                liste[-1].append(list(zip(coord[0],coord[1])))
    return sortie

#######################
#Fonctions d'activation
#######################

class sigmoide():
    @staticmethod
    def fonction(x):
        """bijection de R dans ]0,1[, continue, derivable strictement croissante"""
        return 1/(1+np.exp(-x))

    @staticmethod
    def derivee(x):
        """derivee de la fonction sigmoide
        Entree : x = sig(y) ; Sortie : sig'(y) en fonction de x = sig(y)"""
        return x*(1-x)

class elu():
    @staticmethod
    @np.vectorize
    def fonction(x):
        """bijection de R dans ]-1,+inf[, continue, derivable strictement croissante"""
        return np.exp(x)-1 if x<0 else x

    @staticmethod
    @np.vectorize
    def derivee(x):
        """derivee de la fonction elu
        Entree : x = elu(y) ; Sortie : elu'(y) en fonction de x = elu(y)"""
        return min(x,0)+1

######
#Couts
######

class coutQuad():
    @staticmethod
    def fonction(x):
        return sigmoide.fonction(x) 
    @staticmethod
    def cout(x,y):
        return (1/2)*(x-y)**2
    @staticmethod
    def derivee(x,y):
        return x-y

class coutCroise():
    @staticmethod
    def fonction(x):
        return sigmoide.fonction(x)
    @staticmethod
    def cout(x,y):
        return y*np.log(x)+(1-y)*np.log(1-x)
    @staticmethod
    def derivee(x,y):
        return x-y

########
#Classes
########

class CNN():
    def __init__(self,structure,f,c = 0.1,cout = coutQuad):
        self.coefA = c
        self.foncAct = f
        self.filtres = [[np.random.random(x[1]) for i in range(x[0])] for x in structure]
        self.pools = [x[2] for x in structure]
        self.cout = cout
        self.max = [[] for f in self.filtres]
                 
    def calcul(self,entree):
        self.neurones = [self.foncAct.fonction(entree)]
        for listef,p,i in zip(self.filtres,self.pools,range(0,len(self.filtres))):
            self.neurones.append(self.foncAct.fonction(reduction(conv3D(self.neurones[-1],listef[0]),p,self.max[i])))
            for f in listef[1:]:
                self.neurones[-1] = np.append(self.neurones[-1],self.foncAct.fonction(reduction(conv3D(self.neurones[-2],f),p,self.max[i])),axis = 2)

    def calcErr(self,erreurSortie):
        erreur = [erreurSortie]
        

        
test = CNN([(3,(3,3,1),(2,2,1))],sigmoide)
test.calcul(im)
