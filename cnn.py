import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import glob #gestion de fichier
import random
from scipy.misc import face

flou = np.array([[[1,1,1],[1,1,1],[1,1,1]]]).reshape(3,3,1)
###################
#Fonctions diverses
###################
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

def reduction(entree,masque):
    m,n,p = entree.shape
    r,s,t = masque.shape
    sortie = np.zeros((m-r+1,n-s+1,p-t+1))
    for i in range(0,m-r+1):
        for j in range(0,n-s+1):
            for k in range(0, p-t+1):
                sortie[i,j,k] = np.max(entree[i:i+r,j:j+s,k:k+t])
    return sortie

########
#Classes
########

class CNN():
    def __init__(self, structure, f, c = 0.1, cout= coutQuad):
        self.coefA = c
        self.foncAct = f
        self.filtres = [[np.random.random( (x[1][0],x[1][1],x[1][2]) ) for i in range(x[0])] for x in structure]
        self.pools = [ x[2] for x in structure] 
        self.cout = cout
                 
    #def calcul(self):
    
plt.subplot(2,1,1)
plt.imshow(face())
plt.subplot(2,1,2)
plt.imshow(conv3D(face(),flou))
plt.show()
        
        

