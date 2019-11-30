########
#Imports
########

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import glob #gestion de fichier
import random

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

###################
#Reseau de neurones
###################

class reseauNeuronal():
    def __init__(self,structure,f = elu,c = 0.1,cout = coutQuad):
        self.reglageCoefA(c) #coef d'apprentissage par defaut
        self.foncAct = f
        self.poids = [np.random.random((n,p))*2-1 for n,p in zip(structure[1:],structure[:-1])]
        self.biais = [np.zeros((n,1)) for n in structure[1:]]
        #self.poids = [np.array([[[1]*p]*n]).reshape(n,p) for n,p in zip(structure[1:],structure[:-1])]
        self.cout = cout

    def reglageCoefA(self,x):
        self.coefA = x
                
    def calcul(self,entree):
        self.neurones = [self.foncAct.fonction(entree)] #insertion des valeurs d'entree dans les premiers neurones
        for p,b in zip(self.poids[:-1],self.biais[:-1]):
            self.neurones.append(self.foncAct.fonction(np.dot(p,self.neurones[-1])+b))
        self.neurones.append(self.cout.fonction(np.dot(self.poids[-1],self.neurones[-1])+self.biais[-1]))
        
    def calcErr(self,valTheor):
        valExp = self.neurones[-1] #recuperation des valeurs de sortie calculees
        erreur = [self.foncAct.derivee(valExp)*self.cout.derivee(valExp,valTheor)]
        for p,n in zip(self.poids[::-1],self.neurones[:-1][::-1]):
            erreur.append(self.foncAct.derivee(n)*np.dot(p.transpose(),erreur[-1])) #e^(n-1) = f'(x^(n-1)).((P^T)*e^(n))
        erreur = erreur[::-1]
        erreurPoids=[]
        erreurBiais=[]
        for i in range(len(self.poids)):
            moy_p = np.zeros(self.poids[i].shape)
            moy_b = np.zeros(self.biais[i].shape)
            for j in range(erreur[i].shape[1]):
                moy_p += np.dot(erreur[i+1][:,j].reshape(-1,1),self.neurones[i][:,j].reshape(-1,1).transpose())
                moy_b += erreur[i+1][:,j].reshape(-1,1)
            erreurPoids.append(moy_p/valTheor.shape[1])
            erreurBiais.append(moy_b/valTheor.shape[1])
        return erreurPoids,erreurBiais
        
    def modifPoids(self,var):
        for i in range(len(var)):
            self.poids[i] = self.poids[i] - self.coefA*var[i]
                        
    def modifBiais(self,var):
        for i in range(len(var)):
            self.biais[i] = self.biais[i] - self.coefA*var[i]
            
    def entrainer(self,baseE,nbrEntrainements,tailleBatch = 1):
        for i in range(nbrEntrainements):
            random.shuffle(baseE)
            paquetsEntrainement = [baseE[k:k+tailleBatch] for k in range(0, len(baseE), tailleBatch)]
            for paquet in paquetsEntrainement:
                entree = np.concatenate(list(zip(*paquet))[0],1)
                sortie = np.concatenate(list(zip(*paquet))[1],1)
                self.calcul(entree)
                erreurPoids,erreurBiais = self.calcErr(sortie)
                self.modifPoids(erreurPoids)
                self.modifBiais(erreurBiais)

    def perf(self,baseT):
        score = 0
        for i in baseT:
            self.calcul(i[0])
            if np.array_equal(i[1],normaliser(self.neurones[-1])):
                score += 1
        return score/len(baseT)
                
##xor = reseauNeuronal([2,2,1],sigmoide,0.1)
##L = [[np.array([[1],[0]]),np.array([[1]])],[np.array([[0],[0]]),np.array([[0]])],[np.array([[0],[1]]),np.array([[1]])],[np.array([[1],[1]]),np.array([[0]])]]
##xor.calcul(L[0][0])
##a,b = xor.calcErr(L[0][1])
##xor.modifPoids(a)

#####
#Main
#####

##Obtention des especes
os.chdir("./TRAIN")
donneesEntrainementNoms = glob.glob("*.png")
especes = np.unique(np.array([name.split('_')[0] for name in donneesEntrainementNoms]))
sorties = [(especes[i],np.array([1*(i==j) for j in range(len(especes))])) for i in range(len(especes))]
for X in sorties:
    X[1].shape = (-1,1)
sorties = dict(sorties)

##Obtention des donnees d'entrainement
donneesEntrainement = [(lectImg(X),sorties[X.split('_')[0]]) for X in donneesEntrainementNoms]

##Obtention des donnees de test
os.chdir("../TEST")
donneesTestNoms = glob.glob("*.png")
donneesTest = [(lectImg(X),sorties[X.split('_')[0]]) for X in donneesTestNoms]

##Test
reconnaisseur = reseauNeuronal([54234,500,50,5],sigmoide,0.1)

n = 10
p = 5
T = [p*i for i in range(n+1)]
plt.clf()
for i in range(1,6):
    A = [0]
    random.shuffle(donneesEntrainement)
    random.shuffle(donneesTest)
    for a in range(n):
        reconnaisseur.entrainer(donneesEntrainement[:50*i],p,1)
        A.append(reconnaisseur.perf(donneesTest[:50*i]))
    plt.subplot(1,5,i)
    plt.plot(T,A)
plt.show()
