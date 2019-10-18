########
#Imports
########

import numpy as np
import random

#######################
#Fonctions d'activation
#######################

class sigmoide():
    @staticmethod
    @np.vectorize
    def fonction(x):
        """bijection de R dans ]0,1[, continue, derivable strictement croissante"""
        return 1/(1+np.exp(-x))

    @staticmethod
    @np.vectorize
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
    def __init__(self,structure,c = 0.1, f = elu, tailleBatch = 1, cout = coutQuad):
        self.reglageCoefA(c) #coef d'apprentissage par défaut
        self.foncAct = f
        self.poids = [np.random.randn(n,p)/np.sqrt(p) for n,p in zip(structure[1:],structure[:-1])]
        self.biais = [np.random.randn(n,1) for n in structure[1:]]
        self.t = tailleBatch
        self.cout = cout

    def reglageCoefA(self,x):
        self.coefA = x
    
    def calcul(self,entree):
        entree = np.asarray(entree) #entree sous forme d'array
        entree.shape = (-1,self.t) #entree sous forme de matrice
        self.neurones = [self.foncAct.fonction(entree)] #insertion des valeurs d'entree dans les premiers neurones
        for p,b in zip(self.poids,self.biais):
            self.neurones.append(self.foncAct.fonction(np.dot(p,self.neurones[-1])+b))
        
    def calcErr(self,valTheor):
        valTheor = np.asarray(valTheor) #entree sous forme d'array
        valTheor.shape = (-1,self.t) #entree sous forme de matrice
        valExp = self.neurones[-1] #recuperation des valeurs de sortie calculees
        erreur = [self.foncAct.fonction(valExp)*self.cout.derivee(valExp,valTheor)]
        for p,n in zip(self.poids,self.neurones[:-1]):
            erreur.append(self.foncAct.fonction(n)*np.dot(p.transpose(),erreur[-1])) #e^(n-1) = f'(x^(n-1)).((P^T)*e^(n))
        return erreur[::-1]
        
    def modifPoids(self,var):
        for p,n,v in zip(self.poids,self.neurones[-1],var): #pas de poids sur la premiere couche
            p = p - self.coefA*np.dot(v,n.transpose())
                        
    def modifBiais(self,var):
        for b,v in zip(self.biais,var):
            b = b - self.coefA*v
            
    def entrainer(self,baseE,baseT,tailleBatch = self.t,nbrEntrainements):
        self.t = tailleBatch
        for i in range(nbrEntrainements):
            random.shuffle(baseE)
            paquetsEntrainement = [baseE[k:k+self.t] for k in range(0, len(baseE), self.t)]
            for paquet in paquetsEntrainement:
                calcul(paquet[:][0])
                erreur = calcErr(paquet[:][1])
                modifPoids(erreur)
                modifBiais(erreur)
