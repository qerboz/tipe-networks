########
#Imports
########

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import glob #gestion de fichier

#######################
#Fonctions d'activation
#######################

def elu(x):
    """bijection de R dans ]-1,+inf[, continue, derivable strictement croissante"""
    if x <0:
        return np.exp(x)-1
    return x
v_elu = np.vectorize(elu)

def delu(x):
    """derivee de la fonction elu"""
    if x<0:
        return np.exp(x)
    return 1
v_delu = np.vectorize(delu)

def invelu(x):
    if x<0:
        return np.log(max(x,-0.99)+1)
    return x
v_invelu = np.vectorize(invelu)

def sig(x):
    return 1/(1+np.exp(-x))
v_sig = np.vectorize(sig)

def invsig(x):
    return -np.log((1/x) - 1)
v_invsig = np.vectorize(invsig)

def dsig(x):
    return -np.exp(-x)/(1+np.exp(-x))**2
v_dsig = np.vectorize(dsig)

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
        sortie = np.array([L[0] > 0.5])
    else:
        sortie = np.array([X == np.amax(L) for X in L])
    sortie.shape = (-1)
    return sortie

########
#Classes
########

class couche():
    def __init__(self,n,p):
        '''Entree : nb de neurones sur cette couche, et sur celle d'avant'''
        self.neurones = np.zeros((n,1))
        self.poids = np.random.random((n,p))
        self.biais = np.zeros((n,1))

class reseauNeuronal():
    def __init__(self,structure,c = 0.1, f = elu):
        self.reglageCoefA(c) #coef d'apprentissage par défaut
        self.reglageFoncAct(f) #fonction d'activation par défaut
        self.couches = [couche(structure[0],0)]#liste contenant les couches (des objets)
        for i in range(1,len(structure)): #Ne pas deborder dans la ligne suivante
            self.couches.append(couche(structure[i],structure[i-1]))
            
    def reglageFoncAct(self,f):
        self.foncAct = globals()['v_' + f.__name__]
        self.dFoncAct = globals()['v_d' + f.__name__]
        self.invFoncAct = globals()['v_inv' + f.__name__]
        
    def reglageCoefA(self,x):
        self.coefA = x
    
    def calcul(self,entree):
        entree = np.asarray(entree) #entree sous forme d'array
        entree.shape = (-1,1) #entree sous forme de colonne
        self.couches[0].neurones = self.foncAct(entree) #insertion des valeurs d'entree dans les premiers neurones
        for i in range(1,len(self.couches)):
            #print(self.couches[i].poids.shape,self.couches[i-1].neurones.shape,self.couches[i].biais.shape)
            #valeurs print("poids",self.couches[i].poids,"neur",self.couches[i-1].neurones,"biais",self.couches[i].biais)
            self.couches[i].neurones = self.foncAct(np.dot(self.couches[i].poids,self.couches[i-1].neurones) + self.couches[i].biais)
        
    def calcErr(self,valTheor):
        valTheor = np.asarray(valTheor) #entree sous forme d'array
        valTheor.shape = (-1,1) #entree sous forme de colonne
        valExp = self.couches[-1].neurones #recuperation des valeurs de sortie calculees
        self.erreur = [self.dFoncAct(self.invFoncAct(valExp))*(valExp-valTheor)]
        for c in range(len(self.couches)-1):
            self.erreur.append(self.dFoncAct(self.invFoncAct(self.couches[-2-c].neurones))*np.dot(self.couches[-1-c].poids.transpose(),self.erreur[-1])) #e^(n-1) = f'(x^(n-1)).((W^T)*e^(n))
        self.erreur = self.erreur[::-1]
        
    def modifPoids(self):
        for c in range(1,len(self.couches)): #pas de poids sur la premiere couche
            self.couches[c].poids = self.couches[c].poids - self.coefA*np.dot(self.erreur[c],self.couches[c-1].neurones.transpose())
                        
    def modifBiais(self):
        for c in range(len(self.couches)):
            self.couches[c].biais = self.couches[c].biais - self.coefA*self.erreur[c]
            
    def entrainer(self,baseE,baseT,n):
        e = len(baseE)
        t = len(baseT)
        nbSucces = 0
        for i in range(n):
            for j in range(e):
                self.calcul(baseE[j][0])
                self.calcErr(baseE[j][1])
                self.modifPoids()
                self.modifBiais()
        for j in range(t):
            self.calcul(baseT[j][0])
            if np.array_equal(np.asarray(baseT[j][1]),normaliser(self.couches[-1].neurones)):
                nbSucces += 1
        return np.floor(100*nbSucces/t)
        
#####
#Main
#####

##Obtention des especes
os.chdir(".\TrainDB")
donneesEntrainementNoms = glob.glob("*.png")
especes = np.unique(np.array([name.split('_')[0] for name in donneesEntrainementNoms]))
sorties = [(especes[i],np.array([(i==j) for j in range(len(especes))])) for i in range(len(especes))]
for X in sorties:
    X[1].shape = (-1,1)
sorties = dict(sorties)

##Obtention des donnees d'entrainement
donneesEntrainement = [(lectImg(X),sorties[X.split('_')[0]]) for X in donneesEntrainementNoms]

##Obtention des donnees de test
os.chdir("..\TestDB")
donneesTestNoms = glob.glob("*.png")
donneesTest = [(lectImg(X),sorties[X.split('_')[0]]) for X in donneesTestNoms]

##Test
reconnaisseur = reseauNeuronal([12,10,6,4],0.1,sig)

n = 10
p = 100
T = [p*i for i in range(n+1)]
A = [0]
for a in range(n):
    A.append(reconnaisseur.entrainer(donneesEntrainement,donneesTest,p))
plt.plot(T,A) #, label = int(c*10)/10)
#plt.legend(loc = 'upper left')
plt.show()
