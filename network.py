########
#Imports
########

from random import random
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import glob #gestion de fichier


##########
#Fonctions
##########

###Generales
def enleverDoublons(liste):
    ''' Entree: Une liste
        Sortie: La liste sans les doublons
        note: Preserve l'ordre
        '''
    sortie = []
    for x in liste:
        if x not in sortie:
            sortie.append(x)
    return sortie

def produitListes(L1,L2):
    """produit scalaire de deux listes"""
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

def retourner(L):
    return L[::-1]

def normaliser(L):
    return [X == max(L) for X in L]

###Reseau de neurones  
def elu(x):
    """bijection de R dans ]-1,+inf[, continue, derivable strictement croissante"""
    if x <0:
        return np.exp(x)-1
    return x
  
def dElu(x):
    """derivee de la fonction elu"""
    if x<0:
        return np.exp(x)
    return 1

def invElu(x):
    if x<0:
        return np.log(x+1)
    return x

###Traitement d'images
def pixelValue(pix):
    '''Entree: Un tableau de trois valeurs representant un pixel
        Sortie: Ces trois valeurs concatenees.
    '''
    return ('00'+str(pix[0]))[-3::] #+ ('00'+str(pix[1]))[-3::] + ('00'+str(pix[2]))[-3::]

def imgEnNombres(image):
    '''Entree: Une image de taille nxp sous la forme d'un array numpy contenant les valeurs RGB de chaque pixel
        Sortie: Une liste de même taille que le nombre de pixels de l'image contenant des valeurs representant les pixels''' 
    n= len(image)
    p= len(image[0])
    return [int(pixelValue(image[i//p][i%p]))/255 for i in range(n*p)]

########
#Classes
########

class couche():
    """entrer le nombre de neurones sur cette couche et sur la couche precedente"""
    def __init__(self,n,p):
        self.neurones = [random() for i in range(n)]
        self.poids = [[random() for i in range(p)] for j in range(n)]
        self.biais = [random() for i in range(n)]

class reseauNeuronal():
    """entrer une liste comportant autant de termes qu'il y a de couches dans le reseau, et dont chaque terme correspond au nombre de neurones sur la couche associee à ce terme"""
    def __init__(self,L):
        self.coefA = 0.1
        self.couches = [couche(L[0],0)]#liste contenant les couches (des objets)
        for i in range(1,len(L)): #Ne pas deborder dans la ligne suivante
            self.couches.append(couche(L[i],L[i-1])) #creation de chaque couche du reseau
 
    def setCoefA(self,x):
        self.coefA = x
 
    def calcul(self,X): #calcule la sortie en fonction de l'entree
        for j in range(len(X)):
            self.couches[0].neurones[j] = elu(X[j])
        self.transfert(0,len(self.couches))
        return self.couches[-1].neurones
 
    def transfert(self,i,n): #transfert des donnees des neurones d'une couche i vers la couche suivante
        if i >= n-1: #arrête la recursivite
            return
        for k in range(len(self.couches[i+1].neurones)):
            valeur =  elu(sommeListe(produitListes(self.couches[i].neurones,self.couches[i+1].poids[k]))+self.couches[i+1].biais[k])
            self.couches[i+1].neurones[k] = valeur
        self.transfert(i+1,n)#recursivite pour transferer les donnees de la premiere couche à la derniere
 
    def cout(self,X,Y):
        self.C = 0
        calcul(self,X)
        for i in range(len(self.couches[-1].neurones)):
            self.C += (self.couches[-1].neurones[i]-elu(Y[i]))**2

    def calcErreur(self,theor):
        Y = self.couches[-1].neurones #Couche de sortie/reponse
        self.erreur = [[dElu(invElu(Y[i]))*(Y[i] - theor[i]) for i in range(len(Y))]]#La liste accueillant erreurs du cout par rapport à chaque poids de la derniere couche
        for k in range(len(self.couches)-1): #Pour chaque couche du reseau moins la derniere (deja effectuee)
            self.erreur.append([dElu(invElu(self.couches[-1-k-1].neurones[i]))*sommeListe(produitListes([X[i] for X in self.couches[-1-k].poids],self.erreur[-1])) for i in range(len(self.couches[-1-k-1].neurones))])
        self.erreur = retourner(self.erreur)
                
    def modifPoids(self):
        for k in range(1,len(self.couches)):
            for i in range(len(self.couches[k].poids)):
                for j in range(len(self.couches[k].poids[i])):
                    self.couches[k].poids[i][j] -= self.erreur[k][i]*self.coefA*self.couches[k-1].neurones[j]

    def modifBiais(self):
        for i in range(1,len(self.couches)):
            for j in range(len(self.couches[i].neurones)):
                self.couches[i].biais[j] -= self.erreur[i][j]*self.coefA

    def entrainer(self,baseE,baseT,nbr):
        n = len(baseE)
        m = len(baseT)
        compteur = 0
        for i in range(nbr):
            for j in range(n):
                self.calcul(baseE[j][0])
                self.calcErreur(baseE[j][1])
                self.modifPoids()
                self.modifBiais()
        for j in range(n):
            self.calcul(baseT[j][0])
            if baseT[j][1] == normaliser(self.couches[-1].neurones):
                compteur += 1
        return np.floor(100*compteur/m)


#####
#Main
#####

###obtention des donnees d'entrainement
os.chdir(".\TrainDB")
donneesEntrainementNoms = glob.glob("*.png")
especes = enleverDoublons([name.split("_")[0] for name in donneesEntrainementNoms])

sorties = [(especes[i],[1*(i==j) for j in range(len(especes))]) for i in range(len(especes))]
sorties = dict(sorties)

donneesEntrainement = [np.floor(img.imread(donneesEntrainementNoms[i])*255).astype(np.uint8) for i in range(len(donneesEntrainementNoms))]
donneesEntrainement = [(imgEnNombres(donneesEntrainement[i]),sorties[donneesEntrainementNoms[i].split("_")[0]]) for i in range(len(donneesEntrainement))]


###obtention des donnees de test
os.chdir("..\TestDB")
donneesTestNoms = glob.glob("*.png")
donneesTest = [np.floor(img.imread(donneesTestNoms[i])*255).astype(np.uint8) for i in range(len(donneesTestNoms))]
donneesTest = [(imgEnNombres(donneesTest[i]),sorties[donneesTestNoms[i].split("_")[0]]) for i in range(len(donneesTest))]

reconnaisseur = reseauNeuronal([len(donneesEntrainement[0][0]),int((len(donneesEntrainement[0][0])+len(especes))/2),len(especes)])

n = 10
p = 100
T = [p*i for i in range(n+1)]
A=[0]
for a in range(n):
    A.append(reconnaisseur.entrainer(donneesEntrainement,donneesEntrainement,p))
plt.plot(T,A)
plt.show()
