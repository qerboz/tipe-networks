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
    sortie = np.array([X == np.amax(L) for X in L])
    sortie.shape = (-1,1)
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
    return [1*(X == max(L)) for X in L]

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
 
    def reglerCoefA(self,x):
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
        for c in range(len(self.couches)-1): #Pour chaque couche du reseau moins la derniere (deja effectuee)
            liste_e_c = self.erreur[-1]
            e_c = []
            for k in range(len(self.couches[-1-(c+1)].neurones)):
                der = dElu(invElu(self.couches[-1-(c+1)].neurones[k]))
                somme = 0
                for j in range(len(self.couches[-1-c].poids)):
                    w_j_k = self.couches[-1-c].poids[j][k]
                    somme += w_j_k * liste_e_c[j]
                e_c.append(der*somme)
            self.erreur.append(e_c)
                
            
 ###           self.erreur.append([dElu(invElu(self.couches[-1-k-1].neurones[i]))*sommeListe(produitListes([X[i] for X in self.couches[-1-k].poids],self.erreur[-1])) for i in range(len(self.couches[-1-k-1].neurones))])
        self.erreur = retourner(self.erreur)
                
    def modifPoids(self):
        for c in range(1,len(self.couches)):
            for j in range(len(self.couches[c].poids)):
                for k in range(len(self.couches[c].poids[j])):
                    self.couches[c].poids[j][k] -= self.erreur[c][j]*self.coefA*self.couches[c-1].neurones[k]

    def modifBiais(self):
        for c in range(1,len(self.couches)):
            for k in range(len(self.couches[c].neurones)):
                self.couches[c].biais[k] -= self.erreur[c][k]*self.coefA

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

n = 100
p = 20
T = [p*i for i in range(n+1)]
c = 0.1
for b in range(7):
    A = [0]
    reconnaisseur.reglerCoefA(c)
    for a in range(n):
        A.append(reconnaisseur.entrainer(donneesEntrainement,donneesTest,p))
    plt.plot(T,A, label = int(c*10)/10)
    c += 0.2
plt.legend(loc = 'upper left')
plt.show()








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
 
    def reglerCoefA(self,x):
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

n = 100
p = 20
T = [p*i for i in range(n+1)]
c = 0.1
for b in range(7):
    A = [0]
    reconnaisseur.reglerCoefA(c)
    for a in range(n):
        A.append(reconnaisseur.entrainer(donneesEntrainement,donneesEntrainement,p))
    plt.plot(T,A, label = int(c*10)/10)
    c += 0.2
plt.legend(loc = 'upper left')
plt.show()
