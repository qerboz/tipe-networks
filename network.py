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

n = 10
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
