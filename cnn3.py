########
#Imports
########

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import glob #gestion de fichier
import random
import time

###################
#Fonctions diverses
###################

def lectImg(nom):
    Img = img.imread(nom)
    if Img.shape[2] > 3:
        Img = Img[:,:,0:3]
    return Img

def sauvegarde(reseau,fichier):
    os.chdir('./'+fichier)
    np.save('filtres.npy',reseau.filtres)
    np.save('pools.npy',reseau.pools)

def charger(reseau,fichier):
    os.chdir('./'+fichier)
    reseau.filtres = np.load('filtres.npy')
    reseau.pools = np.load('pools.npy')

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
    r,s = filtre[0],filtre[1]
    a,b = m%r,n%s
    if b != 0:
        entree = np.concatenate((entree,np.zeros((m,r-b,p))-10),axis = 1)
    n = entree.shape[1]
    if a != 0 :
        entree = np.concatenate((entree,np.zeros((r-a,n,p))-10),axis = 0)
    m = entree.shape[0]
    sortie = np.zeros((m//r,n//s,p))
    sortieIndices = np.zeros((m//r,n//s,p,3))
    for i in range(0,m-r+1,r):
        for j in range(0,n-s+1,s):
            for k in range(0,p):
                extrait = entree[i:i+r,j:j+s,k]
                maxp = np.max(extrait)
                sortie[i//r,j//s,k] = maxp
                coord = np.where(entree[i:i+r,j:j+s,k] == maxp)
                u,v = coord[0][0],coord[1][0]
                sortieIndices[i//r,j//s,k] = (u+i,v+j,k)
    liste.append(sortieIndices.astype(int))
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
        self.filtres = [[np.random.random(x[1])*2-1 for i in range(x[0])] for x in structure] #Init des coeffs entre -1 et 1
        self.pools = [x[2] for x in structure]
        self.cout = cout
        self.max = [[] for f in self.filtres]
                 
    def calcul(self,entree):
        self.max = [[] for f in self.filtres]
        self.neurones = [self.foncAct.fonction(entree)]
        for i in range(0,len(self.filtres)):
            self.neurones.append(self.foncAct.fonction(reduction(conv3D(self.neurones[-1],self.filtres[i][0]),self.pools[i],self.max[i])))
            for f in self.filtres[i][1:]:
                self.neurones[-1] = np.append(self.neurones[-1],self.foncAct.fonction(reduction(conv3D(self.neurones[-2],f),self.pools[i],self.max[i])),axis = 2)

    def calcErr(self,erreurSortie):
        erreur = [erreurSortie]
        for i in range(0,len(self.filtres)-1):
            erreur.append(np.zeros((self.neurones[-2-i].shape)))
            for j in range(len(self.filtres[-1-i])):
                f = self.filtres[-1-i][j]
                forme = f.shape
                for k in range(len(erreur[-2])):
                    for l in range(len(erreur[-2][0])):
                        (a,b,c) = tuple(self.max[-1-i][j][k,l,0])
                        erreur[-1][a:a+forme[0],b:b+forme[1],c:c+forme[2]] += erreur[-2][k,l,j]*f*self.foncAct.derivee(self.neurones[-2-i][a:a+forme[0],b:b+forme[1],c:c+forme[2]])
                        #erreur[-1][a:a+forme[0],b:b+forme[1],c:c+forme[2]] += erreur[-2][k,l,j]*f*self.foncAct.derivee(self.neurones[-1-i][k,l,j])
        erreur = erreur[::-1]
        erreurPoids = [[np.zeros(f.shape) for f in listef] for listef in self.filtres]
        for i in range(len(erreurPoids)):
            for j in range(len(erreurPoids[i])):
                forme = erreurPoids[i][j].shape
                for k in range(len(erreur[i])):
                    for l in range(len(erreur[i][0])):
                        (a,b,c) = tuple(self.max[i][j][k,l,0])
                        erreurPoids[i][j] += erreur[i][k,l,j]*self.neurones[i][a:a+forme[0],b:b+forme[1],c:c+forme[2]]
        return erreurPoids
        
    def calcErrBis(self,erreurSortie):
        erreur = [erreurSortie]
        for n in range(1,len(self.filtres)):
            erreur.append(np.zeros((self.neurones[-1-n].shape)))
            for i in range(len(erreur[n])):
                for j in range(len(erreur[n][0])):
                    for k in range(len(erreur[n][0,0])):
                        for nf in range(len(self.filtres[-n])):
                            f = self.filtres[-n][nf]
                            a,b,c = f.shape
                            for iloc in range(max(i-(a-1),0),min(i+1,i-a+1)):
                                for jloc in range(max(j-(b-1),0),min(j+1,j-b+1)):
                                    erreur[n][i,j,k] += erreur[n-1][iloc,jloc,nf]*self.foncAct.derivee(self.neurones[-n][iloc,jloc,nf])*f[i-iloc,j-jloc,k]
        erreur = erreur[::-1]
        erreurPoids = [[np.zeros(f.shape) for f in listef] for listef in self.filtres]
        for fi in range(len(erreurPoids)):
            for fj in range(len(erreurPoids[fi])):
                a,b,c = erreurPoids[fi][fj].shape
                for i in range(len(erreur[fi])):
                    for j in range(len(erreur[fi][0])):
                        erreurPoids[fi][fj] += erreur[fi][i,j,fj]*self.neurones[fi][i:i+a,j:j+b,:]
        return erreurPoids
        
    def calcErrTer(self,erreurSortie):
        erreur = [erreurSortie]
        erreurPoids = [[np.zeros(f.shape) for f in listef] for listef in self.filtres]
        for i in range(len(self.filtres)):
            erreur.append(np.zeros((self.neurones[-1-i].shape)))
            for j in range(len(self.filtres[i])):
                erreur[-1],erreurPoids[i][j] = conv_backward(erreur[-2],(self.neurones[-i-1],self.filtres[i][j]))
                erreur[-1] = erreur[-1]*self.foncAct.derivee(self.neurones[-i-1])
        return erreurPoids

    def modifPoids(self,var):
        for i in range(len(var)):
            for j in range(len(var[i])):
                self.filtres[i][j] -= self.coefA*var[i][j]

class PMC():
    def __init__(self,structure,f = elu,c = 0.1,cout = coutQuad, fusion = False):
        self.fusion = fusion #fusion du réseau avec un cnn
        self.reglageCoefA(c) #coef d'apprentissage par defaut
        self.foncAct = f
        self.poids = [np.random.random((n,p))*2-1 for n,p in zip(structure[1:],structure[:-1])]
        self.biais = [np.zeros((n,1)) for n in structure[1:]]
        #self.poids = [np.array([[[1]*p]*n]).reshape(n,p) for n,p in zip(structure[1:],structure[:-1])]
        self.cout = cout

    def reglageCoefA(self,x):
        self.coefA = x
    
    def calcul(self,entree):
        if self.fusion:
            self.neurones = [self.foncAct.fonction(entree)]
        else:
            self.neurones = [entree] #insertion des valeurs d'entree dans les premiers neurones
        for p,b in zip(self.poids[:-1],self.biais[:-1]):
            self.neurones.append(self.foncAct.fonction(np.dot(p,self.neurones[-1])+b))
        self.neurones.append(self.cout.fonction(np.dot(self.poids[-1],self.neurones[-1])+self.biais[-1]))
        
    def calcErr(self,valTheor):
        valExp = self.neurones[-1] #recuperation des valeurs de sortie calculees
        erreur = [self.foncAct.derivee(valExp)*self.cout.derivee(valExp,valTheor)]
        for p,n in zip(self.poids[::-1],self.neurones[:-1][::-1]):
            erreur.append(self.foncAct.derivee(n)*np.dot(p.transpose(),erreur[-1])) #e^(n-1) = f'(x^(n-1)).((P^T)*e^(n))
        if self.fusion:
            self.erreurEntree = erreur[-1]
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
            self.poids[i] = 0.99999*self.poids[i] - self.coefA*var[i]
                        
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

class reseau():
    def __init__(self,cnn,pmc):
        self.cnn = cnn
        self.pmc = pmc
        self.pmc.fusion = True

    def calcul(self,entree):
        self.cnn.calcul(entree)
        sortieCNN = self.cnn.neurones[-1]
        entreePMC = sortieCNN.flatten().reshape(-1,1)
        self.pmc.calcul(entreePMC)

    def entrainer(self,baseE,nbrEntrainements):
        for i in range(nbrEntrainements):
            print("entrainement {}.{}".format(epoque,i+1))
            random.shuffle(baseE)
            j = 0
            for donnee in baseE:
                j += 1
                print("image {}.{}.{}".format(epoque,i+1,j))
                entree,sortie = donnee[0],donnee[1]
                self.calcul(entree)
                erreurPoidsPMC,erreurBiaisPMC = self.pmc.calcErr(sortie)
                self.pmc.modifPoids(erreurPoidsPMC)
                self.pmc.modifBiais(erreurBiaisPMC)
                erreurPoidsCNN = self.cnn.calcErrBis(self.pmc.erreurEntree.reshape(self.cnn.neurones[-1].shape))
                self.cnn.modifPoids(erreurPoidsCNN)

    def perf(self,baseT):
        score = 0
        for i in baseT:
            self.calcul(i[0])
            if np.array_equal(i[1],normaliser(self.pmc.neurones[-1])):
                score += 1
        return score/len(baseT)

##Obtention des especes
os.chdir("/home/cmp1/petitjean.antoine/petitjean.antoine/TrainDB")
donneesEntrainementNoms = glob.glob("*.png")
especes = np.unique(np.array([name.split('_')[0] for name in donneesEntrainementNoms]))
sorties = [(especes[i],np.array([1*(i==j) for j in range(len(especes))])) for i in range(len(especes))]
for X in sorties:
    X[1].shape = (-1,1)
sorties = dict(sorties)

##Obtention des donnees d'entrainement
donneesEntrainement = [(lectImg(X),sorties[X.split('_')[0]]) for X in donneesEntrainementNoms]

##Obtention des donnees de test
os.chdir("/home/cmp1/petitjean.antoine/petitjean.antoine/TestDB")
donneesTestNoms = glob.glob("*.png")
donneesTest = [(lectImg(X),sorties[X.split('_')[0]]) for X in donneesTestNoms]


t = time.time()
reconnaisseur = reseau(CNN([[10,(2,2,3),(1,1)],[10,(2,2,10),(1,1)]],sigmoide),PMC([90,50,20,7],sigmoide))

def conv_backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    
    dx, dw, db = None, None, None

    # Récupération des variables
    x, w= cache
    pad = 1
    stride = 1
    
    # Initialisations
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    
    # Dimensions
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_, W_ = dout.shape
    
    # - dout (N, F, H', W')
    # On somme sur tous les éléments sauf les indices des filtres
    
    # dw = xp * dy
    # 0-padding juste sur les deux dernières dimensions de x
    xp = np.pad(x, ((0,), (0,), (pad,), (pad, )), 'constant')
    
    # Version sans vectorisation
    for n in range(N):       # On parcourt toutes les images
        for f in range(F):   # On parcourt tous les filtres
            for i in range(HH): # indices du résultat
                for j in range(WW):
                    for k in range(H_): # indices du filtre
                        for l in range(W_):
                            for c in range(C): # profondeur
                                dw[f,c,i,j] += xp[n, c, stride*i+k, stride*j+l] * dout[n, f, k, l]

    # dx = dy_0 * w'
    # Valide seulement pour un stride = 1
    # 0-padding juste sur les deux dernières dimensions de dy = dout (N, F, H', W')
    doutp = np.pad(dout, ((0,), (0,), (WW-1,), (HH-1, )), 'constant')

    # 0-padding juste sur les deux dernières dimensions de dx
    dxp = np.pad(dx, ((0,), (0,), (pad,), (pad, )), 'constant')

    # filtre inversé dimension (F, C, HH, WW)
    w_ = np.zeros_like(w)
    for i in range(HH):
        for j in range(WW):
            w_[:,:,i,j] = w[:,:,HH-i-1,WW-j-1]
    
    # Version sans vectorisation
    for n in range(N):       # On parcourt toutes les images
        for f in range(F):   # On parcourt tous les filtres
            for i in range(H+2*pad): # indices de l'entrée participant au résultat
                for j in range(W+2*pad):
                    for k in range(HH): # indices du filtre
                        for l in range(WW):
                            for c in range(C): # profondeur
                                dxp[n,c,i,j] += doutp[n, f, i+k, j+l] * w_[f, c, k, l]
    #Remove padding for dx
    dx = dxp[:,:,pad:-pad,pad:-pad]

    return dx,dw

nbEpoques = 10
nbLect = 50
T = [nbLect*i for i in range(nbEpoques+1)]
A = [0]
for epoque in range(1,nbEpoques+1):
    print("epoque : {}".format(epoque))
    random.shuffle(donneesEntrainement)
    random.shuffle(donneesTest)
    reconnaisseur.entrainer(donneesEntrainement,nbLect)
    T.append(T[-1]+nbLect)
    A.append(reconnaisseur.perf(donneesTest))
    print("perf : {}".format(A[-1]))
plt.plot(T,A)
plt.show()