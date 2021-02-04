from itertools import chain
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from scipy import *
import math
import sys
from numpy.polynomial import Polynomial
import numpy

#%%%-------------------------------------------------------------------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%
#%%%------------------------------- DEBUT PARTIE ANALYSE D'IMAGES -----------------------%%%
#%%%-------------------------------------------------------------------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%


###################################################
#Fonction : rgb2gray
#But : Transforme une image de couleur composée de pixels RGB en une image grise
#Preconditions : aucunes
#Paramètre(s) :
# 	- rgb : array
#Postconditions : renvoie un array de pixels remanié pour obtenir des nuances de gris
#Auteur :  Guillaume NOUAL
###################################################

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#-----------------


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  DEBUT LBP  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


###################################################
#Fonction : recup_matrice3x3
#But : Récupère par rapport a à un élément d'une matrice , les éléments qui sont collés à lui , pour obtenir une matrice de taille 3x3
#Preconditions : matrice non vide , l initialisé à []
#Paramètre(s) :
# 	- matrice : liste de liste d'entiers
#	- l : liste d'entiers
#	- i : entier
#	- j : entier
#Postconditions : Renvoie une liste d'entiers de longueur 9
#Auteur :  Guillaume NOUAL
###################################################

def recup_matrice3x3(matrice,l,i,j):
	for k in range(i-1,i+2):
		for p in range(j-1,j+2):
			l.append(matrice[k][p])
	return(l)
#-------------------------------------------    # fonction à appeler
def recup_matrice_REC(matrice,i,j):
	return(recup_matrice3x3(matrice,[],i,j))

#-----------------


###################################################
#Fonction : somme
#But : Ajoute tous les éléments de la liste sauf celui du milieu à la position 4
#Preconditions : l non vide , stock initialisé à 0
#Paramètre(s) :
#	- l : liste d'entiers
# 	- stock : entier
#Postconditions : Renvoie un entier
#Auteur :  Guillaume NOUAL
###################################################

def somme(l,stock):
	for i in range(0,len(l)):
		if i==4:
			stock=stock+0
		else:
			stock=stock+l[i]
	return(stock)
#------------------------------------------- # fonction à appeler
def somme_REC(l):
	return(somme(l,0))

#-----------------


###################################################
#Fonction : LBP_calcul
#But : Applique les étapes de 'Différence' , 'Signe' et de 'Pondération' de l'algorithme de calcul des LBP
#Preconditions : l non vide , acc initialisé à l[4] , l2 initialisé à [] et stock initialisé à 0
#Paramètre(s) :
#	- l : liste d'entiers
# 	- acc : entier
#	- l2 : liste d'entiers
#	- stock : entier
#Postconditions : Renvoie une liste d'entiers
#Auteur :  Guillaume NOUAL
###################################################

def LBP_calcul(l,acc,l2,stock):
	for i in range(0,len(l)):
		if i==4:
			l2.append(l[i])
		else:
			res=l[i]-acc
			if res<0:
				l2.append(0)
				stock=stock+1
			else:
				l2.append(1*(2**stock))
				stock=stock+1
	return(l2)
#------------------------------------------- # fonction à appeler
def LBP_calcul_rec(l):
	return(LBP_calcul(l,l[4],[],0))

#-----------------


###################################################
#Fonction : LBP
#But : Applique toutes les étapes de l'algorithme du calcul des LBP sur un élément d'une matrice
#Preconditions : l non vide
#Paramètre(s) :
#	- l : liste de liste d'entiers
#	- i : entier
# 	- j : entier
#Postconditions : Renvoie un entier
#Auteur :  Guillaume NOUAL
###################################################

def LBP(l,i,j):
	return(somme_REC(LBP_calcul_rec(recup_matrice_REC(l,i,j))))

#-----------------


###################################################
#Fonction : apply_LBP
#But : A partir d'une matrice , crée une nouvelle matrice en lui appliquant des changements sur chacun des éléments de la première matrice
#Preconditions : matrice non vide , matrice2 initialisé à []
#Paramètre(s) :
#	- matrice : liste de liste d'entiers
# 	- matrice2 : liste de liste d'entiers
#Postconditions : Renvoie une liste de liste d'entiers
#Auteur :  Guillaume NOUAL
###################################################

def apply_LBP(matrice,matrice2):
	matrice2.append((matrice[0]))
	for i in range(1,(len(matrice)-1)):
		stock=1
		lstock=[]
		lstock.append(matrice[i][0])
		while stock<(len(matrice[0])-1):
			lstock.append((LBP(matrice,i,stock)))
			stock=stock+1
		lstock.append(matrice[i][-1])
		matrice2.append(lstock)
	matrice2.append((matrice[-1]))
	return(matrice2)

#-----------------


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FIN LBP  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  DEBUT POI  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

###################################################
#Fonction : calcul_Gx
#But : A partir d'une matrice déjà existante , on calcule une opération par rapport à elle 
#Preconditions : tab matrice non vide
#Paramètre(s) : 
#    - i : entier
#	 - j : entier
#	 - tab : liste de liste d'entiers
#Postconditions : renvoie un entier
#Auteur :  Guillaume NOUAL
###################################################

def calcul_Gx(i,j,tab):
	res=tab[i+1][j]-tab[i-1][j]
	return(res)

#-----------------


###################################################
#Fonction : calcul_Gy
#But : A partir d'une matrice déjà existante , on calcule une opération par rapport à elle 
#Preconditions : tab matrice non vide
#Paramètre(s) : 
#    - i : entier
#	 - j : entier
#	 - tab : liste de liste d'entiers
#Postconditions : renvoie un entier
#Auteur :  Guillaume NOUAL
###################################################

def calcul_Gy(i,j,tab):
	res=tab[i][j+1]-tab[i][j-1]
	return(res)

#-----------------


###################################################
#Fonction : matrice_Gx_Gy
#But : Crée deux matrices une en fonction de la fonction de calcul_Gx et une autre en fonction de calcul_Gy
#Preconditions : matrice non vide
#Paramètre(s) : 
#	 - matrice : liste de liste d'entiers
#Postconditions : retourne un couple de liste de liste d'entiers
#Auteur :  Guillaume NOUAL
###################################################

def matrice_Gx_Gy(matrice):
	final_Gx=[]
	final_Gy=[]
	final_Gx.append(matrice[0])
	final_Gy.append(matrice[0])
	for i in range(1,len(matrice)-1):
		l_Gx=[]
		l_Gy=[]
		l_Gx.append(matrice[i][0])
		l_Gy.append(matrice[i][0])

		for j in range(1,len(matrice[0])-1):
			res_Gx=calcul_Gx(i,j,matrice)
			res_Gy=calcul_Gy(i,j,matrice)
			l_Gx.append(res_Gx)
			l_Gy.append(res_Gy)

		l_Gx.append(matrice[i][-1])
		l_Gy.append(matrice[i][-1])
		final_Gx.append(l_Gx)
		final_Gy.append(l_Gy)

	final_Gx.append(matrice[-1])
	final_Gy.append(matrice[-1])
	return(final_Gx,final_Gy)

#-----------------


###################################################
#Fonction : balayage_second
#But : Correspond à la Step 1 du sujet sur l'algorithme du calcul des points d'intérêts par la méthode de détecteur de Harris
#Preconditions : k compris entre 0.05 et 0.15 , valeur optimale 0.04 , matrice non vide
#Paramètre(s) : 
#	 - matrice : liste de liste d'entiers
#	 - k : float
#Postconditions : retourne une liste de liste d'entiers
#Auteur :  Guillaume NOUAL
###################################################

def balayage_second(matrice,k):
	lfinal=[]
	for i in range(1,len(matrice[0])-1):
		l=[]
		l.append((matrice[0])[i][0])
		for j in range(1,len((matrice[0])[0])-1):

			res_GX=0
			res_GY=0
			res_GXY=0
			for iw in range(-1,1):
				for jw in range(-1,1):
					res_GX=res_GX+ (((matrice[0])[i+iw][j+jw])**2)
					res_GY=res_GY+ (((matrice[1])[i+iw][j+jw])**2)
					res_GXY=res_GXY+ ((matrice[0])[i+iw][j+jw])*((matrice[1])[i+iw][j+jw])

			combo=[[res_GX,res_GXY],[res_GXY,res_GY]]
			# det=(combo[0][0])*(combo[1][1])-(combo[1][0])^2
			det=(combo[0][0])*(combo[1][1])-(combo[1][0])*(combo[1][0])
			tr=combo[0][0]+combo[1][1]
			res_combo=det-float(k)*tr*tr
			l.append(res_combo)
		lfinal.append(l)
	# lfinal.append((matrice[0])[-1])
	return(lfinal)

#-----------------


###################################################
#Fonction : seuil
#But : Calcule le seuil d'une image en additionnant tous ses pixels et en divisant cette valeur par le nombre d'éléments présents dans la matrice
#Preconditions : matrice non vide
#Paramètre(s) : 
#	 - matrice : liste de liste d'entiers
#Postconditions : retourne un float
#Auteur :  Guillaume NOUAL
###################################################

def seuil(matrice):
	acc_compteur=0
	acc_add=0
	res=0
	for i in range(0,len(matrice)-1):
		for j in range(0,len(matrice[0])-1):
			if matrice[i][j]>=0:
				acc_add=acc_add+abs(matrice[i][j])
				acc_compteur=acc_compteur+1
	res=acc_add/acc_compteur
	return(res)

#-----------------


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FIN HARRIS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




###################################################
#Fonction : critere_comparaison
#But : évaluer un score pour quantifier les résultats du déflouage ,visant à évaluer la ressemblance entre deux images
#Preconditions : matrice non vide , l initialisé à []
#Paramètre(s) :
# 	- original : liste de liste d'entiers (image de référence)
#	- image : liste de liste d'entiers (image à comparer)
#Postconditions : Renvoie un pourcentage 
#Auteur :  Guillaume NOUAL
###################################################

def critere_comparaison (original,image):
	tmp = 0
	ligne = len(original)
	colonne = len(original[0])
	for x in range (ligne):
		for y in range (colonne):
			if (original[x][y]-10<image[x][y]) and (image[x][y]<original[x][y]+10) :
				tmp += 1
	nbrpixel = ligne*colonne
	res = (tmp/nbrpixel)*100
	return(res)





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  DEBUT BLUR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


###################################################
#Fonction : blur
#But : Appliquer blur sur une image
#Preconditions : I est initialisée
#Postconditions : Renvoie la matrice après le traitement blur (floutage)
#Auteur :  Guillaume NOUAL
###################################################

def blur(I):
	M = [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
	for k in range(2):
		I = signal.convolve2d(I,M,mode='full', boundary='fill', fillvalue=0)
	ligneI = len(I)
	colonneI =len(I[0])
	for y in range(ligneI):
		for x in range(colonneI):
			I[y][x] = np.around(I[y][x])
			if I[y][x]<0 :
				I[y][x]=0
			elif I[y][x]>255:
				I[y][x]=255
			else:
				I[y][x]
	return(I)

#-----------------


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FIN BLUR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  DEBUT DEBLUR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


###################################################
#Fonction : deblur
#But : Appliquer deblur sur une image
#Preconditions : I est initialisée
#Postconditions : Renvoie la matrice après le traitement deblur (défloutage)
#Auteur :  Guillaume NOUAL
###################################################

def deblur(I):
	J = I
	d =[[[0,0,0,0,1],[0,0,0,1,1],[0,0,1,1,1],[0,0,0,0,0],[0,0,0,0,0]],[[0,0,1,1,1],[0,0,1,1,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],[[1,1,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],[[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],[[0,0,0,0,0],[0,0,0,0,0],[1,1,1,0,0],[1,1,0,0,0],[1,0,0,0,0]],[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,1,1,0,0],[1,1,1,0,0]],[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,1,1,0],[0,0,1,1,1]],[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,1,1],[0,0,0,1,1],[0,0,0,0,1]]]
	n = 5
	ligneI = len(I)
	colonneI = len(I[0])
	for y in range(1,ligneI-4):
		for x in range(1,colonneI-4):
			SDG = 0
			SG = 0
			for k in range(8):
				S = 0
				for j in range(5):
					for i in range(5):
						res = I[y+j-2][x+i-2]*d[k][j][i]
						S = S + res
						
				S = S/6
				S2 = 0
				for j in range(5):
					for i in range(5):
						res = I[y+j-2][x+i-2]*d[k][j][i]
						S2 = S2 +(res**2)
				S2 = S2/6
				delta = S-I[y][x]
				G = (abs(S2-(S**2)))**(n/2)
				SDG = SDG +delta*G
				SG = SG +G			
			if SG !=0 :
				J[y][x] = round(I[y][x]-(SDG/SG))
			else :
				J[y][x] = I[y][x]
			if J[y][x] < 0 :
				J[y][x] = 0
			elif J[y][x] > 255 :
				J[y][x] = 255

	return(J)
		
#-----------------


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FIN DEBLUR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




#%%%-------------------------------------------------------------------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%
#%%%------------------------------- FIN PARTIE ANALYSE D'IMAGES -------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%




#%%%-------------------------------------------------------------------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%
#%%%------------------------------- DEBUT PARTIE AUTOMATIQUE ----------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  DEBUT ROUTH  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


###################################################
#Fonction : polynome
#But : Demande le degré et les coefficients d'un polynome à l'utilisateur
#Preconditions : aucunes
#Paramètre(s) : aucuns
#Postconditions : Renvoie un couple d'un polynome sous forme de liste et un polynome en string
#Auteur :  Guillaume NOUAL
###################################################

def polynome():
	degre=int(input("Quel est le degre de votre polynome ? \n"))
	while degre<=0:
		print("Entrer un degre different de "+str(degre)+" ")
		degre=int(input("Quel est le degre de votre polynome ? \n"))
	l=[]
	poly=""
	for i in range(0,degre+1):
		b=str(i)
		valeur=input("Entrer le coefficient de degres "+b+" \n")
		poly=poly+str(valeur)+"*p^"+b+" + "
		l.append(valeur)
	liste=poly.split("+")
	del liste[-1]
	liste.reverse()
	liste2=[]
	for k in range(0,len(liste)):
		liste2.append(liste[k])
		liste2.append("+")
	del liste2[-1]
	poly="".join(liste2)
	return (l,poly)


#-----------------

###################################################
#Fonction : demande_polynome
#But : Affiche un polynome sur le terminal
#Preconditions : aucunes
#Paramètre(s) : aucuns
#Postconditions : Renvoie la liste des coefficients du polynome
#Auteur :  Guillaume NOUAL
###################################################
def demande_polynome_routh():
	x=polynome()
	print("Votre polynme est : "+x[1]+"\n")
	liste=x[0]
	liste.reverse()
	return (liste)


#-----------------

###################################################
#Fonction : paire
#But : Permet de savoir si un nombre est paire
#Preconditions : aucunes
#Paramètre(s) : x : float ou int
#Postconditions : renvoie true si le chiffre est paire
#Auteur :  Guillaume NOUAL
###################################################

def paire(x):
	return(x%2==0)


#-----------------

###################################################
#Fonction : deux_premieres_lignes
#But : Permet de "faire" les 2 premiere lignes du tableau avec les puissances de p
#Preconditions : liste de la forme que renvoie demande_polynome_routh()
#Paramètre(s) : l_poly : liste reçu par la fonction demande_polynome_routh()
#Postconditions : 
#Auteur :  Guillaume NOUAL
###################################################

def deux_premieres_lignes(l_poly):
	degres=len(l_poly)-1
	tableau=[["p^"+str(degres)],["p^"+str(degres-1)]]
	for i in range(0,degres+1):
		if paire(degres)==paire(degres-i):
			tableau[0].append(l_poly[i])
		else:
			tableau[1].append(l_poly[i])
	if paire(degres):
		tableau[1].append("0")
	return(tableau)


#-----------------

###################################################
#Fonction : elem_i_j
#But : Calculer l'element (i,j) du tableau de routh
#Preconditions : i <= length(tableau) , j <= length (tableau(i))
#Paramètre(s) : tableau : liste de liste
#				i : int
# 				j : int
#Postconditions : retourne une valeur en float 
#Auteur :  Guillaume NOUAL
###################################################

def elem_i_j(tableau,i,j):
	if float(tableau[i-1][1])==0:
		div=0.000000000001
	else:
		div=float(tableau[i-1][1])
	if len(tableau[i-1])-1>j:
		x=(-1/div)*(float(tableau[i-2][1])*float(tableau[i-1][j+1])-div*float(tableau[i-2][j+1]))
	else:
		x=0
	return (x)


#-----------------

###################################################
#Fonction : tableauRouth
#But : Fabriquer tout le tableau de Routh
#Preconditions : tableau doit etre une liste de liste comprenant les 2 premieres lignes deja faites
#Paramètre(s) : tableau : liste de liste 
#Postconditions : tableau de routh avec toutes les valeurs 
#Auteur :  Guillaume NOUAL
###################################################
def tableauRouth(tableau):
	degres=int(tableau[0][0].split("^")[-1]) #recupére le degres qui est le n dans p^n du premiere elem du tableau
	for i in range (2,degres+1):
		tableau.append(["p^"+str(degres-i)])
		for j in range (1,len(tableau[0])):
			tableau[i].append(str(elem_i_j(tableau,i,j)))
	return(tableau)


#-----------------

###################################################
#Fonction : affichage_tableau
#But : Permet d'afficher un tableau
#Preconditions : aucunes
#Paramètre(s) : matrice : liste de liste
#Postconditions : Affiche un tableau
#Auteur :  Guillaume NOUAL
###################################################

def print_tableau(tableau):
	for i in range(0,len(tableau)):
		ligne=""
		for k in range(0,len(tableau[i])):
			if k==0:
				ligne = str(tableau[i][k])
			else:
				ligne = ligne+" | "+str(tableau[i][k])
		print(ligne)


#-----------------

###################################################
#Fonction : stabilite_routh
#But : Vérifie le critère de stabilité de routh
#Preconditions : tableau : liste de liste complet
#Paramètre(s) : tableau : liste de liste 
#Postconditions : Renvoie un string 
#Auteur :  Guillaume NOUAL
###################################################

def stabilite_routh(tableauFini):
	stack=0
	for i in range (len(tableauFini)-1):
		if float(tableauFini[i][1])*float(tableauFini[i+1][1])<0:
			return("Le système est instable !")
		if float(tableauFini[i][1])*float(tableauFini[i+1][1])==0:
			stack=stack+1
	if stack>0:
		return("Le système est juste-stable !")
	else:
		return("Le système est stable !")


#-----------------

###################################################
#Fonction : critereRouth
#But : Fonction qui appelle toutes les fonctions pour le critere de routh
#Preconditions : aucune
#Paramètre(s) : aucun
#Postconditions : affichage dans le terminal du tableau et de la stabilité
#Auteur :  Guillaume NOUAL
###################################################

def critereRouth():
	x=demande_polynome_routh()
	tableau=tableauRouth(deux_premieres_lignes(x))
	print_tableau(tableau)
	print(stabilite_routh(tableau))


#-----------------

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FIN ROUTH  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  DEBUT JURY  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


###################################################
#Fonction : polynome
#But : Demande le degré et les coefficients d'un polynome à l'utilisateur
#Preconditions : aucunes
#Paramètre(s) : aucuns
#Postconditions : Renvoie un couple d'un polynome sous forme de liste et un polynome en string
#Auteur :  Guillaume NOUAL
###################################################

def polynome():
	degre=int(input("Quel est le degre de votre polynome ? \n"))
	l=[]
	poly=""
	for i in range(0,degre+1):
		b=str(i)
		valeur=input("Entrer le coefficient de degres "+b+" \n")
		poly=poly+str(valeur)+"*z^"+b+" + "
		l.append(float(valeur))
	liste=poly.split("+")
	del liste[-1]
	liste.reverse()
	liste2=[]
	for k in range(0,len(liste)):
		liste2.append(liste[k])
		liste2.append("+")
	del liste2[-1]
	poly="".join(liste2)
	return (l,poly)


#-----------------

###################################################
#Fonction : demande_polynome_jury
#But : récupérer les coefficients du polynome
#Preconditions : aucunes
#Paramètre(s) : aucuns
#Postconditions : renvoie la list des coefficients du polynome
#Auteur :  Guillaume NOUAL
###################################################

def demande_polynome_jury():
	x=polynome()
	print("Votre polynome est : "+x[1]+"\n")
	liste=x[0]
	return (liste)


#-----------------

###################################################
#Fonction : condition1
#But : vérifier la première condition de jury c'est à dire a_n > |a_0|
#Preconditions : polynome degre initialisés
#Paramètre(s) : 
#	-polynome : liste de int sous la forme polynome
#	-degre : int
#Postconditions : renvoie True si la condition est vérifiée c'est à dire que le système est stable , sinon renvoie False correspondant instable
#Auteur :  Guillaume NOUAL
###################################################

def condition1(polynome,degre):
	return(abs(polynome.coef[0])<polynome.coef[degre])


#-----------------

###################################################
#Fonction : zEgal1
#But : verifier la seconde condition de jury Q(z) >0 pour z=1
#Preconditions : polynome , degre initialisés
#Paramètre(s) : 
#	-polynome : liste de int sous la forme polynome
#Postconditions : renvoie True si la condition est vérifiée c'est à dire que le système est stable sinon renvoie False correspondant instable
#Auteur :  Guillaume NOUAL
###################################################

def zEgal1(polynome):
	return(polynome(1)>0)


#-----------------

###################################################
#Fonction : zEgalMoins1
#But : vérifier que Q(z)>0 pour z=-1 et degré pair // Q(z)<0 pour z=-1 et degré impair
#Preconditions : polynome , degre initialisés
#Paramètre(s) : 
#	-polynome : liste de int sous la forme polynome
#	-degre : int
#Postconditions : renvoie True si la condition est vérifiée c'est à dire que le système est stable sinon renvoie False correspondant instable
#Auteur :  Guillaume NOUAL
###################################################

def zEgalMoins1(polynome,degre):
	if (degre%2) == 0:
		return(polynome(-1)>0)
	else:
		return(polynome(-1)<0)
	

#-----------------

###################################################
#Fonction : tableau
#But : verifier les les sous conditions du critère de jury si n>= 3
#Preconditions : polynome , degre, stable ,n initialisés
#Paramètre(s) : 
#	-polynome : liste de int sous la forme polynome
#	-degre : int
#	-stable :booléen
#	-n : int (position du coefficient de plus haut degré)
#Postconditions : renvoie True si la condition est vérifiée c'est à dire que le système est stable sinon renvoie False correspondant instable
#Auteur :  Guillaume NOUAL
###################################################

def tableau(polynome,degre,stable,n):
	tmp = []
	for i in range(degre+1):
		# on calcule chaque terme des lignes du tableau de jury
		terme = numpy.linalg.det([[polynome.coef[0],polynome.coef[degre-i]],[polynome.coef[degre],polynome.coef[i]]])
		tmp.append(terme)
	tmp.reverse()
	# on verifie la condition |b_n-1|>b_0 pour la ligne
	stable = stable & (tmp[n-1]<tmp[0])
	# on répéte l'opération pour chauqe ligne du tableau
	if n!=2 :
		tableau(Polynomial(tmp),degre,stable,(n-1))
	return(stable)
	

#-----------------

###################################################
#Fonction : CalculDegre
#But : vérifie les différents condition de jury selon le degrès du polynome
#Preconditions : polynome initialisé
#Paramètre(s) :
#	-polynome : liste de int sous la forme polynome
#Postconditions : renvoie un booleen True si le système est stable False sinon
#Auteur :  Guillaume NOUAL
###################################################

def CalculDegre(polynome):
	degre = polynome.degree()
	# on teste la stabilite des trois premières conditions
	stabilite = condition1(polynome,degre) & zEgal1(polynome) & zEgalMoins1(polynome,degre)
	#si polynome de degrés supérieur ou égale à 3 on teste le tableau
	if ((degre >= 3) & stabilite):
		tmp = tableau(polynome,degre,stabilite,degre)
		stabilite = stabilite & tmp
	return(stabilite)

	
#-----------------

###################################################
#Fonction : affichageStabilité
#But : afficher la stabilité du système
#Preconditions : polynome initialisé
#Paramètre(s) : 
#	-polynome : liste de int sous la forme polynome
#Auteur :  Guillaume NOUAL
###################################################

def affichageStabilité(polynome):
	if (CalculDegre(polynome)):
		print("STABLE")
	else:
		justeStable = True
		for racine in (polynome.roots()):
			print("Les racines sont donc : ",racine)
			sol=round(abs(racine),3)
			justeStable = justeStable & (sol == 1.0)
		if justeStable: 
			print("JUSTE STABLE")
		else:
			print("INSTABLE")
	

#-----------------




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FIN JURY  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




#%%%-------------------------------------------------------------------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%
#%%%------------------------------- FIN PARTIE AUTOMATIQUE ------------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%
#%%%-------------------------------------------------------------------------------------%%%



###################################################
#Fonction : transform_lbp
#But : Applique le traitement LBP sur une image en renvoyant l'image initiale et finale
#Preconditions : tab et img initialisés
#Paramètre(s) : 
#	 - tab : liste de liste d'entiers
#	 - img : image chargée
#Postconditions : retourne deux images
#Auteur :  Guillaume NOUAL
###################################################

def transfom_lbp(tab,img):
	
	rendu=apply_LBP(tab,[])
	rendu_arr=np.array(rendu)  # transforme la liste en array
	mi=Image.new("L",((rendu_arr.shape)[1] , (rendu_arr.shape)[0]),123)   #création d'une image de dimension égale à l'image choisie et chacun de ses éléments a une valeur de 123
	mi2arr=np.array(mi)
	for i in range(0,len(mi2arr)-1):
		for j in range(0,len(mi2arr[0])-1):
			mi2arr[i][j]=rendu_arr[i][j]
	final=Image.fromarray(mi2arr) #transforme l'array en image
	img2=Image.fromarray(img)
	img2.show() # permet de visualiser une image 
	final.show()


#-----------------

###################################################
#Fonction : transform_poi
#But : Applique le traitement POI sur une image en renvoyant l'image initiale et finale
#Preconditions : tab , img initialisés et k compris entre 0.05 et 0.15 ou 0.04 valeur optimale
#Paramètre(s) : 
#	 - tab : liste de liste d'entiers
#	 - img : image chargée
#	 - k : float
#Postconditions : retourne deux images
#Auteur :  Guillaume NOUAL
###################################################

def transform_poi(tab,img,k):
	list_modif1=(balayage_second(matrice_Gx_Gy(tab),k))
	l2arr=np.array(list_modif1)

	new_img=Image.new("L",((l2arr.shape)[1] , (l2arr.shape)[0]),123)
	mi2arr=np.array(new_img)
	seuili=seuil(l2arr)
	for i in range(0,len(mi2arr)-2):
		for j in range(0,len(mi2arr[0])-2):
			if l2arr[i][j]>=seuili:
				mi2arr[i][j]=255
			else:
				mi2arr[i][j]=0
	img_final=Image.fromarray(mi2arr)
	img2=Image.fromarray(img)
	img2.show()
	img_final.show()


#-----------------

###################################################
#Fonction : transform_blur
#But : Applique le traitement blur sur une image en renvoyant l'image initiale et finale
#Preconditions : matrice et img initialisés
#Paramètre(s) : 
#	 - matrice : liste de liste d'entiers
#	 - img : image chargée
#Postconditions : retourne deux images
#Auteur :  Guillaume NOUAL
###################################################

def transform_blur(matrice,img):
	rendu = np.array(blur(matrice))
	stock1 = Image.new("L",((rendu.shape)[1] , (rendu.shape)[0]),123)
	mi2arr = np.array(stock1)
	for i in range(0,len(mi2arr)-1):
		for j in range(0,len(mi2arr[0])-1):
			mi2arr[i][j]=rendu[i][j]
	final = Image.fromarray(mi2arr)
	img2=Image.fromarray(img)
	img2.show()
	final.show()


#-----------------

###################################################
#Fonction : transform_deblur
#But : Applique le traitement deblur sur une image en renvoyant l'image initiale et finale
#Preconditions : matrice et img initialisés
#Paramètre(s) : 
#	 - matrice : liste de liste d'entiers
#	 - img : image chargée
#Postconditions : retourne deux images
#Auteur :  Guillaume NOUAL
###################################################

def transform_deblur(matrice,img):
	rendu = np.array(matrice)
	stock1 = Image.new("L",((rendu.shape)[1] , (rendu.shape)[0]),123)
	mi2arr = np.array(stock1)
	for i in range(0,len(mi2arr)-1):
		for j in range(0,len(mi2arr[0])-1):
			mi2arr[i][j]=rendu[i][j]
	final = Image.fromarray(mi2arr)
	img2=Image.fromarray(img)
	img2.show()
	final.show()


#-----------------

###################################################
#Fonction : traitement_start_gris
#But : Transforme une image de nuance grise en une matrice 
#Preconditions : image initialisé
#Paramètre(s) : 
#	 - image : image chargée
#Postconditions : retourne une liste de liste d'entiers
#Auteur :  Guillaume NOUAL
###################################################

def traitement_start_gris(image):
		img = mpimg.imread(image)     
		im2arr=np.array(img)
		return(img,im2arr.tolist())


#-----------------

###################################################
#Fonction : traitement_start_couleur
#But : Transforme une image couleur en une matrice
#Preconditions : image initialisé
#Paramètre(s) : 
#	 - image : image chargée
#Postconditions : retourne une liste de liste d'entiers
#Auteur :  Guillaume NOUAL
###################################################

def traitement_start_couleur(image):
	img = mpimg.imread(image)     
	gray = rgb2gray(img)
	return(img,gray.tolist())


#-----------------

###################################################
#Fonction : menu_line
#But : Lance le programme en utlisant les lignes de commandes intégrées (pour plus d'informations sur comment les utiliser voir le README)
#Preconditions : aucunes
#Paramètre(s) : aucuns
#Postconditions : applique un effet souhaité du traitement de l'image ou un des deux critères ou renvoie une erreur à cause d'une mauvaise utilisation des lignes de commande
#Auteur :  Guillaume NOUAL
###################################################

def menu_line():
	tab_arg=sys.argv
	if len(tab_arg)>1:
		if tab_arg[1]=="traitement" and (len(tab_arg)>2):
			if (len(tab_arg)==5) and(tab_arg[2]=="lbp")and(tab_arg[4]=="gris") :
				img,arr2list=traitement_start_gris(tab_arg[3])
				transfom_lbp(arr2list,img)
			#------------------------------------------
			elif (len(tab_arg)==5) and (tab_arg[2]=="lbp")and(tab_arg[4]=="couleur") :
				img,arr2list=traitement_start_couleur(tab_arg[3])
				transfom_lbp(arr2list,img)
			#------------------------------------------
			elif (len(tab_arg)==6) and (tab_arg[2]=="poi")and(tab_arg[4]=="gris") and ((float(tab_arg[5])>=0.04)and (float(tab_arg[5])<=0.15)):
				img,arr2list=traitement_start_gris(tab_arg[3])
				transform_poi(arr2list,img,float(tab_arg[5]))

			#------------------------------------------
			elif (len(tab_arg)==6) and (tab_arg[2]=="poi")and(tab_arg[4]=="couleur") and ((float(tab_arg[5])>=0.04)and (float(tab_arg[5])<=0.15)):
				img,arr2list=traitement_start_couleur(tab_arg[3])
				transform_poi(arr2list,img,float(tab_arg[5]))

			#------------------------------------------
			elif (len(tab_arg)==5) and(tab_arg[2]=="blur")and(tab_arg[4]=="gris") :
				img,arr2list=traitement_start_gris(tab_arg[3])
				transform_blur(arr2list,img)
				score = critere_comparaison(arr2list,blur(arr2list))
				print("Le score de ressemblance est de :")
				print(score)

			#------------------------------------------
			elif (len(tab_arg)==5) and (tab_arg[2]=="blur")and(tab_arg[4]=="couleur") :
				img,arr2list=traitement_start_couleur(tab_arg[3])
				transform_blur(arr2list,img)
				score = critere_comparaison(arr2list,blur(arr2list))
				print("Le score de ressemblance est de :")
				print(score)
			#------------------------------------------
			elif (len(tab_arg)==5) and (tab_arg[2]=="deblur")and(tab_arg[4]=="gris") :
				img,arr2list=traitement_start_gris(tab_arg[3])
				matrice_change=deblur(blur(arr2list))
				transform_deblur(matrice_change,img)
				score = critere_comparaison(arr2list,matrice_change)
				print("Le score de ressemblance est de :")
				print(score)

			#------------------------------------------
			elif (len(tab_arg)==5) and (tab_arg[2]=="deblur")and(tab_arg[4]=="couleur") :
				img,arr2list=traitement_start_couleur(tab_arg[3])
				matrice_change=deblur(blur(arr2list))
				transform_deblur(matrice_change,img)
				score = critere_comparaison(arr2list,matrice_change)
				print("Le score de ressemblance est de :")
				print(score)

			else:
				print("Veuillez rentrer un bon modèle!")
				
		elif tab_arg[1]=="automatique" and (len(tab_arg)>2):
			if (tab_arg[2]=="routh")and (len(tab_arg)==3):
				critereRouth()

			elif (tab_arg[2]=="jury")and(len(tab_arg)==3):
				affichageStabilité(Polynomial(demande_polynome_jury()))

			else:
				print("Veuillez rentrer un bon modèle!")
		else:
			print("Veuillez rentrer un bon modèle!")

	else:
		print("Veuillez rentrer un bon modèle!")






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                        FONCTION QUI LANCE LE PROGRAMME
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


menu_line()
