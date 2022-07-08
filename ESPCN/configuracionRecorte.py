

import cv2
import numpy as np
from math import floor
from math import ceil


def calcularIncremento(size):
    return int(size/16)

#Incrementa en una coordenada una cantidad constante  
def ajusteConstante(ci,ctam,cf,inc):
    Nci= ci-inc
    Ncf=cf+inc
    Nctam=abs(Ncf-Nci)
    
    return Nci,Nctam,Ncf
 
#Ajusta en una coordenada dado un tamaño minimo 
def ajusteTamMinimo(ci,ctam,cf,minTam):

	Nci=ci
	Ncf=cf
	Nctam=ctam
	
	if ctam<minTam:
	
	    diff=minTam-ctam
	    Nci=ci-int(floor(diff/2.0))
	    Ncf=cf+int(ceil(diff/2.0))
	    Nctam=abs(Ncf-Nci)
	    
	return Nci,Nctam,Ncf 
    
    
#Ajusta en funciòn de la relacion w/h  
def ajusteMinimo(cx,cy,cw,ch,cu,cv, sizex, sizey):

	Ncx=cx
	Ncu=cu
	Ncw=cw
	
	Ncy=cy
	Ncv=cv
	Nch=ch
	
	if cw/ch>(sizex/sizey):
	    #print ("ajusto por y")
	    diffy=(cw*sizey/sizex)-ch
	    Ncy=cy-int(floor(diffy/2.0))
	    Ncv=cv+int(ceil(diffy/2.0))
	    Nch=abs(Ncv-Ncy)
	    
	if cw/ch<(sizex/sizey):
	    #print ("ajusto por w")
	    diffx=(ch*sizex/sizey)-cw
	    Ncx=cx-int(floor(diffx/2.0))
	    Ncu=cu+int(ceil(diffx/2.0))
	    Ncw=abs(Ncu-Ncx)
	    
	#print(cw/ch)
	
	return (Ncx,Ncy,Ncw,Nch,Ncu,Ncv)
 
 
#Ajusta en una coordenada dado un tamaño minimo 
def ajustePos(ci,ctam,cf,maxTam):
    
    Nci=ci
    Ncf=cf
    Nctam=ctam
    
    if ci<0:
        diff=-ci
        Nci=ci+diff
        Ncf=cf+diff
    
    if cf>maxTam:
        diff=cf-maxTam
        Nci=ci-diff
        Ncf=cf-diff 
        
    Nctam=abs(Ncf-Nci)
    
    return Nci,Nctam,Ncf 

#Se asegura que los valores de la ventana sean congruentes     
def rectificador(ci,ctam,cf,maxTam):

	Nci=ci
	Ncf=cf
	Nctam=ctam
	
	if ci<0:
	    Nci=0
	    
	if cf>=maxTam:
	    Ncf=maxTam
	
	Nctam=abs(Ncf-Nci)
	
	return Nci,Nctam,Ncf
 
 
def ajustarRecorte(roi,imageSizex,imageSizey):
 
 	#pos iniciales
    cx=roi[0]
    cy=roi[1]
 
 	#tamaños
    cw=roi[2]
    ch=roi[3]
 
 	#pos finales
    cu=cx+cw
    cv=cy+ch
 
 	#MAXIMOS
    maximoy=imageSizey
    maxtamx=imageSizex
 
    #print ("antes0", cx,", ",cu,", ",cy,", ",cv,", ",cw,", ",ch)
  
  
    #Incremento en todas las direcciones
    inc=calcularIncremento(imageSizex)
    #print(inc)
    cy,ch,cv=ajusteConstante(cy,ch,cv,inc)
    
    #print ("antes1", cx,", ",cu,", ",cy,", ",cv,", ",cw,", ",ch)
    
    
    #Incrementa en funciòn del tamaño minimo
    #cx,cw,cu=ajusteTamMinimo(cx,cw,cu,mintamx)
    #cy,ch,cv=ajusteTamMinimo(cy,ch,cv,mintamy)
    
    #print ("antes2", cx,", ",cu,", ",cy,", ",cv,", ",cw,", ",ch)
    
    #Incrementa en funciòn de la relacion w/h
    cx,cy,cw,ch,cu,cv=ajusteMinimo(cx,cy,cw,ch,cu,cv,imageSizex,imageSizey)
    
    #print(cw/ch)
    #print ("durante", cx,", ",cu,", ",cy,", ",cv,", ",cw,", ",ch)
    
    #Verificacion de posiciones
    cx,cw,cu=ajustePos(cx,cw,cu,imageSizex)
    cy,ch,cv=ajustePos(cy,ch,cv,imageSizey)
    
    #print ("solo pos", cx,", ",cu,", ",cy,", ",cv,", ",cw,", ",ch)
    
    #Rectificaciòn
    cx,cw,cu=rectificador(cx,cw,cu,imageSizex)
    cy,ch,cv=rectificador(cy,ch,cv,imageSizey)
    
    #print ("despues", cx,", ",cu,", ",cy,", ",cv,", ",cw,", ",ch)
    return (cx,cy,cw,ch)
 


