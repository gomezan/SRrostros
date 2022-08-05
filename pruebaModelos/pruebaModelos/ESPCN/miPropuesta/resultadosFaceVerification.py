

import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.optim as optim

import glob
import numpy as np
import cv2

import face_recognition

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, preprocessRGB


#Script usado para la evaluación de la distancia de rostro 

#Permite mapear el método con el indicativo del mismo método en opencv
interpolacion={
    "lan": cv2.INTER_LANCZOS4,
    "nn": cv2.INTER_NEAREST,
    "bi2": cv2.INTER_LINEAR,
    "bi3": cv2.INTER_CUBIC
}

#Permite mapear el método con el nombre de dicho método
nombres={
    "lan": "lanczos",
    "nn": "NEAREST",
    "bi2": "LINEAR",
    "bi3": "CUBIC",
    "rgb": "red",
    "y": "red"
}


#Permite mapear la escala deseada con la ubicación
pesos={
    2: r"/HDDmedia/supermri/pesosFinales/espcn_x2Full.pth",
    4: r"/HDDmedia/supermri/pesosFinales/best0x4.pth",
    8: r"/HDDmedia/supermri/pesosFinales/espcnx8.pth"
}

#Calula la distancia de rostro en un método y la escala de dicho metodo
def calcularOtras(factor, metodo, correr, y, RGB, mtd, lote):

    #sobre el conjunto de validación se dividen en lotes: 16, 17 , 18 , 19 , 20, 21 
    numLote=16+lote
    #Ubicación ground truth
    gtPath=r"/HDDmedia/supermri/groundTruth/eval/rostroHR"+str(numLote)+"???.png"
    # Ubicación imagenes decimadas
    scPath=r"/HDDmedia/supermri/decimadasX"+str(factor)+"/eval/rostroLR"+str(numLote)+"???.png"
    # Ubicación destino almacenar resultados de la distancia
    pathDest=r"/HDDmedia/supermri/resultados/x"+str(factor)
    
    #Se carga el método, el nombre y los pesos
    tecnica=interpolacion[metodo]
    interpolationName=nombres[metodo]
    weights_file=pesos[factor]

    #verificacion GPU
    cudnn.benchmark = True
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    #creacion de modelo
    model = ESPCN(scale_factor=factor).to(device)
    model = nn.DataParallel(model, device_ids=[2,3])
    model.to(device)    

    #subida de pesos
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    #switch evaluacion/ejecucion
    model.eval()


    #Comparación entre las estartegias:
    gtFiles=sorted(glob.glob(gtPath))
    scFiles=sorted(glob.glob(scPath))

    ntpRes=[]
    redRes=[]

    #empleando solo Y
	#En caso de querer evaluar los métodos de super resolución usando el canal Y
    if (y):

	#Si se desea calcular las métricas
            #Es posible no calcular las metricas y cargar resultados almacenados previamente

        if (correr):

            for i,j in zip(gtFiles, scFiles):
                print(i)
                print(j)
                #obtener imagenes
                gt = cv2.imread(i, 1)
                sc = cv2.imread(j, 1)
                image_width = sc.shape[0]
                image_height = sc.shape[1]
                lan = cv2.resize(sc, (image_width * factor, image_height * factor), interpolation=cv2.INTER_LANCZOS4)
                #procesarImagenes
                scPrep, _ = preprocess(sc, device)
                _, ycbcr = preprocess(lan, device)
                #prediccion del modelo
                with torch.no_grad():
                    preds = model(scPrep).clamp(0.0, 1.0)
                #pos procesamiento
                preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
                output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
                
                # Se almacena la imagen a comparar en super resolución
                cv2.imwrite("comparacion.png",output)
                
                #Se carga el ground truth y la imagen a comparar
                gt_image = face_recognition.load_image_file(i)
                comp_image = face_recognition.load_image_file("comparacion.png")

		#Se codifican sus razgos 
                gt_encoding = face_recognition.face_encodings(gt_image)[0]
                comp_encoding = face_recognition.face_encodings(comp_image)[0]

		#almacenar resultados
                results = face_recognition.face_distance([gt_encoding], comp_encoding)
                redRes.append( results)

	    #convertir a array
            rdRes=np.array(redRes)

	    #Guardar el vector entero de resultados como archivo npy
            np.save(pathDest+r"\redFace_y"+str(lote)+".npy",rdRes)
            
	#cargar resultados
        rdRes=np.load(pathDest+r"\redFace_y"+str(lote)+".npy")
       
	#mostrar resultados
        print("MODELO Y")
        print("*************** Face ***********")
        print("promedio ", rdRes.mean())
        print("std ", np.std(rdRes))
        print("mediana ", np.median(rdRes))
        print("minimo ", np.amin(rdRes))
        print("maximo ", np.amax(rdRes))
        print("***")

    #empleando RGB

    redRes=[]

    # En caso de querer evaluar los métodos de super resolución usando los canales RGB
    if(RGB):

	# Si se desea calcular las métricas
            # Es posible no calcular las metricas y cargar resultados almacenados previamente
        if(correr):

            for i,j in zip(gtFiles, scFiles):
                print(i)
                print(j)
                #obtener imagenes
                gt = cv2.imread(i, 1)
                sc = cv2.imread(j, 1)
                image_width = sc.shape[0]
                image_height = sc.shape[1]

                R = sc[..., 0]
                G = sc[..., 1]
                B = sc[..., 2]
                # preprocesamiento
                RPrep = preprocessRGB(R, device)
                GPrep = preprocessRGB(G, device)
                BPrep = preprocessRGB(B, device)
                # Prediccion
                with torch.no_grad():
                    predsR = model(RPrep).clamp(0.0, 1.0)
                    predsG = model(GPrep).clamp(0.0, 1.0)
                    predsB = model(BPrep).clamp(0.0, 1.0)
                # pos procesamiento
                predsR = predsR.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                predsG = predsG.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                predsB = predsB.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                output = np.array([predsR, predsG, predsB]).transpose([1, 2, 0]).astype(np.uint8)


                # Se almacena la imagen a comparar en super resolución
                cv2.imwrite("comparacion.png",output)
                
                #Se carga el ground truth y la imagen a comparar
                gt_image = face_recognition.load_image_file(i)
                comp_image = face_recognition.load_image_file("comparacion.png")

                #Se codifican sus razgos 
                gt_encoding = face_recognition.face_encodings(gt_image)[0]
                comp_encoding = face_recognition.face_encodings(comp_image)[0]


		#almacenar resultados
                results = face_recognition.face_distance([gt_encoding], comp_encoding)
                redRes.append( results)
                
                
               

	    #convertir array
            rdRes=np.array(redRes)
           
		# Guardar el vector entero de resultados como archivo npy
            np.save(pathDest+r"\redFace_RGB"+str(lote)+".npy",rdRes)
          
        # Cargar resultados  
        rdRes=np.load(pathDest+r"\redFace_RGB"+str(lote)+".npy")

	# Mostrar resultados
        print("MODELO RGB ")
        print("************** Face **************")
        print("promedio ", rdRes.mean())
        print("std ", np.std(rdRes))
        print("mediana ", np.median(rdRes))
        print("minimo ", np.amin(rdRes))
        print("maximo ", np.amax(rdRes))
        print("***")

    ntpRes=[]

    #empleando metodos de interpolacion
    # En caso de querer evaluar los métodos de interpolación
    if (mtd):


	# Si se desea calcular las métricas
        # Es posible no calcular las metricas y cargar resultados almacenados previamente
        if (correr):

            for i,j in zip(gtFiles, scFiles):
                print(i)
                print(j)
                #obtener imagenes
                gt = cv2.imread(i, 1)
                sc = cv2.imread(j, 1)
                image_width = sc.shape[0]
                image_height = sc.shape[1]
                ntp = cv2.resize(sc, (image_width * factor, image_height * factor), interpolation=tecnica)

             
                
                # Se almacena la imagen a comparar interpolada
                cv2.imwrite("comparacion.png",ntp)
                
                #Se carga el ground truth y la imagen a comparar
                gt_image = face_recognition.load_image_file(i)
                comp_image = face_recognition.load_image_file("comparacion.png")
                
		#Se codifican sus razgos 
                gt_encoding = face_recognition.face_encodings(gt_image)[0]
                comp_encoding = face_recognition.face_encodings(comp_image)[0]

		#almacenar resultados
                results = face_recognition.face_distance([gt_encoding], comp_encoding)
                ntpRes.append( results)


	    # Guardar el vector entero de resultados como archivo npy
            np.save(pathDest +r"\I"+interpolationName+"Face"+str(lote)+".npy", ntpRes)

	# Cargar resultados
        ntpRes=np.load(pathDest +r"\I"+interpolationName+"Face"+str(lote)+".npy")
      

	# Cargar resultados
        print(interpolationName)
        print("*************** Face***************")
        print("promedio ", ntpRes.mean())
        print("std ", np.std(ntpRes))
        print("mediana ", np.median(ntpRes))
        print("minimo ", np.amin(ntpRes))
        print("maximo ", np.amax(ntpRes))
        print("***")


	

a=5
calcularOtras(factor=2, metodo="nn", correr= True, y= True, RGB= True, mtd= False,lote=a)
calcularOtras(factor=2, metodo="nn", correr= True, y= False, RGB= False, mtd= True,lote=a)
calcularOtras(factor=2, metodo="bi2", correr= True, y= False, RGB= False, mtd= True,lote=a)
calcularOtras(factor=2, metodo="bi3", correr= True, y= False, RGB= False, mtd= True,lote=a)
calcularOtras(factor=2, metodo="lan", correr= True, y= False, RGB= False, mtd= True,lote=a)

calcularOtras(factor=4, metodo="nn", correr= True, y= False, RGB= False, mtd= True,lote=a)
calcularOtras(factor=4, metodo="nn", correr= True, y= True, RGB= True, mtd= False,lote=a)
calcularOtras(factor=4, metodo="bi2", correr= True, y= False, RGB= False, mtd= True,lote=a)
calcularOtras(factor=4, metodo="bi3", correr= True, y= False, RGB= False, mtd= True,lote=a)
calcularOtras(factor=4, metodo="lan", correr= True, y= False, RGB= False, mtd= True,lote=a)

calcularOtras(factor=8, metodo="nn", correr= True, y= True, RGB= True, mtd= False,lote=a)
calcularOtras(factor=8, metodo="nn", correr= True, y= False, RGB= False, mtd= True,lote=a)
calcularOtras(factor=8, metodo="bi2", correr= True, y= False, RGB= False, mtd= True,lote=a)
calcularOtras(factor=8, metodo="bi3", correr= True, y= False, RGB= False, mtd= True,lote=a)
calcularOtras(factor=8, metodo="lan", correr= True, y= False, RGB= False, mtd= True,lote=a)
