

import numpy as np
import cv2
import torch
import sys
import time

from configuracionRecorte import ajustarRecorte
from configuracionGstreamer import ajustePipeline
from configuracionModelo import ajusteModelo

from utils import preprocess, posprocess
    


if __name__ == "__main__":

	# La escala es un argumento usado como entrada
	escala=int (sys.argv[1])
	
	inicio=time.time()

	# Se instancia el modelo dada la instancia
	model, device =ajusteModelo(escala)

	window_title = "Camara en super resolucion"

	# Se toma el tamaño del estandar y se divide por la escala
	imageSizex,imageSizey = 1280/escala , 960/escala

	# Se configura el pipeline que controla la comunicación de la camara fisica
	pipe=ajustePipeline( capture_width=imageSizex,
	capture_height=imageSizey,
	display_width=imageSizex,
	display_height=imageSizey)

	print(pipe)    

	#Inicializaciòn
	# instanciación del Pipeline
	video_capture = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
	# instanciación clasificador de detección de rostro
	face_cascade = cv2.CascadeClassifier('clasificadorFrontal/haarcascade_frontalface_default.xml')
	
	inicioFin=time.time()
	   
	window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

	inicioCaptura=time.time()
		
	# Captura la imagen
	ret_val, frame = video_capture.read()
	
	inicioDeteccion=time.time()

	#Detecciòn de rostro
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)
	
	finDeteccion=time.time()
	
	inicioRoi=time.time()

	# Estas cuatro variables contienen el roi
	cx,cy,cw,ch=ajustarRecorte(faces[0],imageSizex,imageSizey)
		
	inicioPre=inicioFin=time.time()
	# Descomposiicón RGB del roi
	R=frame[...,0][cy: cy+ch, cx: cx+cw]
	G=frame[...,1][cy: cy+ch, cx: cx+cw]
	B=frame[...,2][cy: cy+ch, cx: cx+cw]
	img= np.array([R,G, B]).transpose([1, 2, 0])


	image_width = img.shape[0]
	image_height = img.shape[1]

	#obtener imagen de lanczos a partir del recorte
	lanczos =cv2.resize(img, (image_height * escala, image_width *escala),interpolation=cv2.INTER_LANCZOS4)

	#preprocesamiento y device
	lrPrep, _ = preprocess(img, device)
	_, ycbcr = preprocess(lanczos, device)

	inicioModelo=time.time()
	# apaga gradiente y filtra solo los valores entre 0 y 1
	#predicciòn de las imagenes
	with torch.no_grad():
		preds = model(lrPrep).clamp(0.0, 1.0)
		    
    #Pos-procesamiento
	inicioPos=inicioFin=time.time()    
	frame=posprocess(preds, ycbcr[..., 1], ycbcr[..., 2]) 
		    
	inicioVisualizacion=time.time()  
	cv2.imshow(window_title, frame)
	finVisualizacion=time.time()
	
	print("Tempo de inicializaciòn ",inicioFin -inicio)
	print("Tempo de captura ", inicioDeteccion-inicioCaptura)
	print("Tempo de deteccion ", finDeteccion -inicioDeteccion)
	print("Tempo de roi ", inicioPre -inicioRoi)
	print("Tempo de pre ",inicioModelo -inicioPre )
	print("Tempo de modelo ",inicioPos-inicioModelo)
	print("Tempo de pos ",inicioVisualizacion- inicioPos)
	print("Tempo de visual ",finVisualizacion-inicioVisualizacion)
