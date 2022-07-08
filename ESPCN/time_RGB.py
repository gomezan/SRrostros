

import numpy as np
import cv2
import torch
import sys
import time

from configuracionRecorte import ajustarRecorte
from configuracionGstreamer import ajustePipeline
from configuracionModelo import ajusteModelo

from utils import  posprocessRGB , preprocessRGB
    


if __name__ == "__main__":

	escala=int (sys.argv[1])

	inicio=time.time()

	model, device =ajusteModelo(escala)

	window_title = "Camara en super resolucio"

	imageSizex,imageSizey = 1280/escala , 960/escala

	pipe=ajustePipeline( capture_width=imageSizex,
	capture_height=imageSizey,
	display_width=imageSizex,
	display_height=imageSizey)

	video_capture = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
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

	# Si existe un rostro 



	inicioRoi=time.time()
	cx,cy,cw,ch=ajustarRecorte(faces[0],imageSizex,imageSizey)




	inicioPre=inicioFin=time.time()
	R=frame[...,0][cy: cy+ch, cx: cx+cw]
	G=frame[...,1][cy: cy+ch, cx: cx+cw]
	B=frame[...,2][cy: cy+ch, cx: cx+cw]

	#preprocesamiento y device
	RPrep=preprocessRGB(R,device)
	GPrep = preprocessRGB(G, device)
	BPrep = preprocessRGB(B, device)

	inicioModelo=time.time()

	# apaga gradiente y filtra solo los valores entre 0 y 1
	#predicciòn de las imagenes
	with torch.no_grad():
		predsR = model(RPrep).clamp(0.0, 1.0)
		predsG = model(GPrep).clamp(0.0, 1.0)
		predsB = model(BPrep).clamp(0.0, 1.0)
		
	inicioPos=inicioFin=time.time()    

	#Pos-procesamiento    
	frame=posprocessRGB(predsR,predsG, predsB)
		

	print("ESta muy cerca, alejese")	

	inicioVisualizacion=time.time()    



	cv2.imshow(window_title, frame)
	finVisualizacion=time.time()	
		  
		        
	#cerrar ventana y capturadora

	video_capture.release()
	cv2.destroyAllWindows()
	
	print("Tempo de inicializaciòn ",inicioFin -inicio)
	print("Tempo de captura ", inicioDeteccion-inicioCaptura)
	print("Tempo de deteccion ", finDeteccion -inicioDeteccion)
	print("Tempo de roi ", inicioPre -inicioRoi)
	print("Tempo de pre ",inicioModelo -inicioPre )
	print("Tempo de modelo ",inicioPos-inicioModelo)
	print("Tempo de pos ",inicioVisualizacion- inicioPos)
	print("Tempo de visual ",finVisualizacion-inicioVisualizacion)
	


