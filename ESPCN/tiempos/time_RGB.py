

import numpy as np
import cv2
import torch
import sys
import time

from configuracionRecorte import ajustarRecorte
from configuracionGstreamer import ajustePipeline
from configuracionModelo import ajusteModelo

from utils import  posprocessRGB , preprocessRGB
    

def show_camera(model, device,escala):
    window_title = "Camara en super resolucion"
    
    imageSizex,imageSizey = 1280/escala , 960/escala
  
    pipe=ajustePipeline( capture_width=imageSizex,
    capture_height=imageSizey,
    display_width=imageSizex,
    display_height=imageSizey)
    
    print(pipe)    
    
    #Inicializaciòn
    video_capture = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    face_cascade = cv2.CascadeClassifier('clasificadorFrontal/haarcascade_frontalface_default.xml')
    
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            
    		# Captura la imagen
            ret_val, frame = video_capture.read()
            
            #Detecciòn de rostro
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)
            
            # Si existe un rostro 
            if (len(faces)!=0):
            
                cx,cy,cw,ch=ajustarRecorte(faces[0],imageSizex,imageSizey)
                #cx,cy,cw,ch=faces[0]
                
                try:
            
                    R=frame[...,0][cy: cy+ch, cx: cx+cw]
                    G=frame[...,1][cy: cy+ch, cx: cx+cw]
                    B=frame[...,2][cy: cy+ch, cx: cx+cw]
                    
                    #preprocesamiento y device
                    RPrep=preprocessRGB(R,device)
                    GPrep = preprocessRGB(G, device)
                    BPrep = preprocessRGB(B, device)
                
                    # apaga gradiente y filtra solo los valores entre 0 y 1
                    #predicciòn de las imagenes
                    with torch.no_grad():
                        predsR = model(RPrep).clamp(0.0, 1.0)
                        predsG = model(GPrep).clamp(0.0, 1.0)
                        predsB = model(BPrep).clamp(0.0, 1.0)
                    
                    #Pos-procesamiento    
                    frame=posprocessRGB(predsR,predsG, predsB)
                    
                except TypeError:
                
                	print("ESta muy cerca, alejese")
            
                

            #mostar imagen frame
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
            
                try:
                    cv2.imshow(window_title, frame)
                    
                except e:
                    print(e)
                    
        #cerrar ventana y capturadora
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error no se puede iniciar la camara")


if __name__ == "__main__":

	escala=int (sys.argv[1])
	
	inicio=time.time()
	
	model, device =ajusteModelo(escala)
	
	#Iniciar camara
	show_camera(model, device, escala)
