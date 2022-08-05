

import numpy as np
import cv2
import torch
import sys

from configuracionRecorte import ajustarRecorte
from configuracionGstreamer import ajustePipeline
from configuracionModelo import ajusteModelo

from utils import preprocess, posprocess
    
#Esta función envuelve el ciclo principal detrás del sistema
def show_camera(model, device, escala):
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
    
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
        		# Captura la imagen
                ret_val, frame = video_capture.read()
                
                #Detecciòn de rostro
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray)
                
                # Si existe un rostro 
                if (len(faces)!=0):

                    # Estas cuatro variables contienen el roi
                    cx,cy,cw,ch=ajustarRecorte(faces[0],imageSizex,imageSizey)
                    #cx,cy,cw,ch=faces[0]
                    
                    try:

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
                      
                    
                        # apaga gradiente y filtra solo los valores entre 0 y 1
                        #predicciòn de las imagenes
                        with torch.no_grad():
                            preds = model(lrPrep).clamp(0.0, 1.0)
                        
                        #Pos-procesamiento    
                        frame=posprocess(preds, ycbcr[..., 1], ycbcr[..., 2]) 
                        
                    except TypeError:
                    
                    	print("ESta muy cerca, alejese")
                
                    

                #mostar imagen frame
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                
                    try:
                        cv2.imshow(window_title, frame)
                        
                    except e:
                        print(e) 

                else:
                    break 
                #finalizar si se ha presionado q o "esc"    
                keyCode = cv2.waitKey(10) & 0xFF 
                if keyCode == 27 or keyCode == ord('q'):
                    break
                    
        #cerrar ventana y capturadora
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error no se puede iniciar la camara")


if __name__ == "__main__":

    # La escala es un argumento usado como entrada
	escala=int (sys.argv[1])

    # Se instancia el modelo dada la instancia
	model, device =ajusteModelo(escala)
	
	#Iniciar camara
	show_camera(model, device, escala)
