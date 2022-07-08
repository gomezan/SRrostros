import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, preprocessRGB


def graficarImagen(img, window_name: str):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=256,
    capture_height=256,
    display_width=512,
    display_height=512,
    framerate=100,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

   

if __name__ == '__main__':
      
    weight = r"/home/nvidia/Documentos/ESPCN/espcn_x2Full.pth"
    #weight = r"/home/nvidia/Documentos/ESPCN/best0x4.pth"
   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default=weight)
    parser.add_argument('--image-file', type=str, default=imagen)
    parser.add_argument('--path', type=str, default=ruta)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    print("Este experimento concibe RGB")
    print("Configurando red")

	inicioInicio=time.time()

    #verificacion GPU
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #creacion de modelo
    model = ESPCN(scale_factor=args.scale).to(device)

    #subida de pesos
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n[7:] in state_dict.keys():
            state_dict[n[7:]].copy_(p)
        else:
            raise KeyError(n)

    #switch evaluacion/ejecucion
    model.eval()

    
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)    
    
    #Tomar imagen
    ret_val , img = video_capture.read()
    
    inicio=time.time()

    R=img[...,0]
    G=img[...,1]
    B=img[...,2]

    #preprocesamiento y device
    RPrep=preprocessRGB(R,device)
    GPrep = preprocessRGB(G, device)
    BPrep = preprocessRGB(B, device)
    
    pre=time.time()
	
	# apaga gradiente y ubica todos los valores entre 0 y 1
    with torch.no_grad():
        predsR = model(RPrep).clamp(0.0, 1.0)
        predsG = model(GPrep).clamp(0.0, 1.0)
        predsB = model(BPrep).clamp(0.0, 1.0)

    pos=time.time()

    predsR = predsR.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    predsG = predsG.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    predsB = predsB.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([predsR, predsG, predsB]).transpose([1, 2, 0]).astype(np.uint8)
    
    fin=time.time()
    print("El tiempo de procesamiento total de imagen es: ", str(fin-inicio))
    print("El tiempo de pre-procesamiento es : ", str(pre-inicio))
    print("El tiempo de procesamiento en la red neuronal es : ", str(pos-pre))
    print("El tiempo de pos-procesamiento es : ", str(fin-pos))
    
    graficarImagen(output, "Compoenentes RGB")


