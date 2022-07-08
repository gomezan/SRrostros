import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, preprocessRGB

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


def graficarImagen(img, window_name: str):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    weight = r"/home/nvidia/Documentos/ESPCN/espcn_x2Full.pth"
    #weight = r"/home/nvidia/Documentos/ESPCN/best0x4.pth"
    imagen = r"test.jpg"
    ruta=r"/mnt/tmp/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default=weight)
    parser.add_argument('--image-file', type=str, default=imagen)
    parser.add_argument('--path', type=str, default=ruta)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    print("Este experimento solo tiene en cuenta la componente Y")
    print("Configurando red")

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

    #working=True
    #while(working):
        
    print("Tomando imagen")

    
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)    
    
    #Tomar imagen
    ret_val , img = video_capture.read()
    
    inicio=time.time()
		
        #obtener tamaño)
    image_width = img.shape[0]
    image_height = img.shape[1]

        #obtener imagen de lanczos a partir de la iamgen escalada
    lanczos =cv2.resize(img, (image_width * args.scale, image_height * args.scale),                interpolation=cv2.INTER_LANCZOS4)

        #obtiene las imagenes normalizadas y el ycbcr de la imagen interpolada por lanczos;  transfiere a device
    lrPrep, _ = preprocess(img, device)
    _, ycbcr = preprocess(lanczos, device)

    pre=time.time()

        #apaga gradiente y filtra valores entre 0 y 1
    with torch.no_grad():
        preds = model(lrPrep).clamp(0.0, 1.0)
        
    pos=time.time()

        #trata la imagen
	        #desnormaliza
	        #quita dimensiones
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)


    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        #Clip y converion RGB
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    #graficarImagen(output,"solo componente Y" )
        
    fin=time.time()
    
    print("El tiempo total de procesamiento de la imagen es: ", str(fin-inicio))
    print("El tiempo de pre-procesamiento de la imagen es: ", str(pre-inicio))
    print("El tiempo de procesamiento de red es : ", str(pos-pre))
    print("El tiempo de pos-procesamiento de la imagen es: ", str(fin-pos))




   
