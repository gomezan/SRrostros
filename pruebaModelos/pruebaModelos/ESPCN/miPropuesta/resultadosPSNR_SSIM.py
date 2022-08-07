

import torch
import torch.backends.cudnn as cudnn
import glob
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from pruebaModelos.ESPCN.models import ESPCN
from pruebaModelos.ESPCN.utils import convert_ycbcr_to_rgb, preprocess, preprocessRGB


#Este script permite calcular y almacenar los resultados del psnr y ssim

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
    "bi3": "CUBIC"
}

#Parametros de entrada
factor=8
#Metodo de interpolación a evaluar
metodo="bi2"
#ubicación archivo de pesos de ESPCN
weights_file=r"C:\Users\Estudiante\Documents\dataset\prueba\espcnx8.pth"
#bool que indica si se desea calular la métrica desde 0
correr= False
#bool que indica si se desean cargar los resultados de super resolución sobre canal y
y= True
#bool que indica si se desean cargar los resultados de super resolución sobre canales RGB
RGB= True
#bool que indica si se desean cargar los resultados de interpolación del método descrito anteriormente
mtd= False


#Ubicación del ground truth
gtPath=r"C:\Users\Estudiante\Documents\dataset\groundTruth\eval\*.png"
#Ubicación de imagenes decimadas
scPath=r"C:\Users\Estudiante\Documents\dataset\decimadasX"+str(factor)+"\eval\*.png"
#Ubicación de destino donde almacenar los resultados
pathDest=r"C:\Users\Estudiante\Documents\dataset\resultados\performance\x"+str(factor)+"\eval"

#Carga de método de interpolación y nombre del mismo
tecnica=interpolacion[metodo]
interpolationName=nombres[metodo]

#verificacion GPU
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#creacion de modelo
model = ESPCN(scale_factor=factor).to(device)

#subida de pesos
state_dict = model.state_dict()
for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
    if n[7:] in state_dict.keys():
        state_dict[n[7:]].copy_(p)
    else:
        raise KeyError(n)

#switch evaluacion/ejecucion
model.eval()


#Comparación entre las estartegias:
gtFiles=sorted(glob.glob(gtPath))
scFiles=sorted(glob.glob(scPath))

ntpPSNR=[]
redPSNR=[]
ntpSSIM=[]
redSSIM=[]


#empleando solo Y
    #En caso de querer evaluar los métodos de super resolución usando el canal Y

if (y):

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
            #almacenar resultados
            redPSNR.append(psnr(gt, output))
            redSSIM.append(ssim(im1=gt, im2=output, channel_axis=2))

        # convertir a array
        rdPSNR=np.array(redPSNR)
        rdSSIM=np.array(redSSIM)

        # Guardar el vector entero de resultados como archivo npy
        np.save(pathDest+r"\redPSNR_y.npy",rdPSNR)
        np.save(pathDest+r"\redSSIM_y.npy", rdSSIM)

    # cargar resultados
    rdPSNR=np.load(pathDest+r"\redPSNR_y.npy")
    rdSSIM=np.load(pathDest+r"\redSSIM_y.npy")

    #mostrar resultados
    print("MODELO Y")
    print("*************** PSNR *************** SSIM")
    print("promedio ", rdPSNR.mean(), "  ", rdSSIM.mean())
    print("std ", np.std(rdPSNR), "  ", np.std(rdSSIM))
    print("mediana ", np.median(rdPSNR), "  ", np.median(rdSSIM))
    print("minimo ", np.amin(rdPSNR), "  ", np.amin(rdSSIM))
    print("maximo ", np.amax(rdPSNR), "  ", np.amax(rdSSIM))
    print("***")

#empleando RGB

redPSNR=[]
redSSIM=[]

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

            # almacenar resultados
            redPSNR.append(psnr(gt, output))
            redSSIM.append(ssim(im1=gt, im2=output, channel_axis=2))

        # convetir resultados a array
        rdPSNR=np.array(redPSNR)
        rdSSIM=np.array(redSSIM)

        # Guardar el vector entero de resultados como archivo npy
        np.save(pathDest+r"\redPSNR_RGB.npy",rdPSNR)
        np.save(pathDest+r"\redSSIM_RGB.npy", rdSSIM)

    # Cargar resultados
    rdPSNR=np.load(pathDest+r"\redPSNR_RGB.npy")
    rdSSIM=np.load(pathDest+r"\redSSIM_RGB.npy")

    # mostrar resultados
    print("MODELO RGB ")
    print("************** PSNR **************** SSIM")
    print("promedio ", rdPSNR.mean(), "  ", rdSSIM.mean())
    print("std ", np.std(rdPSNR), "  ", np.std(rdSSIM))
    print("mediana ", np.median(rdPSNR), "  ", np.median(rdSSIM))
    print("minimo ", np.amin(rdPSNR), "  ", np.amin(rdSSIM))
    print("maximo ", np.amax(rdPSNR), "  ", np.amax(rdSSIM))
    print("***")

ntpPSNR=[]
ntpSSIM=[]


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
            #procesarImagenes
            _, ycbcr = preprocess(ntp, device)

            #almacenar resultados
            ntpPSNR.append(psnr(gt, ntp))
            ntpSSIM.append(ssim(im1=gt, im2=ntp, channel_axis=2))

        # convertir array
        ntpPSNR=np.array(ntpPSNR)
        ntpSSIM=np.array(ntpSSIM)

        # Guardar el vector entero de resultados como archivo npy
        np.save(pathDest +r"\I"+interpolationName+"PSNR.npy", ntpPSNR)
        np.save(pathDest +r"\I"+interpolationName+"SSIM.npy", ntpSSIM)

    # Cargar resultados
    ntpPSNR=np.load(pathDest +r"\I"+interpolationName+"PSNR.npy")
    ntpSSIM=np.load(pathDest +r"\I"+interpolationName+"SSIM.npy")

    # Cargar resultados
    print(interpolationName)
    print("*************** PSNR *************** SSIM")
    print("promedio ", ntpPSNR.mean(), "  ", ntpSSIM.mean())
    print("std ", np.std(ntpPSNR), "  ", np.std(ntpSSIM))
    print("mediana ", np.median(ntpPSNR), "  ", np.median(ntpSSIM))
    print("minimo ", np.amin(ntpPSNR), "  ", np.amin(ntpSSIM))
    print("maximo ", np.amax(ntpPSNR), "  ", np.amax(ntpSSIM))
    print("***")
