
import glob
import cv2
import os

import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import randint

from scipy.linalg import sqrtm
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

import torch
import torch.backends.cudnn as cudnn

from pruebaModelos.ESPCN.models import ESPCN
from pruebaModelos.ESPCN.utils import convert_ycbcr_to_rgb, preprocess, preprocessRGB

#Este script busca calcular el fid de grupos de imágenes

#Calcula el fid con base en los vectores de caracteristicas de ambos grupos de imagenes
def calculate_fid(act1, act2):

    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2 )**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


#Calcula la descomposición de caracteristicas de un grupo de imagenes y las almacena como npy
def calularActivacion(model, path, nombre):
    act = []
    files = glob.glob(path)
    for i in files:
        print(i)
        img=np.expand_dims(cv2.imread(i, 1), axis=0)
        act.append(np.reshape(model.predict(img),(2048)))
    res=np.array(act)
    np.save(r"C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion"+"\IMG"+nombre,res)

#Calcula la activación de imágenes interpoladas dado método de interpolación
def calularActivacionMetodo(model, path, nombre, tecnica, factor):
    act = []
    files = glob.glob(path)
    for i in files:
        print(i)
        img=cv2.imread(i, 1)
        image_width = img.shape[0]
        image_height = img.shape[1]
        img=cv2.resize(img, (image_width * factor, image_height * factor), interpolation=tecnica)
        img=np.expand_dims(img, axis=0)
        act.append(np.reshape(model.predict(img),(2048)))
    res=np.array(act)
    np.save(r"C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion"+"\IMG"+nombre,res)


#Calcula la activación de imágenes en super resolución empleando la ESPCN en RGB
def calularActivacionModeloRGB(model, path, nombre, factor, weights_file):
    #lista para almacenar los resultados
    act = []
    files = glob.glob(path)

    # verificacion GPU
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # creacion de modelo
    red = ESPCN(scale_factor=factor).to(device)

    # subida de pesos
    state_dict = red.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n[7:] in state_dict.keys():
            state_dict[n[7:]].copy_(p)
        else:
            raise KeyError(n)

    # switch evaluacion/ejecucion
    red.eval()

    for i in files:
        print(i)
        img=cv2.imread(i, 1)

        R = img[..., 0]
        G = img[..., 1]
        B = img[..., 2]
        # preprocesamiento
        RPrep = preprocessRGB(R, device)
        GPrep = preprocessRGB(G, device)
        BPrep = preprocessRGB(B, device)
        # Prediccion
        with torch.no_grad():
            predsR = red(RPrep).clamp(0.0, 1.0)
            predsG = red(GPrep).clamp(0.0, 1.0)
            predsB = red(BPrep).clamp(0.0, 1.0)
        # pos procesamiento
        predsR = predsR.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        predsG = predsG.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        predsB = predsB.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array([predsR, predsG, predsB]).transpose([1, 2, 0]).astype(np.uint8)


        img=np.expand_dims(output, axis=0)
        act.append(np.reshape(model.predict(img),(2048)))
    res=np.array(act)
    np.save(r"C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion"+"\IMG"+nombre,res)


#Calcula la activación de imágenes en super resolución empleando la ESPCN en Y
def calularActivacionModeloY(model, path, nombre, factor, weights_file):
    # lista para almacenar los resultados
    act = []
    files = glob.glob(path)

    # verificacion GPU
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # creacion de modelo
    red = ESPCN(scale_factor=factor).to(device)

    # subida de pesos
    state_dict = red.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n[7:] in state_dict.keys():
            state_dict[n[7:]].copy_(p)
        else:
            raise KeyError(n)

    # switch evaluacion/ejecucion
    red.eval()

    for i in files:
        print(i)
        img=cv2.imread(i, 1)
        image_width = img.shape[0]
        image_height = img.shape[1]

        lan = cv2.resize(img, (image_width * factor, image_height * factor), interpolation=cv2.INTER_LANCZOS4)
        # procesarImagenes
        scPrep, _ = preprocess(img, device)
        _, ycbcr = preprocess(lan, device)
        # prediccion del modelo
        with torch.no_grad():
            preds = red(scPrep).clamp(0.0, 1.0)
        # pos procesamiento
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

        img=np.expand_dims(output, axis=0)
        act.append(np.reshape(model.predict(img),(2048)))
    res=np.array(act)
    np.save(r"C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion"+"\IMG"+nombre,res)

#Permite mapear tecnica con el nombre de dicha tecnica
nombres={
    "lan": "lanczos",
    "nn": "NEAREST",
    "bi2": "LINEAR",
    "bi3": "CUBIC",
    "rgb": "RGB",
    "y": "Y"
}

#Permite mapear tecnica con el string de opencv que describe dicha técnica
interpolacion={
    "lan": cv2.INTER_LANCZOS4,
    "nn": cv2.INTER_NEAREST,
    "bi2": cv2.INTER_LINEAR,
    "bi3": cv2.INTER_CUBIC,
}

#Permite mapear tecnica con el método de calculo de activación de dicha técnica
proImagen={
    "rgb": calularActivacionModeloRGB,
    "y": calularActivacionModeloY
}

#Permite mapear la escala con laubicación de los pesos de dicha escala/factor
pesos={
    2: r"C:\Users\Estudiante\Documents\dataset\prueba\espcn_x2Full.pth",
    4: r"C:\Users\Estudiante\Documents\dataset\prueba\best0x4.pth",
    8: r"C:\Users\Estudiante\Documents\dataset\prueba\espcnx8.pth"
}

#Clase encargada de calcular FID
class CalculadorFid:

    #Calula fid
    def calcularFid(self, red,mtd,escala):

        #Obtengo nombre del método de interpolación
        interpolationName=nombres[mtd]
        #Si es una tecnica de super resoluciuón
        if(red):
            #Carga del método deacuerdo si es RGB y Y
            procedimiento=proImagen[mtd]
            weight = pesos[escala]
        # Si es una tecnica de interpolación
        else:
            #Carga del indicativo de opencv del método de interpolación
            tcn = interpolacion[mtd]

        print("Inicio")

        # CrearModelo
        model = InceptionV3(include_top=False, pooling='avg', input_shape=(1024 ,1024 ,3))
        print("Modelo creado")

        # Caluclo de vectores de caracteristicas del ground truth
            #Si no existe ... debe ser calculado
        if not (os.path.exists(r'C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion\IMGgt.npy')):
            calularActivacion(model, r"C:\Users\Estudiante\Documents\dataset\groundTruth\eval\*.png", "gt.npy")

        #carga del vector de caracteristicas del ground truth
        act1=np.load(r'C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion\IMGgt.npy')
        print("Ground truth cargado")

        #Si es una técnica de interpolación
        if not(red):

            # Si no existe ... debe ser calculado
            if not (os.path.exists(r'C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion\IMG'+interpolationName+str(escala)+'.npy')):
                print("no existe activacion " + interpolationName)
                calularActivacionMetodo(model, r"C:\Users\Estudiante\Documents\dataset\decimadasX"+str(escala)+"\eval\*.png", interpolationName+str(escala)+".npy", tcn, escala)

            # carga del vector de caracteristicas del método
            act2=np.load(r'C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion\IMG'+interpolationName+str(escala)+'.npy')
            print("metodo cargado")

        # Si es una técnica de super resolución
        else:
            # Si no existe ... debe ser calculado
            if not (os.path.exists(
                    r'C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion\IMG' + interpolationName + str(
                            escala) + '.npy')):
                print("no existe activacion " + interpolationName)
                procedimiento(model, r"C:\Users\Estudiante\Documents\dataset\decimadasX"+str(escala)+"\eval\*.png",
                                        interpolationName + str(escala) + ".npy", escala, weight)

            # carga del vector de caracteristicas del método
            act2 = np.load(r'C:\Users\Estudiante\Documents\dataset\resultados\FID\activacion\IMG' + interpolationName + str(
                escala) + '.npy')
            print("metodo cargado")


        #calculo del FID
        print("Iniciando calculo")
        fid = calculate_fid( act1, act2)
        print('FID (different): %.3f' % fid)

worker=CalculadorFid()
worker.calcularFid(red=False,mtd="lan",escala=8)


