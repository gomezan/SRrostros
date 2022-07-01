

import requests
import os
import cv2

import numpy as np

import tensorflow as tf
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from io import BytesIO

def descargarImagenesPillow(inicio: int, fin: int, link: str):
    for i in range (inicio,fin):
        name = 'rostroHR'+ str(i)
        with open(name.replace(' ', '-').replace('/', '') + '.jfif', 'wb') as f:
            im = requests.get(link)
            f.write(im.content)
            print('Writing: ', name)

def descargarImagenPillow(nombre: str, link: str):
    with open(nombre.replace(' ', '-').replace('/', '') + '.png', 'wb') as f:
        im = requests.get(link)
        f.write(im.content)
        print('Writing: ', nombre)

def descargarImagenesListaPillow(lista: [], link: str, nombre: str):
    for i in lista:
        name = nombre+ str(i)
        with open(name.replace(' ', '-').replace('/', '') + '.jfif', 'wb') as f:
            im = requests.get(link)
            f.write(im.content)
            print('Writing: ', name)

def getImagen(link: str):
    im = requests.get(link)
    bgr=np.array(Image.open(BytesIO(im.content)))
    return cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)

def descargarImagen(nombre: str, link: str, url: str):
    img=getImagen(link)
    name=nombre.replace(' ', '-').replace('/', '') + '.png'
    cv2.imwrite(url+'/'+name, img)
    print('Writing: ', nombre)

def descargarImagenes(inicio: int, fin: int, link: str, url :str):
    for i in range(inicio, fin):
        name = 'rostroHR' + str(i)
        descargarImagen(name,link,url)

def descargarImagenesLista(lista: [], link: str, nombre: str, url :str):
    for i in lista:
        name = nombre + str(i)
        descargarImagen(name, link, url)

def graficarImagen(img, window_name: str):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def procesarImagen(img):
    return cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

def cargarImagen(url: str):
    img = cv2.imread(url, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def tensorToNumpy(img):
    graph = img.numpy().astype(dtype=np.uint8)
    return cv2.cvtColor(graph, cv2.COLOR_RGB2BGR)

def pillowToNumpy(img):
    almohadaNp = np.array(img)
    return cv2.cvtColor(almohadaNp, cv2.COLOR_RGB2BGR)

def generadorDecimador(ruta:str, indicativo : int, pos: str):
    direccion=ruta+str(indicativo)+".png"
    rutaCompleta=pos+"/"+direccion

    # imagen opencv
    img=cargarImagen(rutaCompleta)
    cv = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    # imagen tensor flow
    image_open = open(rutaCompleta, 'rb')
    read_image = image_open.read()
    image_decode = tf.image.decode_jpeg(read_image)
    tensor = tf.image.resize(image_decode, (512, 512), method='area')
    # imagen pillow
    with Image.open(rutaCompleta) as im:
        almohada = im.resize((512, 512), Image.BOX)

    return cv,tensor,almohada

def generadorExpansivas(cv,tensor, almohada):
    pruebaCV2 = cv2.resize(cv, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    pruebaTensor = tf.image.resize(tensor, (1024, 1024), method='nearest')
    pruebaPillow = almohada.resize((1024, 1024), Image.NEAREST)
    return pruebaCV2, pruebaTensor, pruebaPillow

def calculadorSSIM(pruebaCV2, pruebaTensor, pruebaPillow, img):
    ssimCv2 = ssim(im1=img, im2=pruebaCV2, channel_axis=2)
    ssimTensor = ssim(im1=img, im2=tensorToNumpy(pruebaTensor), channel_axis=2)
    ssimPillow = ssim(im1=img, im2=pillowToNumpy(pruebaPillow), channel_axis=2)
    return ssimCv2,ssimTensor,ssimPillow

def calculadorPSNR(pruebaCV2, pruebaTensor, pruebaPillow, img):
    psnrCv2 = psnr(img, pruebaCV2)
    psnrTensor = psnr(img, tensorToNumpy(pruebaTensor))
    psnrPillow = psnr(img, pillowToNumpy(pruebaPillow))
    return psnrCv2,psnrTensor,psnrPillow

def cargarImagenIndice(nombre: str, indicativo: int, url: str):
    direccion = nombre + str(indicativo) + ".png"
    rutaCompleta = url + "/" + direccion
    img = cv2.imread(rutaCompleta, 1)
    return img

def pruebas(inicio:int, fin:int, url: str, nombre:str):
    cvListPSNR = []
    tensorListPSNR = []
    pillowListPSNR = []

    cvListSSIM = []
    tensorListSSIM = []
    pillowListSSIM = []

    for i in range(inicio, fin):
        # imagen original
        imgOriginal = cargarImagenIndice(nombre, i, url)
        # imágenes decimadas
        cv, tensor, almohada = generadorDecimador(nombre, i, url)
        # Imagenes expandidas
        pruebaCV2, pruebaTensor, pruebaPillow = generadorExpansivas(cv, tensor, almohada)
        # Prueba sobre las metricas
        ssimCv2, ssimTensor, ssimPillow = calculadorSSIM(pruebaCV2, pruebaTensor, pruebaPillow, imgOriginal)
        psnrCv2, psnrTensor, psnrPillow = calculadorPSNR(pruebaCV2, pruebaTensor, pruebaPillow, imgOriginal)

        cvListPSNR.append(psnrCv2)
        tensorListPSNR.append(psnrTensor)
        pillowListPSNR.append(psnrPillow)

        cvListSSIM.append(ssimCv2)
        tensorListSSIM.append(ssimTensor)
        pillowListSSIM.append(ssimPillow)


        print("El SSIM de opencv es ", ssimCv2)
        print("El SSIM de tensorflow es ", ssimTensor)
        print("El SSIM de pillow es ", ssimPillow)
        print("El psnr de opencv es ", psnrCv2)
        print("El psnr de tensorflow es ", psnrTensor)
        print("El psnr de pillow es ", psnrPillow)
                

    cvarrSSIM=np.array(cvListSSIM)
    tsarrSSIM=np.array(tensorListSSIM)
    pllarrSSIM=np.array(pillowListSSIM)
    cvarrPSNR=np.array(cvListPSNR)
    tsarrPSNR=np.array(tensorListPSNR)
    pllarrPSNR=np.array(pillowListPSNR)

    np.save("opencvPSNR.npy", cvarrPSNR)
    np.save("tensorPSNR.npy", tsarrPSNR)
    np.save("pillowPSNR.npy", pllarrPSNR)
    np.save("opencvSSIM.npy",  cvarrSSIM)
    np.save("tensorSSIM.npy", tsarrSSIM)
    np.save("pillowSSIM.npy", pllarrSSIM)

    print("El SSIM de opencv es ", np.mean(cvarrSSIM))
    print("El SSIM de tensorflow es ", np.mean(tsarrSSIM))
    print("El SSIM de pillow es ", np.mean(pllarrSSIM))
    print("El psnr de opencv es ", np.mean(cvarrPSNR))
    print("El psnr de tensorflow es ", np.mean(tsarrPSNR))
    print("El psnr de pillow es ", np.mean(pllarrPSNR))

def comparadorImagenes(im1, im2):
    comparacion = np.equal(im1, im2)
    return comparacion .all()

def buscadorRepetidos(inicio: int, fin: int, nombre: str, ruta:str ):
    listaRepetidos = []
    i=inicio
    while (i < fin):
        img1 = cargarImagenIndice(nombre, i, ruta)
        img2 = cargarImagenIndice(nombre, i + 1, ruta)

        if (comparadorImagenes(img1, img2)):
            listaRepetidos.append(i + 1)
            i = i + 1

        i = i + 1

    return listaRepetidos

def reemplazarDuplicados(inicio: int, fin: int, link: str, nombre:str, ruta:str):
    repetidos = buscadorRepetidos(inicio, fin, nombre, ruta)
    descargarImagenesLista(repetidos, link, nombre, ruta)

def almacenarImagen(img, nombre: str, pos: int, url: str):
    cv2.imwrite(url + '/'+nombre+str(pos)+'.png', img)

def decimar(inicio: int, fin: int, rutaO:str, rutaD:str):
    for i in range (inicio, fin):
        img=cargarImagenIndice("rostroHR",i,rutaO)
        img=procesarImagen(img)
        almacenarImagen(img,"rostroLR",i,rutaD)
        print("decimate rostroLR",i)

# Función sobre sitio https://thispersondoesnotexist.com
if __name__ == '__main__':
    link = 'https://thispersondoesnotexist.com/image'
    #os.chdir(os.path.join(os.getcwd(), 'imagenesHR'))
    rutaHR = r"C:\Users\Estudiante\Documents\dataset\groundTruth\eval"
    rutaLR = r"C:\Users\Estudiante\Documents\dataset\decimadasX8\eval"

    inicio=16500
    fin=22000
    cargar=True
    escala=8

    #pruebas(inicio, fin, rutaHR ,"rostroHR")
    #descargarImagenes(inicio,fin,link,rutaHR)
    #reemplazarDuplicados(inicio, fin, link, "rostroHR", rutaHR)
    #decimar(inicio,fin,rutaHR,rutaLR)

if (cargar):

    opencvPSNR=np.load(r"C:\Users\Estudiante\Documents\dataset\resultados\librerias\opencvPSNR.npy")
    tensorPSNR=np.load(r"C:\Users\Estudiante\Documents\dataset\resultados\librerias\tensorPSNR.npy")
    pillowPSNR = np.load(r"C:\Users\Estudiante\Documents\dataset\resultados\librerias\pillowPSNR.npy")
    opencvSSIM=np.load(r"C:\Users\Estudiante\Documents\dataset\resultados\librerias\opencvSSIM.npy")
    tensorSSIM=np.load(r"C:\Users\Estudiante\Documents\dataset\resultados\librerias\tensorSSIM.npy")
    pillowSSIM = np.load(r"C:\Users\Estudiante\Documents\dataset\resultados\librerias\pillowSSIM.npy")


    print("OpenCV ")
    print("*************** PSNR *************** SSIM")
    print("promedio ", opencvPSNR.mean(), "  ", opencvSSIM.mean())
    print("std ", np.std(opencvPSNR), "  ", np.std(opencvSSIM))
    print("mediana ", np.median(opencvPSNR), "  ", np.median(opencvSSIM))
    print("minimo ", np.amin(opencvPSNR), "  ", np.amin(opencvSSIM))
    print("maximo ", np.amax(opencvPSNR), "  ", np.amax(opencvSSIM))
    print("***")

    print("tensorflow ")
    print("*************** PSNR *************** SSIM")
    print("promedio ", tensorPSNR.mean(), "  ", tensorSSIM.mean())
    print("std ", np.std(tensorPSNR), "  ", np.std(tensorSSIM))
    print("mediana ", np.median(tensorPSNR), "  ", np.median(tensorSSIM))
    print("minimo ", np.amin(tensorPSNR), "  ", np.amin(tensorSSIM))
    print("maximo ", np.amax(tensorPSNR), "  ", np.amax(tensorSSIM))
    print("***")

    print("pillow ")
    print("*************** PSNR *************** SSIM")
    print("promedio ", pillowPSNR.mean(), "  ", pillowSSIM.mean())
    print("std ", np.std(pillowPSNR), "  ", np.std(pillowSSIM))
    print("mediana ", np.median(pillowPSNR), "  ", np.median(pillowSSIM))
    print("minimo ", np.amin(pillowPSNR), "  ", np.amin(pillowSSIM))
    print("maximo ", np.amax(pillowPSNR), "  ", np.amax(pillowSSIM))
    print("***")






