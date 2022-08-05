

import requests
import os
import cv2

import numpy as np

import tensorflow as tf
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from io import BytesIO

#Descarga varias imagenes del sitio web utilizando pillow entre las posiciones de incio y fin
def descargarImagenesPillow(inicio: int, fin: int, link: str):
    for i in range (inicio,fin):
        name = 'rostroHR'+ str(i)
        with open(name.replace(' ', '-').replace('/', '') + '.jfif', 'wb') as f:
            im = requests.get(link)
            f.write(im.content)
            print('Writing: ', name)

#Descarga una imagen utilizando la libreria pillow
def descargarImagenPillow(nombre: str, link: str):
    with open(nombre.replace(' ', '-').replace('/', '') + '.png', 'wb') as f:
        im = requests.get(link)
        f.write(im.content)
        print('Writing: ', nombre)

#Descarga imagenes del sitio web utilizando pillow de acuerdo a la lista de posiciones usado como entrada
def descargarImagenesListaPillow(lista: [], link: str, nombre: str):
    for i in lista:
        name = nombre+ str(i)
        with open(name.replace(' ', '-').replace('/', '') + '.jfif', 'wb') as f:
            im = requests.get(link)
            f.write(im.content)
            print('Writing: ', name)

##Descarga una imagen utilizando la libreria opencv
def getImagen(link: str):
    im = requests.get(link)
    bgr=np.array(Image.open(BytesIO(im.content)))
    return cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)

#Descarga una imagen y la almacena utilizando la libreria opencv
def descargarImagen(nombre: str, link: str, url: str):
    img=getImagen(link)
    name=nombre.replace(' ', '-').replace('/', '') + '.png'
    cv2.imwrite(url+'/'+name, img)
    print('Writing: ', nombre)

#Descarga varias imagenes del sitio web utilizando opencv entre las posiciones de incio y fin
def descargarImagenes(inicio: int, fin: int, link: str, url :str):
    for i in range(inicio, fin):
        name = 'rostroHR' + str(i)
        descargarImagen(name,link,url)

#Descarga imagenes del sitio web utilizando opencv de acuerdo a la lista de posiciones usado como entrada
def descargarImagenesLista(lista: [], link: str, nombre: str, url :str):
    for i in lista:
        name = nombre + str(i)
        descargarImagen(name, link, url)

#Grafica una imagen
def graficarImagen(img, window_name: str):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Disminuyo la resolución de una imagen
   #Usar inter area como metodo de interpolación evita el aliasing
def procesarImagen(img):
    return cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

#Carga una imagen dada la ubicación de este en el computador
def cargarImagen(url: str):
    img = cv2.imread(url, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Permite convertir un tensor de tensorflow a numpy
def tensorToNumpy(img):
    graph = img.numpy().astype(dtype=np.uint8)
    return cv2.cvtColor(graph, cv2.COLOR_RGB2BGR)

# Permite convertir un objeto de pillow a numpy
def pillowToNumpy(img):
    almohadaNp = np.array(img)
    return cv2.cvtColor(almohadaNp, cv2.COLOR_RGB2BGR)

#Carga una imagen dada una ruta en las tres librerias y las decima con cada una
   #El resultado de la función son las tres imagenes decimadas
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

#Incrementa la resolución de la imagen a la resolución de las imagenes originales
   #Usar inter area como metodo de interpolación evita el aliasing
def generadorExpansivas(cv,tensor, almohada):
    pruebaCV2 = cv2.resize(cv, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    pruebaTensor = tf.image.resize(tensor, (1024, 1024), method='nearest')
    pruebaPillow = almohada.resize((1024, 1024), Image.NEAREST)
    return pruebaCV2, pruebaTensor, pruebaPillow

#Calcula el SSIM de tres imagenes distintas proveniente de cada una de las librerias
def calculadorSSIM(pruebaCV2, pruebaTensor, pruebaPillow, img):
    ssimCv2 = ssim(im1=img, im2=pruebaCV2, channel_axis=2)
    ssimTensor = ssim(im1=img, im2=tensorToNumpy(pruebaTensor), channel_axis=2)
    ssimPillow = ssim(im1=img, im2=pillowToNumpy(pruebaPillow), channel_axis=2)
    return ssimCv2,ssimTensor,ssimPillow

#Calcula el PSNR de tres imagenes distintas proveniente de cada una de las librerias
def calculadorPSNR(pruebaCV2, pruebaTensor, pruebaPillow, img):
    psnrCv2 = psnr(img, pruebaCV2)
    psnrTensor = psnr(img, tensorToNumpy(pruebaTensor))
    psnrPillow = psnr(img, pillowToNumpy(pruebaPillow))
    return psnrCv2,psnrTensor,psnrPillow

# carga una imagen con base en la ruta, indicativo o posición de dicha imagen y su nombre: rostroHR o rostroLR
def cargarImagenIndice(nombre: str, indicativo: int, url: str):
    direccion = nombre + str(indicativo) + ".png"
    rutaCompleta = url + "/" + direccion
    img = cv2.imread(rutaCompleta, 1)
    return img

# Esta fucnión embebe todo el protocolo de pruebas realizado para las librerias evaluadas
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

#Compara dos imagenes y determina si estas son en realidad la misma imagen
def comparadorImagenes(im1, im2):
    comparacion = np.equal(im1, im2)
    return comparacion .all()

# Con base en una ruta y un intervalo, se desea determinar cuales son las imagenes repetidas
   #Esta función usa como supuesto que las imágenes repetidas se encuentran una al lado de la otra
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

# Con base en una ruta y un intervalo, se desea reemplazar imagenes repetidas por imagenes diferentes
def reemplazarDuplicados(inicio: int, fin: int, link: str, nombre:str, ruta:str):
    repetidos = buscadorRepetidos(inicio, fin, nombre, ruta)
    descargarImagenesLista(repetidos, link, nombre, ruta)

# Almacena una imagen con base en un nombre: rostroHR o RostroLR, posición y dirección.
def almacenarImagen(img, nombre: str, pos: int, url: str):
    cv2.imwrite(url + '/'+nombre+str(pos)+'.png', img)

# Con base en una ruta y un intervalo, se desea decimar un grupo de imágenes y almacenarlas en un lugar distinto
def decimar(inicio: int, fin: int, rutaO:str, rutaD:str):
    for i in range (inicio, fin):
        img=cargarImagenIndice("rostroHR",i,rutaO)
        img=procesarImagen(img)
        almacenarImagen(img,"rostroLR",i,rutaD)
        print("decimate rostroLR",i)


if __name__ == '__main__':
    #Sitio web que genera las imágenes
    link = 'https://thispersondoesnotexist.com/image'
    #Ruta donde se almacenan las imagenes en alta resolución
    rutaHR = r"C:\Users\Estudiante\Documents\dataset\groundTruth\eval"
    # Ruta donde se almacenan las imagenes en baja resolución
    rutaLR = r"C:\Users\Estudiante\Documents\dataset\decimadasX8\eval"

    #Punto de inicio de intervalo de interes
    inicio=16500
    #Punto final de intervalo de interes
    fin=22000
    #Este bool determina si se desean cargar los resultados de las pruebas sobrer las librerias
    cargar=True
    #Escala con la cual se desea disminuir la resolución de la imagen
    escala=8

    #Realiza las pruebas sobre las librerias
    #pruebas(inicio, fin, rutaHR ,"rostroHR")

    #Este grupo de funciones buscan descargar imagenes del sitio web, eliminar duplicados y decimar dicho resultado
    #descargarImagenes(inicio,fin,link,rutaHR)
    #reemplazarDuplicados(inicio, fin, link, "rostroHR", rutaHR)
    #decimar(inicio,fin,rutaHR,rutaLR)

# Con base en las ubicaciones de los archivos npy donde se almacenan los resultados, se cargan los datos y se obtienen información valiosa de estos.
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






