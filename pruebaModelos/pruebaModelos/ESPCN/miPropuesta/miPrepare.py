import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from pruebaModelos.ESPCN.utils import convert_rgb_to_y
import cv2

#Este scipt es una variante del script prepare del ESPCN original
#Busca construir el archivo extraible del conjunto de datos de entrenamiento y evaluación

#Construye el dataset de entrenamiento empleando la componente y
def train(args):
    #archivo de destino
    h5_file = h5py.File(args.output_path, 'w')

    #Cargo las rutas de las imagenes en las carpeta del ground truth y las imagenes decimadas
    gtGlob=sorted(glob.glob('{}/*'.format(args.images_dir)))
    sGlob=sorted(glob.glob('{}/*'.format(args.images_pro)))

    #Esta información es utilizada para la indexación de las imágenes dentro del archivo h5
    NoImg=len(gtGlob)
    sizeLR=args.patch_size
    sizeHR=args.patch_size * args.scale
    noTrozos=len(range(0, 204 - args.patch_size + 1, args.stride))
    total = NoImg * noTrozos*noTrozos

    #Creo datatset vacio con el tamaño total de todas las imágenes
    dataLr = h5_file.create_dataset('lr', (total, sizeLR, sizeLR))
    dataHr = h5_file.create_dataset('hr', (total, sizeHR, sizeHR))

    # carga cada imagen y la transforma en formato Y
    for index, image_gt, image_s in zip( range(0,NoImg), gtGlob, sGlob):
        hr = cv2.imread(image_gt, 1)
        lr = cv2.imread(image_s, 1)
        hr = np.float32(hr)
        lr = np.float32(lr)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        #Indexa cada una de las imágenes dentro del archivo h5 de acuerdo a la información suministrada arriba
        for i,k in  zip(range(0, lr.shape[0] - args.patch_size + 1, args.stride),range(0,noTrozos)):
            for j,l in zip(range(0, lr.shape[1] - args.patch_size + 1, args.stride),range(0,noTrozos)):
                pos=index*noTrozos*noTrozos+k*noTrozos+l
                print("en train se esta desarrollando la imagen : ", pos)
                dataLr[pos]=lr[i:i + args.patch_size, j:j + args.patch_size]
                dataHr[pos]=hr[i * args.scale:i * args.scale + args.patch_size * args.scale, j * args.scale:j * args.scale + args.patch_size * args.scale]

    h5_file.close()


#decima cada imagen la carpeta, la almacena en formato Y, asigna un indice y almacena la información
def eval(args):

    print("finalmente se prepara eval")
    #Ubicación carpeta con imagenes ground truth
    origenGT = r"C:\Users\Estudiante\Documents\dataset\groundTruth\eval"
    #Ubicación carpeta con imagenes decimadas
    origenX2 = r"C:\Users\Estudiante\Documents\dataset\decimadasX8\eval"
    # Ubicación archivo h5 de destino
    destino = r"C:\Users\Estudiante\Documents\dataset\prueba\evalx8.h5"

    #abrir archivo h5
    h5_file = h5py.File(destino, 'w')

    #Se crean dos grupos de imagenes para el ground truth y las imagenes decimadas
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    # Cargo las rutas de las imagenes en las carpeta del ground truth y las imagenes decimadas
    gtGlob = sorted(glob.glob('{}/*'.format(origenGT)))
    sGlob = sorted(glob.glob('{}/*'.format(origenX2)))

    NoImg = len(gtGlob)

    #Carga las imagenes las transforma a formato Y y las almacena como conjunto de datos dentro del grupo
    for index, image_gt, image_s in zip(range(0, NoImg), gtGlob, sGlob):
        print("en eval se esta desarrollando la imagen : ", image_gt)
        hr = cv2.imread(image_gt, 1)
        lr = cv2.imread(image_s, 1)
        hr=np.float32(hr)
        lr=np.float32(lr)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(index), data=lr)
        hr_group.create_dataset(str(index), data=hr)

    h5_file.close()


if __name__ == '__main__':
    # Ubicación carpeta con imagenes ground truth
    origenGT = r"C:\Users\Estudiante\Documents\dataset\groundTruth\train"
    # Ubicación carpeta con imagenes decimadas
    origenX2 = r"C:\Users\Estudiante\Documents\dataset\decimadasX8\train"
    # Ubicación archivo h5 de destino
    destino = r"C:\Users\Estudiante\Documents\dataset\prueba\trainx8.h5"

    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default=origenGT)
    parser.add_argument('--images-pro', type=str, default=origenX2)
    parser.add_argument('--output-path', type=str, default=destino)
    parser.add_argument('--scale', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=17)
    parser.add_argument('--stride', type=int, default=13)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()


    #Crear conjunto de entrenamiento y evaluación
    #Verificar tener espacio suficiente
    train(args)
    eval(args)

