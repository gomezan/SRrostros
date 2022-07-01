import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from pruebaModelos.ESPCN.utils import convert_rgb_to_y
import cv2


def train(args):
    #archivo de destino
    h5_file = h5py.File(args.output_path, 'w')

    #decima cada imagen la carpeta y la almacena en formato Y y almacena la información
    gtGlob=sorted(glob.glob('{}/*'.format(args.images_dir)))
    sGlob=sorted(glob.glob('{}/*'.format(args.images_pro)))

    NoImg=len(gtGlob)
    sizeLR=args.patch_size
    sizeHR=args.patch_size * args.scale
    noTrozos=len(range(0, 204 - args.patch_size + 1, args.stride))
    total = NoImg * noTrozos*noTrozos

    dataLr = h5_file.create_dataset('lr', (total, sizeLR, sizeLR))
    dataHr = h5_file.create_dataset('hr', (total, sizeHR, sizeHR))

    for index, image_gt, image_s in zip( range(0,NoImg), gtGlob, sGlob):
        hr = cv2.imread(image_gt, 1)
        lr = cv2.imread(image_s, 1)
        hr = np.float32(hr)
        lr = np.float32(lr)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)


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
    origenGT = r"C:\Users\Estudiante\Documents\dataset\groundTruth\eval"
    origenX2 = r"C:\Users\Estudiante\Documents\dataset\decimadasX8\eval"
    destino = r"C:\Users\Estudiante\Documents\dataset\prueba\evalx8.h5"

    h5_file = h5py.File(destino, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    gtGlob = sorted(glob.glob('{}/*'.format(origenGT)))
    sGlob = sorted(glob.glob('{}/*'.format(origenX2)))

    NoImg = len(gtGlob)

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
    origenGT = r"C:\Users\Estudiante\Documents\dataset\groundTruth\train"
    origenX2 = r"C:\Users\Estudiante\Documents\dataset\decimadasX8\train"
    #origenX3 = r"C:\Users\Estudiante\Documents\dataset\decimadasX3"

    #origenGT = r"C:\Users\Estudiante\Documents\dataset\pequenoGT\eval"
    #origenX2 = r"C:\Users\Estudiante\Documents\dataset\pequenoX2\eval"

    destino = r"C:\Users\Estudiante\Documents\dataset\prueba\trainx8.h5"
    #destino= r"C:\Users\Estudiante\Documents\dataset\prueba\eval.h5"
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default=origenGT)
    parser.add_argument('--images-pro', type=str, default=origenX2)
    parser.add_argument('--output-path', type=str, default=destino)
    parser.add_argument('--scale', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=17)
    parser.add_argument('--stride', type=int, default=13)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
        eval(args)
    else:
        #train(args)
        eval(args)
