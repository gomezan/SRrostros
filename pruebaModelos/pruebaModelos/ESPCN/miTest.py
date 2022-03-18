import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
#import PIL.Image as pil_image
import cv2

from  pruebaModelos.ESPCN.models import ESPCN
from pruebaModelos.ESPCN.utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


if __name__ == '__main__':
    weight = r"C:\Users\Estudiante\Documents\dataset\prueba\rostros1.pth"
    imagen = r"C:\Users\Estudiante\Documents\dataset\decimadasX2\rostroLR0.png"
    gt=r"C:\Users\Estudiante\Documents\dataset\groundTruth\rostroHR0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default=weight)
    parser.add_argument('--image-gt', type=str, default=gt)
    parser.add_argument('--image-file', type=str, default=imagen)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    #verificacion GPU
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #creacion de modelo
    model = ESPCN(scale_factor=args.scale).to(device)

    #subida de pesos
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    #switch evaluacion/ejecucion
    model.eval()

    #subir imagen ground truth y homologo de menor resolucion
    hr = cv2.imread(args.image_gt, 1)
    lr = cv2.imread(args.image_file, 1)

    #obtener tamaño
    image_width = lr.shape[0]
    image_height = lr.shape[1]

    #obtener imagen de lanczos a partir de la iamgen escalada
    lanczos =cv2.resize(lr, (image_width * args.scale, image_height * args.scale), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(args.image_file.replace('.', '_lanczos_x{}.'.format(args.scale)), lanczos)

    #obtiene las imagenes normalizadas y el ycbcr de la imagen interpolada por lanczos;  transfiere a device
    lr, _ = preprocess(lr, device)
    _, ycbcr = preprocess(lanczos, device)

    #apaga gradiente y filtra valores entre 0 y 1
    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    #psnr con respecto a la imagen ground truth y la predicha
    psnr = calc_psnr(hr, preds)
    print('PSNR: {:.2f}'.format(psnr))

    #trata la imagen
        #desnormaliza
        #quita dimensiones
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    #???? axes
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    #Clip y converion RGB
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

    #almacena imagen
    cv2.imwrite(args.image_file.replace('.', '_espcn_x{}.'.format(args.scale)), output)
