import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from  pruebaModelos.ESPCN.models import ESPCN
from pruebaModelos.ESPCN.utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


if __name__ == '__main__':
    weight = r"C:\Users\Estudiante\Documents\dataset\prueba\rostros1.pth"
    imagen = r"C:\Users\Estudiante\Documents\dataset\imagenesHR\rostroHR0.png"
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default=weight)
    parser.add_argument('--image-file', type=str, default=imagen)
    parser.add_argument('--scale', type=int, default=3)
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

    #subir imagen Pillow
    image = pil_image.open(args.image_file).convert('RGB')

    #obtener numeros enteros del tamaño
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    #resize de cauerdo al tamaño
    #imagenes:
        #hr
        #lr
        #bicubica--guarda inmediatamente
    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    #obtiene las imagenes normalizadas y el ycbcr de la imagen bicubica;  transfiere a device
    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    #apaga gradiente y filtra valores entre 0 y 1
    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    #psnr con respecto a la imagen hr y la predicha
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
    #convierte imagen a PIL
    output = pil_image.fromarray(output)
    #almacena imagen
    output.save(args.image_file.replace('.', '_espcn_x{}.'.format(args.scale)))
