import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from  ESPCN.models import ESPCN
from ESPCN.utils import convert_ycbcr_to_rgb, preprocess, preprocessRGB


def graficarImagen(img, window_name: str):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def graficarComparacion(img1, img2, window_name: str):
    concat_horizontal = cv2.hconcat([img1, img2])
    cv2.imshow(window_name, concat_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

pesos = {
        2: r"C:\Users\Estudiante\Documents\dataset\prueba\espcn_x2Full.pth",
        4: r"C:\Users\Estudiante\Documents\dataset\prueba\best0x4.pth",
        8: r"C:\Users\Estudiante\Documents\dataset\prueba\espcnx8.pth"
}


if __name__ == '__main__':

    escala=8
    imagen = r"C:\Users\Estudiante\Documents\dataset\decimadasX"+str(escala)+r"\eval\rostroLR16858.png"
    gt=r"C:\Users\Estudiante\Documents\dataset\groundTruth\eval\rostroHR16858.png"

    weight = pesos[escala]

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default=weight)
    parser.add_argument('--image-gt', type=str, default=gt)
    parser.add_argument('--image-file', type=str, default=imagen)
    parser.add_argument('--scale', type=int, default=escala)
    args = parser.parse_args()

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

    #subir imagen ground truth y homologo de menor resolucion
    hr = cv2.imread(args.image_gt, 1)
    lr = cv2.imread(args.image_file, 1)

    #obtener tamaño)
    image_width = lr.shape[0]
    image_height = lr.shape[1]

    #obtener imagen de lanczos a partir de la iamgen escalada
    lanczos =cv2.resize(lr, (image_width * args.scale, image_height * args.scale), interpolation=cv2.INTER_LANCZOS4)
    #graficarImagen(lanczos, args.image_file.replace('.', '_lanczos_x{}.'.format(args.scale)) )


    print("Este experimento solo tiene en cuenta la componente Y")

    #obtiene las imagenes normalizadas y el ycbcr de la imagen interpolada por lanczos;  transfiere a device
    lrPrep, _ = preprocess(lr, device)
    _, ycbcr = preprocess(lanczos, device)

    #apaga gradiente y filtra valores entre 0 y 1
    with torch.no_grad():
        preds = model(lrPrep).clamp(0.0, 1.0)

    #trata la imagen
        #desnormaliza
        #quita dimensiones
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    #???? axes
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    #Clip y converion RGB
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    #graficarImagen(output, "solo componente Y")
    cv2.imwrite("imagenPaperY"+str(args.scale)+".png", output)

    # psnr con respecto a la imagen ground truth y la predicha
    res = psnr(hr, output)
    print('PSNR del modelo: {:.2f}'.format(res ))
    res = psnr(hr, lanczos)
    print('PSNR de lanczos: {:.2f}'.format(res))
    res = ssim(im1=hr, im2=output, channel_axis=2)
    print('SSIM del modelo: {:.2f}'.format(res))
    res = ssim(im1=hr, im2=lanczos, channel_axis=2)
    print('SSIM de lanczos: {:.2f}'.format(res))


    #obtener imagen de comparación
    diff=20*np.log10(np.abs(output-hr))
    resY=np.nan_to_num(diff).astype(np.uint8)

    print("Este experimento tiene en cuenta las componentes R, G, B")

    R=lr[...,0]
    G=lr[...,1]
    B=lr[...,2]

    #preprocesamiento y device
    RPrep=preprocessRGB(R,device)
    GPrep = preprocessRGB(G, device)
    BPrep = preprocessRGB(B, device)

    # apaga gradiente y filtra valores entre 0 y 1
    with torch.no_grad():
        predsR = model(RPrep).clamp(0.0, 1.0)
        predsG = model(GPrep).clamp(0.0, 1.0)
        predsB = model(BPrep).clamp(0.0, 1.0)

    predsR = predsR.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    predsG = predsG.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    predsB = predsB.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([predsR, predsG, predsB]).transpose([1, 2, 0]).astype(np.uint8)
    #graficarImagen(output, "Compoenentes RGB")
    cv2.imwrite("imagenPaperRGB" + str(args.scale) + ".png", output)

    # psnr con respecto a la imagen ground truth y la predicha
    res = psnr(hr, output)
    print('PSNR del modelo: {:.2f}'.format(res))
    res = psnr(hr, lanczos)
    print('PSNR de lanczos: {:.2f}'.format(res))
    res = ssim(im1=hr, im2=output, channel_axis=2)
    print('SSIM del modelo: {:.2f}'.format(res))
    res = ssim(im1=hr, im2=lanczos, channel_axis=2)
    print('SSIM de lanczos: {:.2f}'.format(res))

    diff = 20 * np.log10(np.abs(output - hr))
    resRGB = np.nan_to_num(diff).astype(np.uint8)

    #graficarComparacion(resY,resRGB, "comparacion_Log")
