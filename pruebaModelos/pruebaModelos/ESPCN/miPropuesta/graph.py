
import numpy as np
import matplotlib.pyplot as plt
import glob

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data.dataloader import DataLoader

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

from models import ESPCN
from datasets import EvalDataset
from utils import AverageMeter, calc_psnr

#Este script es utilizado para realizar graficas de métricas como el psnr y el ssim con rerspecto a las épocas de entrenamiento


#Su función es graficar una determinada métrica con respecto a las épocas de entrenamiento
def plot_Metricas(ruta, total_psnr, nombre):
    epochs_plot=range(0,410,10)
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, total_psnr , label = nombre+'/ epoch ')
    plt.legend()
    # plt.show()
    plt.xlabel('Número de época', fontsize=28)   
    plt.ylabel(nombre, fontsize=28)
    plt.rcParams.update({'font.size': 22})
    plt.savefig(ruta +nombre+'.png', bbox_inches='tight')
    np.save(ruta +nombre+'.npy' ,np.array(total_psnr))


#Permite cargar determinados pesos sobre la ESPCN
def cargaPesos(model, path):
    state_dict = model.state_dict()

    for n, p in torch.load(path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
            
            

    model.eval()

#Escala deseada a evaluar
escala=4
#Ubicación de archivo comprimido de imágenes usadas para evaluar las métricas
evalPath=r"/HDDmedia/supermri/evalx4.h5"
#Número del dispositivo gpu o cpu usado para el modelo
dispo=2

#Ubicación de carpeta con todos los archivos de pesos a evaluar
pesosPath=r"/HDDmedia/supermri/x"+str(escala)+"/graficarPesos/"


#Instancia el modelo con cada uno de los pesos y evalua las imagenes con cada una de las métricas a evaluar
def calcularMetricas(model, path : str):
    #Carga de rutas de los archivos de pesos
    files=sorted(glob.glob(path+r"*.pth"))
    #Carga de las dataset
    eval_dataset = EvalDataset(evalPath)
    #Instanciación del controlador del dataset
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    psnrProm=[]
    ssimProm=[]
    mseProm =[]

    #Por cada uno de los pesos, evalua las imagenes con cada una de las métricas a evaluar
    for file in files:
        print(file)
        cargaPesos(model, file)

        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_mse = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            prd=preds.cpu().numpy().squeeze(0).squeeze(0)
            lbl=labels.cpu().numpy().squeeze(0).squeeze(0)

            epoch_psnr.update(psnr(lbl, prd ), len(inputs))
            epoch_ssim.update(ssim(im1=lbl, im2=prd, data_range = lbl.max() - prd.min()), len(inputs))
            epoch_mse.update(mse(lbl, prd), len(inputs))


        psnrProm.append(float(epoch_psnr.avg))
        ssimProm.append(float(epoch_ssim.avg))
        mseProm.append(float(epoch_mse.avg))

    return psnrProm, ssimProm , mseProm




#Instanciar modelo
#cudnn.benchmark = True
#device = torch.device('cuda:'+str(dispo) if torch.cuda.is_available() else 'cpu')

#model = ESPCN(scale_factor=escala).to(device)
#model = nn.DataParallel(model, device_ids=[dispo,dispo+1])
#model.to(device)

#IniciarPruebas
#x,y,z=calcularMetricas(model,pesosPath)
#print(x)
#print(y)
#print(z)

a=[36.03717168494674, 36.10606788211882, 36.4658827637694, 36.16103357147971, 36.49666893229572, 36.60324791378373, 36.58829477579161, 36.62237240124559, 36.632809218164724, 36.520311613217956, 36.60586831001927, 36.62370108087458, 36.65615316507024, 36.66356351770049, 36.62723138463519, 36.654110707727796, 36.683261219080265, 36.686107692419434, 36.69206567887002, 36.69491837810155, 36.69719040341448, 36.649268799357166, 36.694140506388976, 36.6776657979092, 36.71091254466728, 36.70903161116539, 36.666377087659626, 36.71292453810052, 36.721594311643955, 36.73232879048559, 36.716646907082016, 36.722588410720896, 36.657254152448296, 36.72690541805319, 36.72903649930638, 36.72609412023107, 36.73257781953943, 36.73890612628241, 36.75210065707174, 36.75553609057586, 36.75339292580925]
b=[0.9055398209762059, 0.90551566881367, 0.9081292998189997, 0.9090532615886643, 0.9096797255915782, 0.9101249839627812, 0.9099134198534123, 0.9103128246857002, 0.910563459220938, 0.910273746828295, 0.9102388176742282, 0.910021696770026, 0.9108477452786808, 0.9109897575007517, 0.9104000632809235, 0.9105075827907614, 0.9110158235336764, 0.9109596620967536, 0.9110778676773564, 0.9111068627005233, 0.911150028506503, 0.910663312829469, 0.9110880738803648, 0.9106247602725254, 0.9110945631874618, 0.911163312963278, 0.91100363886382, 0.9107507861766999, 0.9111644425854146, 0.9111974915949235, 0.910766915462485, 0.911057871244198, 0.9107230060123683, 0.9110235976138146, 0.9109508177305818, 0.9109445810978156, 0.9109675220554497, 0.9111616952568564, 0.9114521551848996, 0.9114353997563484, 0.911439237594983]
c=[0.00028185211371770997, 0.0002773535618499797, 0.00025597807150867503, 0.00027039142064483286, 0.00025330106768906604, 0.0002482768971037816, 0.0002488480657407334, 0.000247154330649997, 0.0002466286625808214, 0.0002516846460854462, 0.0002480725714171468, 0.00024688884736790703, 0.0002453262750936626, 0.0002449240821560861, 0.0002466050860314007, 0.00024531512590105165, 0.00024392286246206562, 0.00024374594199165332, 0.00024345780910336993, 0.00024330841970783044, 0.0002431667070598332, 0.00024533165444020385, 0.0002433012026870394, 0.00024399668096283613, 0.00024232633984529322, 0.0002424197229856136, 0.00024426416181203415, 0.00024208724017617104, 0.00024172468845696688, 0.00024113043524274174, 0.00024186536517125393, 0.00024156176293258958, 0.0002450623921360679, 0.00024132737741080086, 0.00024118062735733374, 0.00024131334264903252, 0.00024098121697273953, 0.00024066539304303216, 0.00024003295330802957, 0.0002398443638324996, 0.00023995260428873343]


#Crea las graficas y las almacena en el path correspondiente
plot_Metricas(pesosPath, b,"SSIM")
plot_Metricas(pesosPath, c,"MSE")
plot_Metricas(pesosPath, a,"PSNR")


