import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


from models import ESPCN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr


#Funciones para graficar parametros del entrenamiento

def plot_psnr(ruta,epochs_plot, total_psnr):
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, total_psnr , label ='psnr/ epoch ')
    plt.legend()
    # plt.show()
    plt.savefig(ruta+'psnr_plot.png', bbox_inches='tight')
    np.save(ruta+'psnrNPY.npy',np.array(total_psnr))
 
    
def plot_losses(ruta, epochs_plot, total_generator_g_error_plot):
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, total_generator_g_error_plot, label ='Genreator G loss')
    plt.legend()
    # plt.show()
    plt.savefig(ruta+'losses_plot.png', bbox_inches='tight')
    np.save(ruta+'lossNPY.npy',np.array(total_generator_g_error_plot))
    
    
#Esta fucnión permite cargar los pesos dada una ruta al modelo especifico   

def cargaPesos(model, path):
    state_dict = model.state_dict()

    for n, p in torch.load(path, map_location=lambda storage, loc: storage).items():
        if n[7:] in state_dict.keys():
            state_dict[n[7:]].copy_(p)
        else:
            raise KeyError(n)

if __name__ == '__main__':

	#Parametros de entrada

	#Escala a entrenar
	escala=8
	#Indica si el modelo debe entrenar desde 0 o si es inicializado por pesos especificos
	nuevo=0
	#Ruta de los pesos a cargar en caso de haber indicado que se entrena desde un pesos especificos
	pesos = r"/HDDmedia/supermri/x8/best1.pth"
	
	#Ruta conjunto de evaluación
    eval=r"/HDDmedia/supermri/evalx"+str(escala)+".h5"
    #Ruta conjunto de entrenamiento
    train=r"/HDDmedia/supermri/trainx"+str(escala)+".h5"
    #Ruta donde se almacenan los resultados obtenidos
    final = r"/HDDmedia/supermri/x"+str(escala)+"/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default=train)
    parser.add_argument('--eval-file', type=str, default=eval)
    parser.add_argument('--outputs-dir', type=str,default=final)
    parser.add_argument('--weights-file', type=str, default=pesos)
    parser.add_argument('--new', type=int, default=nuevo)

    parser.add_argument('--scale', type=int, default=escala)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    #crea carpeta para almacenar los resultados
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    #configura dispotisitvo
    cudnn.benchmark = True
    #Asigna como GPU primaria el dispositivo 4
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print(device)

    #asigna semilla
    torch.manual_seed(args.seed)

    #crea modelo
    model=  ESPCN(scale_factor=args.scale)
    if not (args.new):
    	cargaPesos(model, args.weights_file)

    #Habilita el procesamiento en paralelo entre los GPU 4 (dispositivo primario) y el 5
    model = nn.DataParallel(model, device_ids=[4,5 ])
    model.to(device)
    
	#Ajuste de hiperparámetros
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.module.first_part.parameters()},
        {'params': model.module.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    #carga dataset de entrenamiento y prueba
    train_dataset = TrainDataset(args.train_file)
    print(train_dataset.__len__())
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    #INicialización de variables y estructuras de datos
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    
    psnr=[]
    epochs=[]
    losses=[]

	#Entrenamiento del modelo
    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()

		#Esta libreria permite tener una representación grafica en consola del avance del entrenamiento
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            #Este ciclo corresponde al entrenamiento en una epoca y calcula función de perdida
            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

		#Almacena el resultado de la epoca
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

		# Se calcula la psnr sobre el conjunto de evaluación para estimar el estado de entrenamiento 
        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        
 
 		#Almacena el valor obtenido del psnr, numero de epoca y función deperdidas       
        psnr.append(float(epoch_psnr.avg))
        epochs.append(epoch)
        losses.append(float(loss))


		#El sistema almacena evalua si esta época ha sido aquella con el mejor desempeño para su almacenamiento
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

	#Finalizado el entrenamiento se imprime la epoca con mejor desemepño
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
    
    #Almacena las graficas del psnr y función de perdidas
    plot_psnr(final,epochs, psnr)
    plot_losses(final,epochs, losses)
    
    
    
