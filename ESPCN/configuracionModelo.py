
import torch
import torch.backends.cudnn as cudnn
from models import ESPCN

#Permite cargar los pesos de un archivo determinado a el modelo de entrada
def cargarPesos(weight,model):
    state_dict = model.state_dict()
    for n, p in torch.load(weight, map_location=lambda storage, loc: storage).items():
        if n[7:] in state_dict.keys():
            state_dict[n[7:]].copy_(p)
        else:
            raise KeyError(n)

#Permite instanciar el modelo dada la escala de entrada
def ajusteModelo(escala):

    print("Configurando red")

    #Ruta del archivo de los pesos
    weights = r"/home/nvidia/Documentos/ESPCN/pesos/espcn_x"+str(escala)+".pth"

    #verificacion GPU
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #creacion de modelo
    model = ESPCN(scale_factor=escala).to(device)

    #subida de pesos
    cargarPesos(weights,model)
        
    #switch evaluacion/ejecucion
    model.eval()
    
    return model, device
    
    
