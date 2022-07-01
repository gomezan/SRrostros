

import numpy as np
import seaborn
import pandas as pd
import matplotlib.pyplot as plt

#Permite relacionar el método utilizado con el nombre del array que contienen las metricas.
psnrDic={
    "nn": "\INEARESTPSNR.npy",
    "bi2": "\ILINEARPSNR.npy",
    "bi3": "\ICUBICPSNR.npy",
    "lan": "\IlanczosPSNR.npy",
    "y": r"\redPSNR_Y.npy",
    "rgb": r"\redPSNR_RGB.npy"
}

ssimDic={
    "nn": "\INEARESTSSIM.npy",
    "bi2": "\ILINEARSSIM.npy",
    "bi3": "\ICUBICSSIM.npy",
    "lan": "\IlanczosSSIM.npy",
    "y": r"\redSSIM_Y.npy",
    "rgb": r"\redSSIM_RGB.npy"
}


blurDic={
    "nn": r"NEARESTBLUR.npy",
    "bi2": r"LINEARBLUR.npy",
    "bi3": r"CUBICBLUR.npy",
    "lan": r"lanczosBLUR.npy",
    "y": r"redBlur_y.npy",
    "rgb": r"redBlur_RGB.npy"
}

faceDic={
    "nn": r"INEARESTFace",
    "bi2": r"ILINEARFace",
    "bi3": r"ICUBICFace",
    "lan": r"IlanczosFace",
    "y": r"redFace_y",
    "rgb": r"redFace_RGB"
}


snrDic={
    "nn": r"NEARESTSNR.npy",
    "bi2": r"LINEARSNR.npy",
    "bi3": r"CUBICSNR.npy",
    "lan": r"lanczosSNR.npy",
    "y": r"redSNR_y.npy",
    "rgb": r"redSNR_RGB.npy"
}


mtdName={
    "nn": "Vecinos",
    "bi2": "Bilineal",
    "bi3": "Bicubica",
    "lan": "Lanczos",
    "y": "ESPCN_Y",
    "rgb": "ESPCN_RGB"
}

# Crea un dataframe de los datos de una métrica utilizando el diccionario de PSNR o SSIM.
def crearConjuntoBigotes(dic, metrica, escala):

    x = pd.DataFrame()
    for clave in dic:
        data = np.load(r"C:\Users\Estudiante\Documents\dataset\resultados\calidad\x"+str(escala)+r"\I"+dic[clave])
        df = pd.DataFrame(
            {metrica: data})
        df['Metodo'] = mtdName[clave]
        x = pd.concat([x, df])
    return x


# Crea un dataframe de los datos de una métrica utilizando el diccionario de PSNR o SSIM.
def crearConjuntoBigotesFace(dic, metrica, escala):

    x = pd.DataFrame()
    for clave in dic:
        for k in [0,1,2,4,5]:
            data = np.load(r"C:\Users\Estudiante\Documents\dataset\resultados\identidad\x"+str(escala)+r"_"+dic[clave]+str(k)+".npy")
            print(data.shape)
            df = pd.DataFrame({metrica: data.squeeze()})
            df['Metodo'] = mtdName[clave]
            x = pd.concat([x, df])
    return x

# Crea un diagrama de cajas utilizando el diccionario de PSNR o SSIM.
def graficarBigotes(dic, metrica, escala):
    x=crearConjuntoBigotesFace(dic, metrica, escala)
    print(x)
    seaborn.set(style='whitegrid')
    seaborn.boxplot(x='Metodo', y=metrica, data=x)
    plt.savefig(metrica+str(escala)+ ".png")
    plt.show()

# Estas listas contienen los resultados obtenidos por el modelo cada 50 epocas.
promPSNR=[29.91,33.97,37.51,37.86,37.89,37.91,37.89,37.93,37.89,37.91]
promSSIM=[0.81,0.82,0.85,0.87,0.89,0.90,0.90,0.90,0.90,0.90]

# Con base a una lista de entrada se crea un dataframe para graficar dicha información
def crearConjuntoHist(prom):
    df=pd.DataFrame()
    x=range(50,55*len(prom),50)
    df["Epoca"]=x
    df["SSIM promedio"]=prom
    return df

#Con base en una lista se grafica un diagrama de barras con dichos resultados.
def graficarHist(prom):
    data=crearConjuntoHist(prom)
    print(data)
    seaborn.barplot(x=data["Epoca"],y=data["PSNR promedio"], palette="Blues_d")
    plt.show()

#Con base en una lista se grafica una curva con dichos resultados.
def graficarCurva(prom):
    data=crearConjuntoHist(prom)
    print(data)
    seaborn.relplot(x=data["Epoca"],y=data["SSIM promedio"],kind="line", palette="Blues_d")
    plt.annotate("peso escogido", (data["Epoca"][3] + 0.7, data["SSIM promedio"][3]))
    plt.show(marker="s", ms=12, markevery=[0,1])


metodos=[
     "Vecinos",
     "Bilineal",
     "Bicubica",
    "Lanczos",
     "ESPCN_Y ",
    " ESPCN_RGB"]

fidX2=[450.478,2523.689,621.101,442.130,51.528,33.959]
fidX4=[2647.250,9295.356,5495.174,5193.589,2372.120,2121.905]
fidX8=[3600.985, 19123.293, 15120.286, 14413.109, 8893.360, 8443.401 ]

def crearConjuntoCat(metodos,fid):
    df = pd.DataFrame()
    df['Metodos'] = metodos
    df['FID'] = fid
    return df

def graficarBarras(metodos,fid, nombre):
    x=crearConjuntoCat(metodos,fid)
    seaborn.catplot(x = "Metodos", y = "FID", kind = "bar", data = x, ci = 30)
    plt.title(nombre)
    plt.savefig(nombre+".png")
    plt.show()

# Se selecciona la grafica que se desea obtener.
#graficarBigotes(ssimDic)
#graficarCurva(promPSNR)

# nombre="FID factor 8"
# graficarBarras(metodos,fidX8, nombre )


graficarBigotes(faceDic, "FaceDist", 8)