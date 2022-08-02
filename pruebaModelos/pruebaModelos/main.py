
import sys

if __name__ == '__main__':

    #ESPCN
    test=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\test.py"
    prepare=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\prepare.py"
    train = r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\train.py"


    myTest=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\miTest.py"
    myPrepare=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\miPrepare.py"
    res=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\resultadosPSNR_SSIM.py"
    play=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\playGround.py"
    other=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\resultadosOtrasMetricas.py"


    graph=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\graph.py"
    graficador=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\graficador.py"

    # param
    archivoExec = myTest
    source_code = open(archivoExec).read()
    sys.argv = [archivoExec]
    exec(source_code)


