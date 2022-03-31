
import sys

if __name__ == '__main__':

    #ESPCN
    test=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\test.py"
    prepare=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\prepare.py"
    train = r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\train.py"


    myTest=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\miTest.py"
    myPrepare=r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\ESPCN\miPrepare.py"

    #LESRCNN
    sample = r"C:\Users\Estudiante\Documents\GitHub\SRrostros\pruebaModelos\pruebaModelos\LESRCNN\lesrcnn_b\tcw_sample_b.py"

    # param
    archivoExec = myPrepare
    source_code = open(archivoExec).read()
    sys.argv = [archivoExec]
    exec(source_code)


