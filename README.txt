El presente repositorio busca implementar un modelo de super resolución para vigilancia
sobre rostros humanos. Este puede dividirse en seis partes:

*botImagenes: Contiene toda la lógica que permite construir el conjunto de datos 
del proyecto.
*pruebaModelos: Este contiene el código original de la ESPCN (https://github.com/yjn870/ESPCN-pytorch)
más la propuesta necesaria para adecuar en el área de la vigilancia sobre rostros humanos.   
*ESPCN: Este consta de los modulos necesarios para desplegar el modelo sobre la tarjeta de desarrollo.
*miniConjunto: Esta carpeta contiene una versión muy reducida del conjunto de datos y los resultados en
super resolución obtenidos.
*resultados: Almacena imágenes tomadas por el sistema embebido a diferentes escalas y el resultado de 
incrementar su resolución x2, x4 y x8.
*Texto original: Carpeta que contiene el documento de trabajo de grado final detrás del proyecto.