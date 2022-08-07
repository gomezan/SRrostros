

import face_recognition

#Imagen ground truth a comparar
gt_image = face_recognition.load_image_file("/HDDmedia/supermri/decimadasX2/eval/rostroLR19559.png")
#imagen de menor resolución a comparar
comp_image = face_recognition.load_image_file("/HDDmedia/supermri/decimadasX4/eval/rostroLR19559.png")

#Se codifican las imagenes dentro de la libreria
gt_encoding = face_recognition.face_encodings(gt_image)[0]
comp_encoding = face_recognition.face_encodings(comp_image)[0]

#Se comparan ambos rostros y se obtiene la distancia entre ellos
results = face_recognition.face_distance([gt_encoding], comp_encoding)
print(results)
