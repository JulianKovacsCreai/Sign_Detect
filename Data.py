import cv2
import os
import SeguimientoManos as sm

#Creacion de carpeta
nombre = 'Letra_I'
direccion = 'direccion de la carpeta' 
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print('Carpeta creada: ', carpeta)
    os.makedirs(carpeta)    
    
cap = cv2.VideoCapture(0) # Lectura de la camara
cap.set(3, 640)
cap.set(4, 480)

cont = 550
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    ret, frame = cap.read()#Lectura de captura
    frame = detector.encontrarmanos(frame, dibujar = False) # Extraer info de manos
    lista1, bbox, mano = detector.encontrarposicion(frame, 
                                              manoNumero = 0, 
                                              dibujarPuntos = False, 
                                              dibujarBox = False,
                                              color = [0,255,0]) # Extraer info de dedos
    
    if mano == 1:
        xmin, ymin, xmax, ymax = bbox
        xmin = xmin - 40 
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40 #Margenes de la box  
        recorte = frame[ymin:ymax, xmin:xmax] #Recorte de la imagen
        #recorte = cv2.resize(recorte, (500, 500), 
                             #interpolation = cv2.INTER_CUBIC) #Redimensionar    
        if recorte.size != 0:
            cv2.imwrite(carpeta + "/I_{}.jpg".format(cont), recorte) #Guardar imagen
            cont = cont + 1
        cv2.imshow('Recorte', recorte)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)  
    cv2.imshow('Lenguaje Vocales', frame) # Mostrar FPS
    t = cv2.waitKey(1) #Leer teclado
    if t == 27 or cont == 650:
        break
    
cap.release()  
cv2.destroyAllWindows()