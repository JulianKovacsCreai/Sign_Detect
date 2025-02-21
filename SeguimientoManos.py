import math
import cv2
import mediapipe as mp
import time

class detectormanos():
    def __init__(self, modo = False, maxmanos = 2, 
                 model_complexity = 1, Confdeteccion = 0.5, 
                 ConfSeguimiento = 0.5):
        self.modo = modo
        self.maxmanos = maxmanos
        self.Confdeteccion = Confdeteccion
        self.ConfSeguimiento = ConfSeguimiento
        self.compl = model_complexity
        
        self.mpmanos = mp.solutions.hands  
        self.manos = self.mpmanos.Hands(self.modo, 
                                        self.maxmanos, 
                                        self.compl,
                                        self.Confdeteccion, 
                                        self.ConfSeguimiento 
                                        ) 
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def encontrarmanos(self, frame, dibujar=True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(imgcolor)
        
        if self.resultados.multi_hand_landmarks:
            for manoLms in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.mpDraw.draw_landmarks(frame, manoLms, 
                                               self.mpmanos.HAND_CONNECTIONS)
        return frame
    
    def encontrarposicion(self, frame, manoNumero = 0, 
                          dibujarPuntos = True, 
                          dibujarBox = True, 
                          color = []):
        xlist = []
        ylist = []
        bbox = []
        player = []
        self.lmlist = []
        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[manoNumero]
            prueba = self.resultados.multi_hand_landmarks
            player = len(prueba)
            for id, lm in enumerate(miMano.landmark):
                h, w, c = frame.shape #Hight, Width, Channels
                cx, cy = int(lm.x * w), int(lm.y * h)
                xlist.append(cx)
                ylist.append(cy)
                self.lmlist.append([id, cx, cy])
                if dibujarPuntos:
                    cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax
            if dibujarBox:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), 
                              (xmax + 20, ymax + 20), color, 2)
        return self.lmlist, bbox, player
    
    def dedosarriba(self):
        dedos = []
        # Pulgar
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0]-1][1]:
            dedos.append(1)
        else:
            dedos.append(0)
        # 4 dedos
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id]-2][2]:
                dedos.append(1)
            else:
                dedos.append(0)
        return dedos
    
    def distancia(self, p1, p2, frame, dibujar = True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if dibujar:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]
    
def main():
    ptiempo = 0
    ctiempo = 0
    cap = cv2.VideoCapture(0)  
    detector = detectormanos()
    
    while True:
        ret, frame = cap.read()
        frame = detector.encontrarmanos(frame)
        lista, bbox, = detector.encontrarposicion(frame)
        if len(lista) != 0:
            print(lista[4])
        ctiempo = time.time()
        fps = 1 / (ctiempo-ptiempo)
        ptiempo = ctiempo
        
        cv2.putText(frame, str(int(fps)), (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, 
                    (255, 0, 255), 3)
        
        cv2.imshow('Manos', frame)
        k = cv2.waitKey(1)  
        
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows() 