from cvzone.PoseModule import PoseDetector
import cv2
import cvzone 
import tensorflow as tf
import numpy as np

# Carregar o modelo treinado
model = tf.keras.models.load_model('quedas.keras')

video = cv2.VideoCapture('vd01.mp4')
detector = PoseDetector()

while True:
    check, img = video.read()
    img = cv2.resize(img, (1280, 720))
    result = detector.findPose(img)
    points, bbox = detector.findPosition(img, draw=False)
    if len(points) >= 1:
        x, y, w, h = bbox['bbox']
        head = points[0][1]
        knee = points[26][1]
        difference = knee - head
        if difference <= 0:
            # Pré-processamento da imagem, se necessário
            processed_img = cv2.resize(img, (128, 128)) / 255.0
            # Realizar a previsão com o modelo carregado
            prediction = model.predict(np.expand_dims(processed_img, axis=0))
            
            if prediction[0][0] > 0.5:  # Exemplo de threshold para determinar queda
                cvzone.putTextRect(img, 'QUEDA DETECTADA', (x, y-80), scale=3, thickness=3, colorR=(0,0,255))
    
    cv2.imshow('IMG', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

cv2.destroyAllWindows()
video.release()
    
