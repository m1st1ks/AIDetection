import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model = load_model('my_model.h5')  # Название модели
classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5']  # Названия классов по порядку директорий
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Преобразование изображения для совместимости с моделью
    img = cv2.resize(frame, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Предсказание класса объекта
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    class_label = classes[predicted_class]

    # Определение вероятности класса
    confidence = prediction[0][predicted_class] * 100

    # Проверка условия по вероятности
    if confidence >= 20:
        # Тут можно выполнять различные операции, если объект определился с точностью больше confidence %
        # Отображение прогноза на изображении
        cv2.putText(frame, class_label + f' ({confidence:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение изображения с прогнозом
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

'''
 ██████  ███    ███  ██ ███████ ████████  ██ ██   ██ ██████   ██████   ██████  
██    ██ ████  ████ ███ ██         ██    ███ ██  ██       ██ ██       ██  ████ 
██ ██ ██ ██ ████ ██  ██ ███████    ██     ██ █████    █████  ███████  ██ ██ ██ 
██ ██ ██ ██  ██  ██  ██      ██    ██     ██ ██  ██       ██ ██    ██ ████  ██ 
 █ ████  ██      ██  ██ ███████    ██     ██ ██   ██ ██████   ██████   ██████  
'''