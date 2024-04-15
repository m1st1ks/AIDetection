import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

data_dir = 'YourBaseName'  # Название директории с классами из фотографий (только .png)

image_size = (224, 224)  # Размеры изображений (Стандарт для TensorFlow)

num_classes = 5  # Количество классов в базе фотографий

batch_size = 32
epochs = 10  # Количество эпох - 10-20

# Создание генератора изображений
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

# Загрузка изображений и разделение на тренировочную и валидационную выборки
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Загрузка предобученной модели MobileNet без последнего слоя (top layer)
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Замораживание весов базовой модели
base_model.trainable = False

# Добавление своего классификатора поверх базовой модели
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Создание и сохранение модели в файл .h5 (Формат модели TensorFlow)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

model.save('my_model.h5')

'''
 ██████  ███    ███  ██ ███████ ████████  ██ ██   ██ ██████   ██████   ██████  
██    ██ ████  ████ ███ ██         ██    ███ ██  ██       ██ ██       ██  ████ 
██ ██ ██ ██ ████ ██  ██ ███████    ██     ██ █████    █████  ███████  ██ ██ ██ 
██ ██ ██ ██  ██  ██  ██      ██    ██     ██ ██  ██       ██ ██    ██ ████  ██ 
 █ ████  ██      ██  ██ ███████    ██     ██ ██   ██ ██████   ██████   ██████  
'''