import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

data_augmentation1 = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])



def data_augmentation(dataset):
    new_dataset = dataset.map(lambda x, y: (data_augmentation1(x), y))
    return new_dataset

def test_preprocess(dataset):
    new_dataset = dataset.map(lambda x, y : (resize_and_rescale(x), y))
    return new_dataset
    

def cnn_model(n_classes):
    global input_shape
    global resize_and_rescale
    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.build(input_shape=input_shape)
    return model
    
    
    
# def cnn_model(n_classes):
#     inputs = keras.Input(shape=(256,256,3))
#     x = layers.Conv2D(32, kernel_size = (3,3), activation='relu')(inputs),
#     x = layers.MaxPooling2D((2, 2))(x),
#     x = layers.Conv2D(64,  kernel_size = (3,3), activation='relu')(x),
#     x = layers.MaxPooling2D((2, 2))(x),
#     x = layers.Conv2D(64,  kernel_size = (3,3), activation='relu')(x),
#     x = layers.MaxPooling2D((2, 2))(x),
#     x = layers.Conv2D(64, (3, 3), activation='relu')(x),
#     x = layers.MaxPooling2D((2, 2))(x),
#     x = layers.Conv2D(64, (3, 3), activation='relu')(x),
#     x = layers.MaxPooling2D((2, 2))(x),
#     x = layers.Conv2D(64, (3, 3), activation='relu')(x),
#     x = layers.MaxPooling2D((2, 2))(x),
#     x = layers.Flatten()(x),
#     x = layers.Dense(64, activation='relu')(x),
#     outputs = layers.Dense(n_classes, activation='softmax')(x),
#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model




    