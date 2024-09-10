import os

import tensorflow as tf
import numpy as np
import pickle


def compute(filename):
    labels = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma weed', 'Badipala', 'Balloon Vine', 'Bamboo', 'Beans',
              'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly',
              'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre',
              'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava',
              'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi',
              'Lantana', 'Lemon', 'Lemongrass', 'Malabar Nut', 'Malabar Spinach', 'Mango', 'Marigold', 'Mint', 'Neem',
              'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak', 'Papaya', 'Parijatha', 'Pea',
              'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala',
              'Spinach', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor',
              'kamakasturi', 'kepala']

    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # model = tf.keras.models.load_model("my_model.keras")
    path = f'C:\\Predict\\{filename}'
    img = tf.keras.preprocessing.image.load_img(
        path, target_size=(299, 299)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])
    # return (
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(labels[np.argmax(score)], 100 * np.max(score))
    # )
    return labels[np.argmax(score)]
