from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import cv2
import tensorflow.keras.utils as utils

vgg_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)

img1 = utils.load_img("D:\\k-digital\\source\\siamese\candy\\anchor\\s_0700_13607_187916.jpg", target_size=(224, 224))
img1 = utils.img_to_array(img1)
img1 = np.expand_dims(img1, axis=0)
img1 = preprocess_input(img1)
features1 = model.predict(img1)
features1 = features1.flatten()

img2 = utils.load_img("D:\\k-digital\\source\\siamese\\chair\\test\\chair2.jpg", target_size=(224, 224))
img2 = utils.img_to_array(img2)
img2 = np.expand_dims(img2, axis=0)
img2 = preprocess_input(img2)
features2 = model.predict(img2)
features2 = features2.flatten()
cosine_similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
print("Cosine similarity: ", cosine_similarity)