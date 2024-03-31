from multiprocessing import cpu_count
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import tensorflow as tf
from architecture import InceptionResNetV2
from concurrent.futures import ThreadPoolExecutor

######pathsandvairables#########
face_data = 'Faces'
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
path = 'facenet_keras_weights.h5'
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def process_image(image_path):
    global encodes
    img_RGB = cv2.imread(image_path, cv2.IMREAD_COLOR)

    x = face_detector.detect_faces(img_RGB)
    x1, y1, width, height = x[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img_RGB[y1:y2, x1:x2]

    face = normalize(face)
    face = cv2.resize(face, required_shape)
    face_d = np.expand_dims(face, axis=0)
    encode = face_encoder.predict(face_d).flatten()
    encodes.append(encode)

def process_images(person_dir):
    global encodes
    encodes = []
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        image_paths = [os.path.join(person_dir, image_name) for image_name in os.listdir(person_dir)]
        executor.map(process_image, image_paths)

    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[os.path.basename(person_dir)] = encode

def main():
    global encoding_dict
    for face_names in os.listdir(face_data):
        person_dir = os.path.join(face_data, face_names)
        process_images(person_dir)
    
    path = 'encodings\\encodings.pkl'
    with open(path, 'wb') as file:
        pickle.dump(encoding_dict, file)

if __name__ == "__main__":
    main()
