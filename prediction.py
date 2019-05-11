import numpy as np
from keras.preprocessing import image
import cv2
import pickle

filename = 'finalized_model.sav'
classifier = pickle.load(open(filename, 'rb'))

def predict_image(img, pred):
    test_image = image.load_img(img, target_size = (64, 64, 3))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)

    prediction = '?'
    if result[0][0] >= 0.5:
        prediction = 'dog'
    else:
        prediction = 'cat'

    if pred == prediction:
        print('\t✓ is a', prediction, result[0][0], img)
    else:
        print('\t✘ is a', prediction, result[0][0], img)

    return result

print('\nPredictions:')
print(' - a cat:')
predict_image('cat.jpg', 'cat')

print(' - a dog:')
predict_image('dog.jpg', 'dog')
