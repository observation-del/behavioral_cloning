import numpy as np
import csv
import cv2
from scipy import ndimage

lines  = []
with open('./data/driving_log.csv') as csvfile:
    header = next(csv.reader(csvfile))
    reader = csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    # create adjusted steering measurements for the side camera images
    correction = 0.2
    measurement_center = float(line[3])
    measurement_left = measurement_center + 0.2
    measurement_right = measurement_center - 0.2
    measurement_tmp = [measurement_center, measurement_left, measurement_right]
    
    source_path_center = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    filename_center = source_path_center.split('/')[-1]
    filename_left = source_path_left.split('/')[-1]
    filename_right = source_path_right.split('/')[-1]
    path_center = './data/IMG/' + filename_center
    path_left = './data/IMG/' + filename_left
    path_right = './data/IMG/' + filename_right

    # image = cv2.imread(current_path)
    image_center = ndimage.imread(path_center)
    image_left = ndimage.imread(path_left)
    image_right = ndimage.imread(path_right)
    image_tmp = [image_center, image_left, image_right]

    images.extend(image_tmp)
    measurements.extend(measurement_tmp)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')