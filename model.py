import numpy as np
import os
import csv
import cv2
import sklearn
from math import ceil
from scipy import ndimage

lines  = []
with open('./my_data/driving_log.csv') as csvfile:
    header = next(csv.reader(csvfile))
    reader = csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(lines)

def generator(samples, batch_size):
    num_sumples = len(samples)
    correction = 0.4
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_sumples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                measurement_center = float(line[3])
                measurement_left = measurement_center + correction
                measurement_right = measurement_center - correction
                measurement_tmp = [measurement_center, measurement_left, measurement_right]
                
                source_path_center = line[0]
                source_path_left = line[1]
                source_path_right = line[2]
                filename_center = source_path_center.split('/')[-1]
                filename_left = source_path_left.split('/')[-1]
                filename_right = source_path_right.split('/')[-1]
                path_center = './my_data/IMG/' + filename_center
                path_left = './my_data/IMG/' + filename_left
                path_right = './my_data/IMG/' + filename_right

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

            yield shuffle(X_train, y_train)


# images = []
# measurements = []
# for line in lines:
#     # create adjusted steering measurements for the side camera images
#     correction = 0.2
#     measurement_center = float(line[3])
#     measurement_left = measurement_center + 0.2
#     measurement_right = measurement_center - 0.2
#     measurement_tmp = [measurement_center, measurement_left, measurement_right]
    
#     source_path_center = line[0]
#     source_path_left = line[1]
#     source_path_right = line[2]
#     filename_center = source_path_center.split('/')[-1]
#     filename_left = source_path_left.split('/')[-1]
#     filename_right = source_path_right.split('/')[-1]
#     path_center = './data/IMG/' + filename_center
#     path_left = './data/IMG/' + filename_left
#     path_right = './data/IMG/' + filename_right

#     # image = cv2.imread(current_path)
#     image_center = ndimage.imread(path_center)
#     image_left = ndimage.imread(path_left)
#     image_right = ndimage.imread(path_right)
#     image_tmp = [image_center, image_left, image_right]

#     images.extend(image_tmp)
#     measurements.extend(measurement_tmp)

# augmented_images, augmented_measurements = [], []
# for image, measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flip(image,1))
#     augmented_measurements.append(measurement*-1.0)

# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)

# Set the batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
# model.add(Lambda(lambda x: x / 127.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(128,2,2,activation="relu"))
model.add(Convolution2D(128,2,2,activation="relu"))
model.add(MaxPooling2D(pool_size=1, strides=None, padding='valid'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/batch_size),
            epochs=10, verbose=1)

model.save('model.h5')