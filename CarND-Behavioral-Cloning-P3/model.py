import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def readfile():
	lines = []
	with open('data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	return lines

def collectimages(lines):
	images = []
	measurements = []
	for line in lines:
		steering_center = float(line[3])
		correction = 0.2
		steering_left = steering_center + correction
		steering_right = steering_center - correction

		img_center = cv2.imread('/Users/richardkong/Desktop/Udacity/Self-Driving Cars/CarND-Behavioral-Cloning-P3/data/' + line[0])
		img_left = cv2.imread('/Users/richardkong/Desktop/Udacity/Self-Driving Cars/CarND-Behavioral-Cloning-P3/data/' + line[1])
		img_right = cv2.imread('/Users/richardkong/Desktop/Udacity/Self-Driving Cars/CarND-Behavioral-Cloning-P3/data/' + line[2])

		images.append(img_center)
		images.append(img_left)
		images.append(img_right)
		measurements.append(steering_center)
		measurements.append(steering_left)
		measurements.append(steering_right)

	return list(zip(images, measurements))

def generator(data, batch_size = 40):
	length = len(data)
	while True: # Loop forever so the generator never terminates
		data = shuffle(data)
		for offset in range(0, length, batch_size):
			batch_data = data[offset: offset + batch_size]

			augmented_images = []
			augmented_measurements = []
			for image, measurement in batch_data:
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				augmented_images.append(cv2.flip(image,1))
				augmented_measurements.append(measurement*-1.0)

			inputs = np.array(augmented_images)
			outputs = np.array(augmented_measurements)
			yield shuffle(inputs, outputs)

def trainingmodel():
	model = Sequential()
	# normalizing the data and mean centering the data
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((60,25),(0,0))))
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(64, 3, 3, activation="relu"))
	model.add(Convolution2D(64, 3, 3, activation="relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	return model

files = readfile()
data = collectimages(files)
train_data, validation_data = train_test_split(data, test_size=0.2)
print('Train Data:', len(train_data))
print('Validation Data:', len(validation_data)

train_generator = generator(train_data)
validation_generator = generator(validation_data)

model = trainingmodel()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_data), \
	validation_data = validation_generator, nb_val_samples = len(validation_data), nb_epoch=5, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


exit()