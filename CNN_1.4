from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import save_model
from keras import backend as K
import matplotlib.pyplot as plt 
import numpy as np
from keras_preprocessing import image


img_width = 150
img_height = 150

train_data_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN img classifier kaggle\train'
validation_data_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN img classifier kaggle\validation'

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 10
batch_size = 32

# Resize img if needed
if K.image_data_format() == 'channels_first':
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)

# CNN model
model = Sequential()

model.add(Conv2D(32,3,3, input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.summary()

model.add(Conv2D(32,3,3, input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,3,3, input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# Create train and validation set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

# Create train and validation generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    class_mode='binary')

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

# Train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples/batch_size)

# Save the model to .h5
model = save_model(model,r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN img classifier kaggle\log\model_1.h5',include_optimizer=True)

#Make prediction
img_show = image.load_img(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN img classifier kaggle\test\dogs\9994.jpg')
img_prediction = image.load_img(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN img classifier kaggle\test\dogs\9994.jpg',target_size=(150,150))
img_prediction = image.img_to_array(img_prediction)
img_prediction = np.expand_dims(img_prediction,axis=0)

result = model.predict(img_prediction)
print(result)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediciton ='cat'
print(prediction)

plt.imshow(img_show)
plt.show()
