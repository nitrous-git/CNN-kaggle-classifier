
import matplotlib.pyplot as plt 
from keras.models import load_model
import numpy as np
from keras_preprocessing import image

# load trained weights
model = load_model(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN img classifier kaggle\log\model_1.h5')

# model compile
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

# Make prediction
img_show = image.load_img(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN img classifier kaggle\test\cats\9962.jpg')
img_prediction = image.load_img(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN img classifier kaggle\test\cats\9962.jpg',target_size=(150,150))
img_prediction = image.img_to_array(img_prediction)
img_prediction = np.expand_dims(img_prediction,axis=0)

result = model.predict(img_prediction)
print(result)
if result[0][0] < 1:
    prediction = 'cat'
else:
    prediciton ='dog'
print(prediction)

plt.imshow(img_show)
plt.show()
