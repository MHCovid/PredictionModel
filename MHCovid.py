import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
tf.executing_eagerly()

class MHCovid:
  def __init__(self):
    print('MHCovid at your service!')

  def generateModel(self, path=None):
    # create the base pre-trained model
    base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # add a global spatial average pooling layer
    x = base_model.output
    x = tf.keras.layers.AveragePooling2D()(x)
    # let's add a fully-connected layer
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    # this is the model we will train
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    #model.summary()
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    #print('here')
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
  
  def show(self, model, images, labels, r=4, c=4):
    covid_indexes = [i for i in range(images.shape[0]) if labels[i]==1]
    uncovid_indexes = [i for i in range(images.shape[0]) if labels[i]==0]
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Mohammad Hossein Amini (mhamini@aut.ac.ir)', fontsize = 28)
    self.label = {0:'No :)', 1:'Yes'}
    for i in range(r):
      for j in range(c):
        plt.subplot(r, c, i*r + j + 1)
        plt.axis('off')
        if ((r*i+j)%2):
          index = covid_indexes[np.random.randint(len(covid_indexes))]
          output = round(float(model(images[index:index+1])[0][0]), 3)
          plt.imshow(images[index])
          plt.title(f'COVID-19: {self.label[int(labels[index])]} ,    Prediction: {output}')
        else:   
          index = uncovid_indexes[np.random.randint(len(uncovid_indexes))]
          output = round(float(model(images[index:index+1])[0][0]), 3)
          plt.imshow(images[index])
          plt.title(f'COVID-19: {self.label[int(labels[index])]} ,    Prediction: {output}')
    
    fig.savefig('output.png')
    plt.show()
  
  def predict(self, model, imagefile, show=True):
    ##  Raw image assumption...
    image = cv2.resize(cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB), (224, 224))/255.
    output = round(float(model(np.array([image], dtype=np.float32))[0][0]), 3)
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(image)
    plt.title(f'Mohammad Hossein Amini (mhamini@aut.ac.ir)\n Predicted: {output}')
    return output

if __name__ == '__main__':
  mh = MHCovid()
  model = mh.generateModel()
  #mh.show(model, images_arr_test, labels_arr_test, 1, 2)
  print(mh.predict(model, 'covid1.jpeg'))