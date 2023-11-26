## **Cat and Dog Image Classifier**

This project is a cat and dog image classifier built using TensorFlow and Keras. The classifier can be used to identify whether an image contains a cat or a dog.

## Dataset

The dataset used to train the classifier contains 25,000 images of cats and dogs, split into 20,000 training images and 5,000 test images. The images were collected from the internet and labeled manually.

## Model Architecture

The classifier is a convolutional neural network (CNN) with the following architecture:

* Input layer: 224x224x3 RGB image
* Convolutional layer: 32 filters, 3x3 kernel size, stride 1, padding same
* Max pooling layer: 2x2 pool size, stride 2
* Convolutional layer: 64 filters, 3x3 kernel size, stride 1, padding same
* Max pooling layer: 2x2 pool size, stride 2
* Flatten layer
* Dense layer: 128 units, ReLU activation
* Dropout layer: 50% dropout rate
* Dense layer: 1 unit, Sigmoid activation

## Training

The classifier was trained using the following hyperparameters:

* Optimizer: Adam
* Loss function: Binary crossentropy
* Learning rate: 0.001
* Batch size: 32
* Number of epochs: 10

## Evaluation

The classifier achieved an accuracy of 95% on the test set.

## Usage

To use the classifier, simply load the saved model and then use it to make predictions:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('cat_dog_classifier.h5')

# Make a prediction
image = tf.keras.preprocessing.image.load_img('image.jpg', target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)
image = tf.expand_dims(image, axis=0)

prediction = model.predict(image)

if prediction > 0.5:
    print('Cat')
else:
    print('Dog')
```

Thank You.
