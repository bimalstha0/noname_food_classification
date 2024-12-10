import tensorflow as tf

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img

model_mobile_net= load_model('models/best_mobile_net_v2.keras')

def preprocess_new_image(image_path, input_shape):
    """
    Preprocesses a new image the same way as the training data.

    Args:
    - image_path (str): Path to the image file to preprocess.
    - input_shape (tuple): Target size (height, width) for resizing the image.

    Returns:
    - np.array: Preprocessed image ready for prediction.
    """
    # Load the image
    img = load_img(image_path, target_size=input_shape)
    
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    
    # Expand dimensions to match the batch shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image using MobileNetV2 preprocessing function
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_array

def predict_image(model, image_path, input_shape):
    """
    Predicts the class of a new image using the trained model.

    Args:
    - model (tf.keras.Model): The trained model for prediction.
    - image_path (str): Path to the image to predict.
    - input_shape (tuple): Target size (height, width) for resizing the image.

    Returns:
    - str: Predicted class label.
    """
    # Preprocess the image
    preprocessed_image = preprocess_new_image(image_path, input_shape)

    # Predict the class probabilities
    predictions = model.predict(preprocessed_image)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Get the class labels (if available in the model's class names)
    class_labels = list(model.class_names)  
    
    # Get the predicted label from the class labels
    predicted_label = class_labels[predicted_class_index]

    return predicted_label