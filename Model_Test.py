import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('Model.h5')

# Define class names
class_names = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (50, 50))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Define a function for prediction
def predict(image_path):
    # Read image file
    image = cv2.imread(image_path)
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make predictions
    predictions = model.predict(processed_image)
    # Get predicted class probability and index
    predicted_class_prob = np.max(predictions)
    predicted_class_index = np.argmax(predictions)
    # Get predicted class name
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name, predicted_class_prob

# Example usage:
image_path = r"C:\Users\burag\Downloads\1253_left.jpg"  # Replace with your image path
predicted_class, predicted_prob = predict(image_path)
print("Predicted Class:", predicted_class)
print("Predicted Probability:", predicted_prob)
