import cv2
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('Model.h5')

# Define class names
class_names = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (50, 50))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    # Read image file
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make predictions
    predictions = model.predict(processed_image)
    # Get predicted class probability and index
    predicted_class_prob = np.max(predictions)
    predicted_class_index = np.argmax(predictions)
    # Get predicted class name
    predicted_class_name = class_names[predicted_class_index]
    # Render result template with prediction information
    return render_template('result.html', predicted_class=predicted_class_name, predicted_class_probability=predicted_class_prob)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
