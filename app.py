from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('models/asl_gesture_model.h5')

# Define the class labels (ASL alphabet)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data.get('image')

    # Decode base64 image data
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(BytesIO(img_data))

    # Preprocess the image
    img = img.resize((64, 64))  # Resize to the input shape of the model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the gesture
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Get the predicted letter
    predicted_letter = class_labels[predicted_class[0]]

    return jsonify({'predicted_letter': predicted_letter})

if __name__ == '__main__':
    app.run(debug=True)
