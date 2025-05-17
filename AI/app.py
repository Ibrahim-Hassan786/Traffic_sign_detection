from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

try:
    # Try loading with tf.keras first
    model = tf.keras.models.load_model('traffic_sign_classifier.h5', compile=False)
except Exception as e:
    print(f"First loading attempt failed: {e}")
    try:
        # Try alternative loading method
        model = tf.saved_model.load('traffic_sign_classifier.h5')
    except Exception as e:
        print(f"Second loading attempt failed: {e}")
        # Last resort: try loading with custom object scope
        from tensorflow.keras.models import load_model
        custom_objects = {'CustomLayer': tf.keras.layers.Layer}
        model = load_model('traffic_sign_classifier.h5', custom_objects=custom_objects, compile=False)

# Traffic sign class names - update these based on your model's classes
CLASSES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
    'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at intersection', 
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles > 3.5 tons prohibited', 
    'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right', 
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
    'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 
    'Keep left', 'Roundabout mandatory', 'End of no passing', 
    'End no passing vehicle > 3.5 tons'
]

def preprocess_image(image):
    # Resize image to 32x32
    image = image.resize((32, 32))
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the POST request
        file = request.files['image']
        # Read the image
        image = Image.open(file.stream)
        
        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        predicted_label = CLASSES[predicted_class]
        confidence = float(predictions[0][predicted_class])
        
        return jsonify({
            'success': True,
            'prediction': predicted_label,
            'confidence': f'{confidence:.2%}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 