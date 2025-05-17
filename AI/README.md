# Traffic Sign Classifier Web App

A simple web application that uses a trained Keras model to classify traffic signs from uploaded PNG images.

## Features

- Upload PNG images through drag & drop or file browser
- Real-time image preview
- Displays prediction and confidence score
- Modern, responsive UI
- Supports 32x32 PNG images (automatically resized)

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your trained model file `traffic_sign_classifier.h5` is in the root directory.

3. Run the Flask application:
```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click the upload area or drag & drop a PNG image of a traffic sign
2. The image will be displayed in the preview area
3. The model's prediction and confidence score will appear below the image

## Project Structure

```
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── traffic_sign_classifier.h5  # Trained model
├── static/
│   └── style.css         # CSS styles
└── templates/
    └── index.html        # HTML template
```

## Notes

- The model expects 32x32 RGB images
- Only PNG format is supported
- Images are automatically resized to 32x32 pixels
- The model predicts 43 different traffic sign classes 