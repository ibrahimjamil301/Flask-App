from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
import io

app = Flask(__name__)

# Load your trained model
model = load_model("best_model.keras")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if not file:
            return jsonify({"error": "Invalid file"}), 400

        # Read the file as a file-like object
        file_stream = io.BytesIO(file.read())

        # Load and preprocess the image
        img = load_img(file_stream, target_size=(128, 128))  # Resize to 128x128
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        prediction = model.predict(img_array)
        predicted_class = int(prediction[0][0] > 0.5)  # Binary classification

        return jsonify({
            "prediction": "Benign" if predicted_class == 0 else "Malignant",
            "confidence": float(prediction[0][0])
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5000, debug=True)