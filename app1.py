from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.layers import RandomRotation # type: ignore
import numpy as np
import io

app = Flask(__name__)

MODEL_PATH = "best_model.keras"
custom_objects = {"RandomRotation": RandomRotation}

# Load the model
model = load_model(MODEL_PATH, custom_objects=custom_objects)

# Image size used during training
IMG_SIZE = (224, 224)

def prepare_image(image_bytes, img_size):
    # image_bytes is file.read() content from the request
    image = load_img(io.BytesIO(image_bytes), target_size=img_size)
    img_array = img_to_array(image)  # Shape: (224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    img_array = img_array / 255.0   # Normalize
    return img_array

@app.route("/")
def index():
    return "Flask API for Image Classification is running!"

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        file_bytes = file.read()
        
        # Prepare the image
        prepared_img = prepare_image(file_bytes, IMG_SIZE)

        # Predict
        prediction = model.predict(prepared_img)  # Assume shape: (1,1)
        probability = float(prediction[0][0])

        # Apply threshold logic
        if probability == 1:
            label = "Malignant"
            confidence = probability
        else:
            label = "Benign"
            confidence = 1 - probability

        return jsonify({
            "label": label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)




