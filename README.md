# 🧬 Breast Cancer Detection Web App - Flask

## 📖 Introduction ::

This is a Flask-based web application that classifies breast cancer images into Benign or Malignant categories using a pre-trained deep learning model.
It provides a simple web interface where users can upload images and receive instant predictions.

## 🚀 Features ::

* 📂 Upload breast cancer medical images. 
 
* 🧠 Predict tumor type using a deep learning model.
 
* 📈 Instant results displayed on the screen.
 
* 🖥️ Clean and responsive web interface.

# 🛠️ Tech Stack

* Python 3.11.1

* Flask (Web Framework)

* TensorFlow / Keras (Deep Learning Model)

# 📂 Project Structure

```

Flask-App/
├── app.py              # Main Flask application
├── app1.py             # Alternative/backup app
├── best_model.h5       # Pre-trained model (.h5 format)
├── best_model.keras    # Pre-trained model (.keras format)
├── Benign.jpg          # Example benign tumor image
├── Malignant.jpg       # Example malignant tumor image
├── Malignant1.jpg      # Another malignant sample
├── Malignant2.jpg      # Another malignant sample
└── __pycache__/        # Compiled Python files

```

# 📲 Getting Started

## 1. Clone the Repository 

```
git clone https://github.com/ibrahimjamil301/Flask-App.git
cd Flask-App

```

## 2. Install Dependencies
If you don't have a requirements.txt file, manually install:

```
pip install Flask tensorflow keras pillow numpy

```

## 4. Run the Application

```
python app.py

```

Then open your browser and go to:
👉 http://127.0.0.1:5000/predict

## 🧩 How It Works

1. The user uploads a breast cancer image through the web page.

2. The server loads the image and preprocesses it.

3. The image is passed into the pre-trained deep learning model.

4. The model returns a prediction: Benign or Malignant.

5. The result is displayed instantly to the user.

## 🧠 Contribution

* Pull requests are welcome! Feel free to open issues for bugs or feature requests

## ✅ Quick Checklist

| Feature                                   | Status |
| ------------------------------------------| ------ |
| Upload Image                              | ✅     |
| Predict Benign/Malignant                  | ✅     |
| Pre-trained Model                         | ✅     |
| Web App Running Locally                   | ✅     |

## 📃 License

This project is open-source and available under the MIT License









