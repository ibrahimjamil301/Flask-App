from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image # type: ignore
import numpy as np
import io
#from my_custom_module import CustomLayer # type: ignore

# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل الموديل
MODEL_PATH = 'efficientnetv2B1 (1).keras'  # المسار إلى ملف الموديل
model = load_model(MODEL_PATH)

# إعداد معلمات الصورة
IMAGE_SIZE = (224, 224)  # تعديل حجم الصورة حسب ما يتوقعه النموذج

# دالة التنبؤ
def predict(image):
    # معالجة الصورة لتناسب المدخلات المتوقعة من النموذج
    image = image.resize(IMAGE_SIZE)  # تغيير حجم الصورة
    image_array = np.array(image) / 255.0  # تحويل إلى مصفوفة وتطبيع القيم
    image_array = np.expand_dims(image_array, axis=0)  # إضافة بعد جديد
    prediction = model.predict(image_array)  # تنبؤ باستخدام الموديل
    result = np.argmax(prediction)  # اختيار التصنيف الأكثر احتمالاً
    return "Malignant" if result == 1 else "Benign"  # إعادة النتيجة النصية

# نقطة النهاية الرئيسية
@app.route('/')
def index():
    return "Flask API for Image Classification is running!"

# نقطة النهاية للتنبؤ
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # التحقق من وجود ملف في الطلب
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        # قراءة الملف
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))  # قراءة الصورة

        # التنبؤ باستخدام الموديل
        result = predict(image)

        # إعادة النتيجة
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)
