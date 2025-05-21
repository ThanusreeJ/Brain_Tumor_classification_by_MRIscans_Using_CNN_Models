from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model('model/Tumor_Xception_model.h5')
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    pred = model.predict(img)
    result = class_names[np.argmax(pred)]

    return jsonify({'prediction': result})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
