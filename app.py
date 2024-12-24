from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load MNIST model đã huấn luyện
model = load_model('mnist_cnn.h5')

@app.route('/')
def index():
    """Trang chính"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API để dự đoán"""
    file = request.files['image']
    image = Image.open(file).convert('L')  # Chuyển ảnh sang grayscale
    image = image.resize((28, 28))        # Resize ảnh về 28x28
    # image.save('image.png') #Lưu ảnh để xem
    image_array = np.array(image) / 255.0  # Chuẩn hóa pixel
    image_array = image_array.reshape(1, 28, 28, 1)  # Thêm batch dimension

    # Dự đoán kết quả
    prediction = model.predict(image_array)
    predicted_number = np.argmax(prediction)

    return jsonify({'prediction': int(predicted_number),
                    'confidence': f'{prediction[0][predicted_number]*100:.2f}%'})

if __name__ == '__main__':
    app.run(debug=True)
