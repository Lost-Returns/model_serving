from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from mobilenet_module import load_model # 모델 로드 함수 import
from flask_cors import CORS
from io import BytesIO

app = Flask(__name__)
CORS(app)
model = load_model() # 모델 로드

@app.route('/')
def hello_world():
    return "hello world"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = request.files['image'] # 이미지 파일을 받도록 수정
        img = Image.open(image)
        img = img.resize((224, 224))  # 모델에 맞는 크기로 조정
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

        
        # 이미지 처리 및 예측
        prediction = model.predict(img_array)
        
        # 넘파이 배열을 리스트로 변환하여 JSON으로 전송
        prediction_list = prediction.tolist()
        
        return jsonify({'prediction': prediction_list})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
