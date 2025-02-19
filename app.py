from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# 미리 학습해둔 MNIST 모델 로드 (train_model.py에서 "mnist_model.h5" 저장했다고 가정)
model = tf.keras.models.load_model("mnist_model.h5")

# 1. '/' 경로로 요청이 들어오면 index.html을 반환
@app.route('/')
def serve_index():
    # app.py와 index.html이 같은 폴더에 있다고 가정
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

# 2. '/predict' 경로로 POST 요청이 들어오면 예측 수행
@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    pixel_values = req_data.get("data", [])

    # (1) numpy 배열로 만들고 reshape
    pixel_array = np.array(pixel_values).reshape(1, 28, 28, 1)
    # (2) 0~255 범위를 0~1 범위로 스케일링
    pixel_array = pixel_array / 255.0

    # (3) 모델 예측
    preds = model.predict(pixel_array)
    pred_label = int(np.argmax(preds, axis=1)[0])

    return jsonify({"prediction": pred_label})

# Flask 실행
if __name__ == "__main__":
    # 포트 5000에서 debug 모드로 실행 (VESSL에서 포트 5000을 외부로 노출)
    app.run(host="0.0.0.0", port=5000, debug=True)
