import os
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("mnist_model.h5")

@app.route('/')
def serve_index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    pixel_values = req_data.get("data", [])
    pixel_array = np.array(pixel_values).reshape(1, 28, 28, 1)
    pixel_array = pixel_array / 255.0
    preds = model.predict(pixel_array)
    pred_label = int(np.argmax(preds, axis=1)[0])
    return jsonify({"prediction": pred_label})

if __name__ == "__main__":
    # Render는 PORT라는 환경 변수를 통해 포트 번호를 넘겨줍니다.
    # 해당 값이 없으면 기본값(5000)으로 설정
    port = int(os.environ.get("PORT", 5000))
    
    # production 환경이라면 debug=False 설정을 권장
    app.run(host="0.0.0.0", port=port, debug=False)

