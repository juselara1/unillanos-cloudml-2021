import json, cv2, joblib
from flask import Response, jsonify, Flask, request
import numpy as np

app = Flask(__name__)
model = joblib.load("data/model.joblib")

@app.route("/predict", methods=["POST"])
def model_prediction():
    """
    Esta API colorea una imagen con clusters obtenidos con KMeans.
    """
    data = request.files

    img = np.frombuffer(data["image"].read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR).astype(np.float32) / 255
    X = img.reshape(-1, 1) 
    preds = model.predict(X)
    print(np.unique(preds))
    X_map = preds.reshape(img.shape)

    new_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    for i in range(3):
        new_img[..., i] = (X_map == i).astype(np.uint8) * 255
    _, img_encoded = cv2.imencode(".jpg", new_img)

    return img_encoded.tobytes()

if __name__ == "__main__":
    app.run()
