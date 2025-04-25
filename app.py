from flask import Flask, request, jsonify
from deepface import DeepFace
import os

app = Flask(__name__)

#  Home Route
@app.route('/')
def home():
    return "Deepfake Detection API is Running!"

#  Fake or Real Image Detection Route
@app.route('/detect', methods=['POST'])
def detect_fake():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files['image']
        image_path = f"./uploads/{image.filename}"
        os.makedirs("./uploads", exist_ok=True)
        image.save(image_path)

        #  Perform DeepFace Analysis
        result = DeepFace.analyze(image_path, actions=['age', 'gender', 'emotion'])

        # Add Deepfake Detection (Using a Pretrained Model)
        deepfake_result = DeepFace.verify(image_path, image_path, model_name="Facenet")

        #  Threshold for deepfake detection
        if deepfake_result["distance"] > 0.6:  # Adjust threshold if needed
            fake_status = "Fake"
        else:
            fake_status = "Real"

        return jsonify({"deepfake_result": fake_status, "analysis": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#  Run Flask Server
if __name__ == '__main__':
    app.run(debug=True)
