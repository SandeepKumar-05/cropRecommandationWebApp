from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pickle
from flask_cors import CORS  
import os

app = Flask(__name__)
CORS(app)

# Use relative paths assuming model files are inside ./model/
model = pickle.load(open(r'C:\Users\Sandeep Kumar\OneDrive\Pictures\Documents\OneDrive\Pictures\Documents\Projects\AI ML PROJECTS\CropRecommandationWebApp\flask_server\model\model.pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\Sandeep Kumar\OneDrive\Pictures\Documents\OneDrive\Pictures\Documents\Projects\AI ML PROJECTS\CropRecommandationWebApp\flask_server\model\scaler.pkl', 'rb'))
label_encoder = pickle.load(open(r'C:\Users\Sandeep Kumar\OneDrive\Pictures\Documents\OneDrive\Pictures\Documents\Projects\AI ML PROJECTS\CropRecommandationWebApp\flask_server\model\label_encoder.pkl', 'rb'))

# Serve the frontend (React build) from root URL
@app.route('/predict')

# API endpoint for crop prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract and transform input features
    features = np.array([
        data['Nitrogen'], data['Phosphorus'], data['Potassium'],
        data['Temperature'], data['Humidity'],
        data['ph'], data['Rainfall']
    ]).reshape(1, -1)

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    crop = label_encoder.inverse_transform(prediction)[0]

    return jsonify({'predicted_crop': crop})

# Default port and host for Render and local dev
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
