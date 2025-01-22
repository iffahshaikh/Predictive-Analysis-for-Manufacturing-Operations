from flask import Flask, request, jsonify
import pandas as pd
from model import train_model, predict_downtime

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Manufacturing API"


@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']  

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    
    data = pd.read_csv(file)
    
    data.to_csv('synthetic_data.csv', index=False)
    return jsonify({"message": "Data uploaded successfully"}), 200


@app.route('/train', methods=['POST'])
def train():
    metrics = train_model('synthetic_data.csv')
    return jsonify(metrics), 200


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    prediction = predict_downtime(input_data)
    return jsonify(prediction), 200


if __name__ == '__main__':
    app.run(debug=True)
