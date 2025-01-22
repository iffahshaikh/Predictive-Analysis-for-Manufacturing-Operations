# Manufacturing Downtime Prediction API

## Overview
This project is a Flask-based REST API that predicts machine downtime using a machine learning model. The API allows you to upload data, train the model, and make predictions.

## Requirements
- Python 3.7+
- Flask
- scikit-learn
- pandas
- joblib

## Setup Instructions
1. Install the required dependencies:
   ```bash
   pip install flask scikit-learn pandas joblib
Run the Flask app:

2. RUn the Flask app:
python app.py

3. Use cURL or Postman to interact with API.

API Endpoints:
1. POST /upload: Upload a CSV file to be used for model training.
2. POST /train: Train the logistic regression model on the uploaded data.
3. POST /predict: Predict machine downtime based on input features (Temperature, Run_Time).