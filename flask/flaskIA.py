from flask import Flask, request, jsonify
from joblib import load
import tensorflow
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Chargement des modèles
scaler = load('scaler.pkl')
lstm_model = load_model('model1.pkl')  # LSTM .h5 ou .pkl selon ton format
decision_tree_model = load('decision_tree_model.pkl')
random_forest_model = load('random_forest_model.pkl')
logistic_model = load('logistic_regression_model.pkl')

@app.route('/predict/lstm', methods=['POST'])
def predict_lstm():
    data = request.get_json()
    input_array = np.array(data['input']).reshape(10, -1)
    input_scaled = scaler.transform(input_array)
    input_reshaped = input_scaled.reshape(1, 10, -1)
    prediction = lstm_model.predict(input_reshaped)
    predicted_class = int(np.argmax(prediction[0]))
    return jsonify({'model': 'LSTM', 'prediction': predicted_class})

@app.route('/predict/tree', methods=['POST'])
def predict_tree():
    data = request.get_json()
    input_array = np.array(data['input']).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = decision_tree_model.predict(input_scaled)
    return jsonify({'model': 'Decision Tree', 'prediction': int(prediction[0])})

@app.route('/predict/rf', methods=['POST'])
def predict_rf():
    data = request.get_json()
    input_array = np.array(data['input']).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = random_forest_model.predict(input_scaled)
    return jsonify({'model': 'Random Forest', 'prediction': int(prediction[0])})

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    data = request.get_json()
    input_array = np.array(data['input']).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = logistic_model.predict(input_scaled)
    return jsonify({'model': 'Logistic Regression', 'prediction': int(prediction[0])})

@app.route('/test', methods=['GET'])
def test():
    return "API opérationnelle 🚀"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
