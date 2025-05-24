from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import joblib
import mysql.connector
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MySQL Database Configuration
DB_CONFIG = {
    "host": "sql12.freesqldatabase.com",  
    "user": "sql12780989",               
    "password": "WmfMiyJMPz",           
    "database": "sql12780989",         
    "port": 3306                        
}

# Load LSTM Model
try:
    model = tf.keras.models.load_model("app/model/lstm_model.keras")
except Exception as e:
    print(f"Error loading LSTM model: {e}")

# Load GRU Models
try:
    model1 = tf.saved_model.load("app/model/gru_model1")
    model2 = tf.saved_model.load("app/model/gru_model2")
except Exception as e:
    print(f"Error loading GRU models: {e}")

# Load Scalers
try:
    scaler_features = pickle.load(open("app/utils/scaler_features.pkl", "rb"))
    scaler_targets = pickle.load(open("app/utils/scaler_targets.pkl", "rb"))
    scaler_X1 = joblib.load("app/utils/scaler_X1.pkl")
    scaler_y1 = joblib.load("app/utils/scaler_y1.pkl")
    scaler_X2 = joblib.load("app/utils/scaler_X2.pkl")
    scaler_y2 = joblib.load("app/utils/scaler_y2.pkl")
except Exception as e:
    print(f"Error loading scalers: {e}")

# Function to get DB connection
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

@app.route("/predicts", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "input" not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        
        input_data = np.array(data["input"])
        scaled_input = scaler_features.transform(input_data).reshape(1, 5, 4)
        prediction_scaled = np.maximum(model.predict(scaled_input), 0)
        prediction = scaler_targets.inverse_transform(prediction_scaled)

        response = {
            "chlorine_usage": float(prediction[0][0]),
            "pac_usage": float(prediction[0][1]),
            "lime_usage": float(prediction[0][2])
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/future', methods=['POST'])
def predict_future():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM dailyPrediction ORDER BY id DESC LIMIT 4")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify({"error": "No data found in database"}), 404

        formatted_data = [[row["Turbidity"], row["PH"], row["Conductivity"], row["Water_production"]] for row in reversed(rows)]
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input data"}), 400
        
        sample_input1 = np.array([[data["day"], data["month"], data["year"], 
                                   data["temp_c"], data["humidity"], 
                                   data["precip_mm"], data["cloud"]]])

        sample_input1_scaled = scaler_X1.transform(sample_input1).reshape((1, 1, sample_input1.shape[1]))
        predicted_water_scaled = model1.signatures["serving_default"](tf.constant(sample_input1_scaled, dtype=tf.float32))["output_0"]
        predicted_water = scaler_y1.inverse_transform(predicted_water_scaled.numpy())[0][0]

        sample_input2 = np.array([[predicted_water]])
        sample_input2_scaled = scaler_X2.transform(sample_input2).reshape((1, 1, sample_input2.shape[1]))
        predicted_params_scaled = model2.signatures["serving_default"](tf.constant(sample_input2_scaled, dtype=tf.float32))["output_0"]
        predicted_params = scaler_y2.inverse_transform(predicted_params_scaled.numpy())[0]

        predicted_response = {
            "predicted_water_production": float(predicted_water),
            "predicted_turbidity": float(predicted_params[0]),
            "predicted_ph": float(predicted_params[1]),
            "predicted_conductivity": float(predicted_params[2])
        }

        formatted_data.append([
            predicted_response["predicted_turbidity"],
            predicted_response["predicted_ph"],
            predicted_response["predicted_conductivity"],
            predicted_response["predicted_water_production"]
        ])

        data_input = np.array(formatted_data)
        scaled_input = scaler_features.transform(data_input).reshape(1, 5, 4)
        prediction_scaled = np.maximum(model.predict(scaled_input), 0)
        prediction = scaler_targets.inverse_transform(prediction_scaled)

        final_response = {
            **predicted_response,
            "chlorine_usage": float(prediction[0][0]),
            "pac_usage": float(prediction[0][1]),
            "lime_usage": float(prediction[0][2])
        }

        return jsonify(final_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dailyusage', methods=['GET'])
def get_daily_usage():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM dailyPrediction ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify({"error": "No data found"}), 404

        formatted_data = {"input": [[row["Turbidity"], row["PH"], row["Conductivity"], row["Water_production"]] for row in reversed(rows)]}
        
        data = formatted_data["input"]
        scaled_input = scaler_features.transform(data).reshape(1, 5, 4)
        prediction_scaled = np.maximum(model.predict(scaled_input), 0)
        prediction = scaler_targets.inverse_transform(prediction_scaled)

        response = {
            "chlorine_usage": float(prediction[0][0]),
            "pac_usage": float(prediction[0][1]),
            "lime_usage": float(prediction[0][2])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/lastusage', methods=['GET'])
def get_last_usage():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM dailyPrediction ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify({"error": "No data found"}), 404

        return jsonify(rows)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
