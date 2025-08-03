import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

application = Flask(__name__)

# Load models and scaler with error handling
try:
    ridge_model = pickle.load(open('ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('scaler.pkl', 'rb'))
    logger.info("Models loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    ridge_model = None
    standard_scaler = None
except Exception as e:
    logger.error(f"Error loading models: {e}")
    ridge_model = None
    standard_scaler = None

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    if request.method == 'POST':
        try:
            # Check if models are loaded
            if ridge_model is None or standard_scaler is None:
                return jsonify({'error': 'Models not loaded properly'}), 500
            
            # Extract form data with validation
            temperature = float(request.form.get('Temperature', 0))
            rh = float(request.form.get('RH', 0))
            ws = float(request.form.get('Ws', 0))
            rain = float(request.form.get('Rain', 0))
            ffmc = float(request.form.get('FFMC', 0))
            dmc = float(request.form.get('DMC', 0))
            isi = float(request.form.get('ISI', 0))
            classes = float(request.form.get('Classes', 0))
            region = float(request.form.get('Region', 0))
            
            # Create feature array
            features = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
            
            # Scale the features
            new_data_scaled = standard_scaler.transform(features)
            
            # Make prediction
            result = ridge_model.predict(new_data_scaled)
            
            # Round the result to 2 decimal places for better display
            rounded_result = round(float(result[0]), 2)
            
            logger.info(f"Prediction made: {rounded_result}")
            
            return render_template('home.html', results=rounded_result)
            
        except ValueError as e:
            logger.error(f"Invalid input data: {e}")
            return render_template('home.html', error="Please enter valid numeric values for all fields")
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return render_template('home.html', error="An error occurred during prediction")

@application.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        if ridge_model is None or standard_scaler is None:
            return jsonify({'error': 'Models not loaded properly'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract required fields
        required_fields = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create feature array
        features = np.array([[
            float(data['Temperature']),
            float(data['RH']),
            float(data['Ws']),
            float(data['Rain']),
            float(data['FFMC']),
            float(data['DMC']),
            float(data['ISI']),
            float(data['Classes']),
            float(data['Region'])
        ]])
        
        # Scale and predict
        new_data_scaled = standard_scaler.transform(features)
        result = ridge_model.predict(new_data_scaled)
        
        return jsonify({
            'prediction': float(result[0]),
            'status': 'success'
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@application.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = 'loaded' if ridge_model is not None and standard_scaler is not None else 'not_loaded'
    return jsonify({
        'status': 'healthy',
        'models': model_status
    })

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=8080, debug=True)