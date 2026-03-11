#!/usr/bin/env python
"""
Flask API for revenue prediction model
Provides endpoints for training, prediction, and log retrieval
"""

import os
import sys
import json
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, jsonify
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_ingestion import fetch_data, convert_to_ts
from feature_engineering import engineer_features
from logger import update_train_log, update_predict_log, get_train_log, get_predict_log

# Setup Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model storage
MODEL_DIR = "../models"
MODEL_VERSION = "1.0"
MODEL_TAG = "revenue_forecaster"

def load_model():
    """Load trained model and scaler"""
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def save_model(model, scaler):
    """Save trained model and scaler"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    joblib.dump(model, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    logger.info("Model saved successfully")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_version': MODEL_VERSION
    }), 200

@app.route('/train', methods=['POST'])
def train():
    """
    Train the model on data from a directory
    
    Expected JSON:
    {
        "data_dir": "../cs-train"
    }
    """
    try:
        request_json = request.get_json()
        data_dir = request_json.get('data_dir', '../cs-train')
        
        if not os.path.isdir(data_dir):
            return jsonify({'error': f'Data directory not found: {data_dir}'}), 400
        
        logger.info(f"Starting training with data from {data_dir}")
        start_time = datetime.now()
        
        # Data ingestion and preprocessing
        df = fetch_data(data_dir)
        df_ts = convert_to_ts(df)
        
        # Feature engineering
        X, y, dates = engineer_features(df_ts, target_col='revenue')
        
        # Train-test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Import here to avoid circular imports
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                         learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        save_model(model, scaler)
        
        # Log training
        runtime = str(datetime.now() - start_time)
        metrics = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
        date_range = (str(dates[0]), str(dates[-1]))
        update_train_log(MODEL_TAG, date_range, metrics, runtime, MODEL_VERSION, 
                        "Gradient Boosting model", test=False)
        
        logger.info(f"Training completed in {runtime}")
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'train_samples': int(X_train.shape[0]),
                'test_samples': int(X_test.shape[0]),
                'features': int(X.shape[1])
            },
            'runtime': runtime,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions on new data
    
    Expected JSON:
    {
        "features": [[...], [...]]  # List of feature vectors
    }
    """
    try:
        model, scaler = load_model()
        if model is None or scaler is None:
            return jsonify({'error': 'Model not found. Please train the model first.'}), 400
        
        request_json = request.get_json()
        features = np.array(request_json.get('features', []))
        
        if features.size == 0:
            return jsonify({'error': 'No features provided'}), 400
        
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        
        # Log predictions
        pred_dict = {
            'predictions': predictions.tolist(),
            'n_samples': len(predictions)
        }
        update_predict_log(MODEL_TAG, datetime.now().isoformat(), pred_dict, test=False)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return jsonify({
            'status': 'success',
            'predictions': predictions.tolist(),
            'model_version': MODEL_VERSION,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train_log', methods=['GET'])
def train_log():
    """Retrieve training log"""
    try:
        log_df = get_train_log()
        if log_df is None:
            return jsonify({'error': 'No training log found'}), 404
        
        # Convert to JSON-serializable format
        log_data = log_df.to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'records': log_data,
            'total_records': len(log_data)
        }), 200
    
    except Exception as e:
        logger.error(f"Error retrieving training log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_log', methods=['GET'])
def predict_log():
    """Retrieve prediction log"""
    try:
        log_df = get_predict_log()
        if log_df is None:
            return jsonify({'error': 'No prediction log found'}), 404
        
        # Convert to JSON-serializable format
        log_data = log_df.to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'records': log_data,
            'total_records': len(log_data)
        }), 200
    
    except Exception as e:
        logger.error(f"Error retrieving prediction log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Get latest model metrics"""
    try:
        log_df = get_train_log()
        if log_df is None or len(log_df) == 0:
            return jsonify({'error': 'No training metrics found'}), 404
        
        # Get latest record
        latest = log_df.iloc[-1]
        metrics = json.loads(latest['metrics'])
        
        return jsonify({
            'status': 'success',
            'timestamp': latest['timestamp'],
            'metrics': metrics,
            'model_version': latest['model_version']
        }), 200
    
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get API and model status"""
    model, scaler = load_model()
    model_loaded = model is not None and scaler is not None
    
    log_df = get_train_log()
    training_count = len(log_df) if log_df is not None else 0
    
    pred_log_df = get_predict_log()
    prediction_count = len(pred_log_df) if pred_log_df is not None else 0
    
    return jsonify({
        'status': 'operational',
        'model_loaded': model_loaded,
        'model_version': MODEL_VERSION,
        'model_tag': MODEL_TAG,
        'training_runs': training_count,
        'predictions_made': prediction_count,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Revenue Prediction API")
    app.run(host='0.0.0.0', port=5000, debug=False)
