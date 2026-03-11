#!/usr/bin/env python
"""
Logger module for tracking model training and predictions
"""

import os
import csv
from datetime import datetime
import json

LOG_DIR = "logs"

def ensure_log_dir():
    """Create logs directory if it doesn't exist"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def update_train_log(tag, date_range, metrics, runtime, model_version, model_version_note, test=False):
    """
    Log model training details
    
    Parameters:
    -----------
    tag : str
        Model identifier/tag
    date_range : tuple
        (start_date, end_date) of training data
    metrics : dict
        Dictionary of metrics (e.g., {'rmse': 100.5})
    runtime : str
        Training runtime in HH:MM:SS format
    model_version : float
        Model version number
    model_version_note : str
        Description of model version
    test : bool
        Whether this is a test run
    """
    ensure_log_dir()
    
    log_file = os.path.join(LOG_DIR, "training_log.csv")
    file_exists = os.path.exists(log_file)
    
    timestamp = datetime.now().isoformat()
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'tag', 'start_date', 'end_date', 'runtime', 
                           'model_version', 'model_version_note', 'metrics', 'test'])
        
        metrics_str = json.dumps(metrics)
        writer.writerow([timestamp, tag, date_range[0], date_range[1], runtime, 
                        model_version, model_version_note, metrics_str, test])

def update_predict_log(tag, date, predictions, test=False):
    """
    Log model prediction details
    
    Parameters:
    -----------
    tag : str
        Model identifier/tag
    date : str
        Prediction date
    predictions : dict or list
        Prediction values/details
    test : bool
        Whether this is a test run
    """
    ensure_log_dir()
    
    log_file = os.path.join(LOG_DIR, "prediction_log.csv")
    file_exists = os.path.exists(log_file)
    
    timestamp = datetime.now().isoformat()
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'tag', 'date', 'predictions', 'test'])
        
        pred_str = json.dumps(predictions) if isinstance(predictions, dict) else str(predictions)
        writer.writerow([timestamp, tag, date, pred_str, test])

def get_train_log():
    """Retrieve training log as DataFrame"""
    import pandas as pd
    log_file = os.path.join(LOG_DIR, "training_log.csv")
    if os.path.exists(log_file):
        return pd.read_csv(log_file)
    return None

def get_predict_log():
    """Retrieve prediction log as DataFrame"""
    import pandas as pd
    log_file = os.path.join(LOG_DIR, "prediction_log.csv")
    if os.path.exists(log_file):
        return pd.read_csv(log_file)
    return None
