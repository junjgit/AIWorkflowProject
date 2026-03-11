"""
IBM AI Enterprise Workflow Capstone - Source Module
Complete machine learning pipeline for revenue forecasting
"""

from .data_ingestion import (
    fetch_data,
    convert_to_ts,
    fetch_ts,
    get_data_summary
)

from .feature_engineering import (
    engineer_features,
    create_sequences
)

from .logger import (
    update_train_log,
    update_predict_log,
    get_train_log,
    get_predict_log
)

__version__ = "1.0.0"
__author__ = "HOME AI Team"
__all__ = [
    'fetch_data',
    'convert_to_ts',
    'fetch_ts',
    'get_data_summary',
    'engineer_features',
    'create_sequences',
    'update_train_log',
    'update_predict_log',
    'get_train_log',
    'get_predict_log'
]
