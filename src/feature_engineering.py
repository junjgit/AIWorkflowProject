#!/usr/bin/env python
"""
Feature engineering module for time-series modeling.
Converts daily time-series data into supervised learning format.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def engineer_features(df, target_col='revenue', lookback_days=[1, 7, 14, 30, 90]):
    """
    Transform time-series data into supervised learning format.
    
    Feature Engineering Strategy:
    - Create lagged features to capture temporal dependencies
    - Use revenue from previous 1, 7, 14, 30, and 90 days as predictors
    - Include additional business metrics as features
    - Target: next day's revenue
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Time-series DataFrame with columns: date, revenue, purchases, 
        unique_invoices, unique_streams, total_views
    target_col : str
        Target column name (default: 'revenue')
    lookback_days : list
        List of lag periods to create features
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix (samples x features)
    y : numpy.ndarray
        Target vector (next day revenue)
    dates : numpy.ndarray
        Date array corresponding to samples
    """
    
    df_feat = df.copy()
    df_feat.sort_values(by='date', inplace=True)
    df_feat.reset_index(drop=True, inplace=True)
    
    # Create lagged features
    for lag in lookback_days:
        df_feat[f'{target_col}_lag{lag}'] = df_feat[target_col].shift(lag)
    
    # Create rolling averages
    for window in [7, 14, 30]:
        df_feat[f'{target_col}_ma{window}'] = df_feat[target_col].rolling(window=window).mean()
    
    # Add other business metrics as features
    df_feat['purchases_lag1'] = df_feat['purchases'].shift(1)
    df_feat['unique_invoices_lag1'] = df_feat['unique_invoices'].shift(1)
    df_feat['unique_streams_lag1'] = df_feat['unique_streams'].shift(1)
    df_feat['total_views_lag1'] = df_feat['total_views'].shift(1)
    
    # Create target (next day revenue)
    df_feat['target'] = df_feat[target_col].shift(-1)
    
    # Remove rows with NaN (due to shifting and rolling windows)
    # Keep rows where we have sufficient history (max lookback + rolling window)
    min_lookback = max(lookback_days) + 30  # max lookback + max rolling window
    df_feat = df_feat.iloc[min_lookback:].copy()
    df_feat.reset_index(drop=True, inplace=True)
    
    if len(df_feat) == 0:
        raise ValueError("Not enough data after feature engineering")
    
    # Select features
    feature_cols = [col for col in df_feat.columns 
                   if col.startswith(target_col) or 
                      col.startswith('purchases_lag') or
                      col.startswith('unique_invoices_lag') or
                      col.startswith('unique_streams_lag') or
                      col.startswith('total_views_lag')]
    
    X = df_feat[feature_cols].values
    y = df_feat['target'].values
    dates = df_feat['date'].values
    
    logger.info(f"Feature engineering complete: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"Features: {feature_cols}")
    
    return X, y, dates


def create_sequences(X, y, seq_length):
    """
    Create sequences for LSTM/GRU models.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    seq_length : int
        Length of sequences
        
    Returns:
    --------
    X_seq : numpy.ndarray
        Sequences of features (samples x seq_length x features)
    y_seq : numpy.ndarray
        Corresponding targets
    """
    
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    
    return np.array(X_seq), np.array(y_seq)
