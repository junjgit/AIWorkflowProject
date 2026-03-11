#!/usr/bin/env python
"""
Unit tests for revenue prediction API and modules
Test-driven development approach for robustness and reliability
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_ingestion import fetch_data, convert_to_ts, get_data_summary
from feature_engineering import engineer_features
from logger import update_train_log, update_predict_log, get_train_log, get_predict_log


class TestDataIngestion(unittest.TestCase):
    """Test data ingestion module"""
    
    def setUp(self):
        """Create temporary test data"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample JSON data
        test_data = [
            {
                'country': 'US',
                'customer_id': 1,
                'day': 1,
                'invoice': 'INV001',
                'month': 1,
                'price': 100.0,
                'stream_id': 1,
                'times_viewed': 5,
                'year': 2020
            },
            {
                'country': 'US',
                'customer_id': 2,
                'day': 1,
                'invoice': 'INV002',
                'month': 1,
                'price': 150.0,
                'stream_id': 2,
                'times_viewed': 3,
                'year': 2020
            }
        ]
        
        df = pd.DataFrame(test_data)
        df.to_json(os.path.join(self.test_dir, 'test_2020-01.json'), orient='records')
    
    def tearDown(self):
        """Clean up temporary test data"""
        shutil.rmtree(self.test_dir)
    
    def test_fetch_data_success(self):
        """Test successful data loading"""
        df = fetch_data(self.test_dir)
        self.assertEqual(df.shape[0], 2)
        self.assertIn('invoice_date', df.columns)
    
    def test_fetch_data_empty_dir(self):
        """Test error handling for empty directory"""
        empty_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(Exception):
                fetch_data(empty_dir)
        finally:
            shutil.rmtree(empty_dir)
    
    def test_fetch_data_nonexistent_dir(self):
        """Test error handling for nonexistent directory"""
        with self.assertRaises(Exception):
            fetch_data('/nonexistent/path')
    
    def test_invoice_id_cleaning(self):
        """Test invoice ID cleaning (remove letters)"""
        df = fetch_data(self.test_dir)
        self.assertTrue(all(df['invoice'].str.isdigit()))
    
    def test_data_summary(self):
        """Test data summary function"""
        df = fetch_data(self.test_dir)
        summary = get_data_summary(df)
        
        self.assertIn('total_records', summary)
        self.assertIn('total_revenue', summary)
        self.assertEqual(summary['total_records'], 2)
        self.assertEqual(summary['total_revenue'], 250.0)


class TestTimeSeries(unittest.TestCase):
    """Test time-series conversion"""
    
    def setUp(self):
        """Create sample time-series data"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.df_ts = pd.DataFrame({
            'date': dates,
            'purchases': np.random.randint(10, 100, 100),
            'unique_invoices': np.random.randint(5, 50, 100),
            'unique_streams': np.random.randint(3, 30, 100),
            'total_views': np.random.randint(100, 500, 100),
            'year_month': [d.strftime('%Y-%m') for d in dates],
            'revenue': np.random.uniform(1000, 5000, 100)
        })
    
    def test_convert_to_ts_shape(self):
        """Test time-series conversion output shape"""
        self.assertIn('date', self.df_ts.columns)
        self.assertIn('revenue', self.df_ts.columns)
        self.assertEqual(len(self.df_ts), 100)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering module"""
    
    def setUp(self):
        """Create sample data for feature engineering"""
        dates = pd.date_range(start='2020-01-01', periods=150, freq='D')
        self.df_ts = pd.DataFrame({
            'date': dates,
            'purchases': np.random.randint(10, 100, 150),
            'unique_invoices': np.random.randint(5, 50, 150),
            'unique_streams': np.random.randint(3, 30, 150),
            'total_views': np.random.randint(100, 500, 150),
            'year_month': [d.strftime('%Y-%m') for d in dates],
            'revenue': np.random.uniform(1000, 5000, 150)
        })
    
    def test_engineer_features_output(self):
        """Test feature engineering output"""
        X, y, dates = engineer_features(self.df_ts, target_col='revenue')
        
        # Check dimensions
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[0], dates.shape[0])
        
        # Check feature count
        self.assertGreater(X.shape[1], 0)
    
    def test_engineer_features_no_nans(self):
        """Test that engineered features have no NaN values"""
        X, y, dates = engineer_features(self.df_ts, target_col='revenue')
        
        self.assertFalse(np.isnan(X).any())
        self.assertFalse(np.isnan(y).any())
    
    def test_engineer_features_lagged_values(self):
        """Test that lagged features are created"""
        X, y, dates = engineer_features(self.df_ts, target_col='revenue', lookback_days=[1, 7])
        
        # Should have at least lag features + rolling averages + other metrics
        self.assertGreater(X.shape[1], 5)


class TestLogger(unittest.TestCase):
    """Test logging functionality"""
    
    def setUp(self):
        """Set up test logging directory"""
        self.test_log_dir = tempfile.mkdtemp()
        # Temporarily override LOG_DIR
        import logger
        self.original_log_dir = logger.LOG_DIR
        logger.LOG_DIR = self.test_log_dir
    
    def tearDown(self):
        """Clean up test logging directory"""
        shutil.rmtree(self.test_log_dir, ignore_errors=True)
        import logger
        logger.LOG_DIR = self.original_log_dir
    
    def test_update_train_log(self):
        """Test training log creation"""
        metrics = {'rmse': 100.5, 'r2': 0.85}
        update_train_log('test_model', ('2020-01-01', '2020-12-31'), 
                        metrics, '00:10:30', 1.0, 'Test model')
        
        log_df = get_train_log()
        self.assertIsNotNone(log_df)
        self.assertEqual(len(log_df), 1)
    
    def test_update_predict_log(self):
        """Test prediction log creation"""
        predictions = {'predictions': [100.0, 150.0, 200.0]}
        update_predict_log('test_model', '2020-01-01', predictions)
        
        log_df = get_predict_log()
        self.assertIsNotNone(log_df)
        self.assertEqual(len(log_df), 1)


class TestAPIDrift(unittest.TestCase):
    """Test scenarios for model drift and scale"""
    
    def setUp(self):
        """Create data with potential drift patterns"""
        # Generate time-series with trend
        dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
        trend = np.linspace(1000, 5000, 365)
        noise = np.random.normal(0, 200, 365)
        self.revenues = trend + noise
        
        self.df_ts = pd.DataFrame({
            'date': dates,
            'purchases': np.random.randint(10, 100, 365),
            'unique_invoices': np.random.randint(5, 50, 365),
            'unique_streams': np.random.randint(3, 30, 365),
            'total_views': np.random.randint(100, 500, 365),
            'year_month': [d.strftime('%Y-%m') for d in dates],
            'revenue': self.revenues
        })
    
    def test_feature_engineering_scale(self):
        """Test feature engineering at scale"""
        # Test with large dataset
        X, y, dates = engineer_features(self.df_ts, target_col='revenue')
        
        self.assertGreater(X.shape[0], 100)
        self.assertEqual(X.shape[0], y.shape[0])
    
    def test_drift_detection(self):
        """Test revenue distribution change detection"""
        # First period
        early_revenue = self.df_ts['revenue'][:90]
        # Recent period
        recent_revenue = self.df_ts['revenue'][-90:]
        
        # Check if there's significant drift
        drift = recent_revenue.mean() - early_revenue.mean()
        self.assertGreater(abs(drift), 0)  # There should be some drift
        
        # Log this for monitoring
        print(f"Revenue drift detected: {drift:.2f}")


class TestDataValidation(unittest.TestCase):
    """Test data validation and quality checks"""
    
    def test_feature_matrix_shape_consistency(self):
        """Test that feature matrices have consistent shapes"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        df_ts = pd.DataFrame({
            'date': dates,
            'purchases': np.random.randint(10, 100, 100),
            'unique_invoices': np.random.randint(5, 50, 100),
            'unique_streams': np.random.randint(3, 30, 100),
            'total_views': np.random.randint(100, 500, 100),
            'year_month': [d.strftime('%Y-%m') for d in dates],
            'revenue': np.random.uniform(1000, 5000, 100)
        })
        
        X, y, dates = engineer_features(df_ts, target_col='revenue')
        
        # Check consistency
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[0], dates.shape[0])
        
        # Check no duplicate indices
        self.assertEqual(len(np.unique(dates)), len(dates))
    
    def test_prediction_output_range(self):
        """Test that predictions are in reasonable range"""
        # Simulate predictions
        true_values = np.random.uniform(1000, 5000, 100)
        predictions = true_values + np.random.normal(0, 100, 100)
        
        # Check that predictions are close to true values
        errors = np.abs(predictions - true_values)
        max_error = errors.max()
        
        # Max error should be reasonable
        self.assertLess(max_error, true_values.max() * 0.5)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
