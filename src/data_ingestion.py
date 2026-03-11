#!/usr/bin/env python
"""
Data ingestion module for the IBM AI Enterprise Workflow Capstone
Handles reading JSON files from multiple sources and preparing feature matrices
"""

import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORRECT_COLUMNS = ['country', 'customer_id', 'day', 'invoice', 'month',
                   'price', 'stream_id', 'times_viewed', 'year']


def fetch_data(data_dir):
    """
    Load all JSON formatted files from a directory into a single DataFrame.
    
    Business Context:
    - Aggregates invoice data from multiple files
    - Handles inconsistent column naming across different months
    - Cleans invoice IDs by removing non-numeric characters
    - Creates uniform datetime columns for time-series analysis
    
    Parameters:
    -----------
    data_dir : str
        Path to directory containing JSON files
        
    Returns:
    --------
    df : pandas.DataFrame
        Consolidated DataFrame with standardized columns and cleaned data
        
    Raises:
    -------
    Exception
        If data_dir doesn't exist or is empty
        If columns don't match expected schema after standardization
    """
    
    # Input validation
    if not os.path.isdir(data_dir):
        logger.error(f"Directory does not exist: {data_dir}")
        raise Exception("specified data dir does not exist")
    
    files = os.listdir(data_dir)
    if not len(files) > 0:
        logger.error(f"Directory is empty: {data_dir}")
        raise Exception("specified data dir does not contain any files")

    # Find all JSON files
    file_list = [os.path.join(data_dir, f) for f in files if re.search(r"\.json$", f)]
    if not file_list:
        logger.error(f"No JSON files found in: {data_dir}")
        raise Exception("No JSON files found in specified directory")
    
    logger.info(f"Found {len(file_list)} JSON files to process")

    # Read data into temporary structure
    all_months = {}
    for file_name in file_list:
        try:
            df = pd.read_json(file_name)
            file_basename = os.path.split(file_name)[-1]
            all_months[file_basename] = df
            logger.info(f"Successfully loaded {file_basename}: {df.shape[0]} records")
        except Exception as e:
            logger.warning(f"Error reading {file_name}: {e}")
            continue

    if not all_months:
        raise Exception("No JSON files could be successfully loaded")

    # Standardize column names
    for f, df in all_months.items():
        cols = set(df.columns.tolist())
        
        # Handle inconsistent column naming
        if 'StreamID' in cols:
            df.rename(columns={'StreamID': 'stream_id'}, inplace=True)
        if 'TimesViewed' in cols:
            df.rename(columns={'TimesViewed': 'times_viewed'}, inplace=True)
        if 'total_price' in cols:
            df.rename(columns={'total_price': 'price'}, inplace=True)

        # Validate columns
        cols = set(df.columns.tolist())
        if not cols == set(CORRECT_COLUMNS):
            logger.error(f"Column mismatch in {f}. Found: {sorted(cols)}, Expected: {sorted(CORRECT_COLUMNS)}")
            raise Exception(f"columns name could not be matched to correct cols in {f}")

    # Concatenate all data
    df = pd.concat(list(all_months.values()), sort=True, ignore_index=True)
    logger.info(f"Concatenated data shape: {df.shape}")

    # Create unified date column
    years = df['year'].values
    months = df['month'].values
    days = df['day'].values
    dates = [f"{years[i]}-{str(months[i]).zfill(2)}-{str(days[i]).zfill(2)}" 
             for i in range(df.shape[0])]
    df['invoice_date'] = np.array(dates, dtype='datetime64[D]')
    
    # Clean invoice IDs (remove letters)
    df['invoice'] = [re.sub(r"\D+", "", str(i)) for i in df['invoice'].values]
    
    # Sort by date and reset index
    df.sort_values(by='invoice_date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Data preparation complete. Date range: {df['invoice_date'].min()} to {df['invoice_date'].max()}")
    
    return df


def convert_to_ts(df_orig, country=None):
    """
    Convert transactional data to time-series format by aggregating daily metrics.
    
    Business Context:
    - Aggregates daily transactions into meaningful business metrics
    - Handles missing days by creating complete date range
    - Computes revenue (target variable) and feature metrics
    - Enables time-series analysis and forecasting
    
    Parameters:
    -----------
    df_orig : pandas.DataFrame
        Original transaction-level DataFrame from fetch_data()
    country : str, optional
        Country code to filter data. If None, uses all data.
        
    Returns:
    --------
    df_time : pandas.DataFrame
        Time-series DataFrame with daily aggregations
    """
    
    if country:
        if country not in np.unique(df_orig['country'].values):
            raise Exception(f"country '{country}' not found in data")
        
        mask = df_orig['country'] == country
        df = df_orig[mask].copy()
        logger.info(f"Filtered to country: {country}, {df.shape[0]} records")
    else:
        df = df_orig.copy()

    # Create complete date range (no missing days)
    df_dates = df['invoice_date'].values.astype('datetime64[D]')
    start_date = df_dates.min()
    end_date = df_dates.max()
    days = np.arange(start_date, end_date, dtype='datetime64[D]')
    
    logger.info(f"Creating time-series from {start_date} to {end_date} ({len(days)} days)")

    # Aggregate metrics by day
    purchases = np.array([np.where(df_dates == day)[0].size for day in days])
    invoices = [np.unique(df[df_dates == day]['invoice'].values).size for day in days]
    streams = [np.unique(df[df_dates == day]['stream_id'].values).size for day in days]
    views = [df[df_dates == day]['times_viewed'].values.sum() for day in days]
    revenue = [df[df_dates == day]['price'].values.sum() for day in days]
    year_month = ["-".join(re.split("-", str(day))[:2]) for day in days]

    df_time = pd.DataFrame({
        'date': days,
        'purchases': purchases,
        'unique_invoices': invoices,
        'unique_streams': streams,
        'total_views': views,
        'year_month': year_month,
        'revenue': revenue
    })
    
    logger.info(f"Time-series shape: {df_time.shape}")
    
    return df_time


def fetch_ts(data_dir, clean=False):
    """
    Load time-series data with caching to CSV for fast retrieval.
    
    Parameters:
    -----------
    data_dir : str
        Path to directory containing raw JSON files
    clean : bool
        If True, recreate time-series files from raw data
        
    Returns:
    --------
    dict
        Dictionary with keys for 'all' data and top-10 countries
    """
    
    ts_data_dir = os.path.join(data_dir, "ts-data")
    
    if clean:
        if os.path.exists(ts_data_dir):
            shutil.rmtree(ts_data_dir)
            logger.info(f"Removed cached time-series data")
    
    if not os.path.exists(ts_data_dir):
        os.mkdir(ts_data_dir)

    # Check for cached files
    if len(os.listdir(ts_data_dir)) > 0:
        logger.info("Loading time-series data from cache")
        return {
            re.sub(r"\.csv$", "", cf)[3:]: pd.read_csv(os.path.join(ts_data_dir, cf))
            for cf in os.listdir(ts_data_dir)
        }

    # Process raw data
    logger.info("Processing raw data into time-series format")
    df = fetch_data(data_dir)

    # Find top 10 countries by revenue
    table = pd.pivot_table(df, index='country', values="price", aggfunc='sum')
    table.columns = ['total_revenue']
    table.sort_values(by='total_revenue', inplace=True, ascending=False)
    top_ten_countries = np.array(list(table.index))[:10]
    
    logger.info(f"Top 10 countries by revenue: {list(top_ten_countries)}")

    # Create time-series for all data and top countries
    dfs = {}
    
    # All data
    ts_all = convert_to_ts(df)
    ts_all.to_csv(os.path.join(ts_data_dir, "ts-all.csv"), index=False)
    dfs['all'] = ts_all
    
    # By country
    for country in top_ten_countries:
        country_id = re.sub(r"\s+", "_", country.lower())
        file_name = os.path.join(ts_data_dir, f"ts-{country_id}.csv")
        ts_country = convert_to_ts(df, country=country)
        ts_country.to_csv(file_name, index=False)
        dfs[country_id] = ts_country
    
    logger.info(f"Created {len(dfs)} time-series datasets")
    
    return dfs


def get_data_summary(df):
    """
    Generate summary statistics of the data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original transaction-level DataFrame
        
    Returns:
    --------
    dict
        Summary statistics
    """
    
    return {
        'total_records': len(df),
        'date_range': (df['invoice_date'].min(), df['invoice_date'].max()),
        'unique_countries': df['country'].nunique(),
        'unique_customers': df['customer_id'].nunique(),
        'unique_streams': df['stream_id'].nunique(),
        'total_revenue': df['price'].sum(),
        'avg_transaction_value': df['price'].mean()
    }
