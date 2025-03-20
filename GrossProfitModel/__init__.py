import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.storage.file.share import ShareServiceClient

# Azure Storage credentials from environment variables
storage_account_url = os.environ['STORAGE_ACCOUNT_URL']
storage_account_key = os.environ['STORAGE_ACCOUNT_KEY']
service_client = ShareServiceClient(account_url=storage_account_url, credential=storage_account_key)

# File share details
share_name = 'financial'
directory_path = 'balancegp'
file_name = 'BalanceGPHistory.csv'

def main(mytimer: func.TimerRequest) -> None:
    """
    Azure Function entry point. This function runs on a timer trigger and:
    1. Downloads historical GP data from Azure File Share
    2. Processes the data and generates predictions
    3. Uploads results to Azure Blob Storage
    """
    try:
        # Get data from Azure File Share
        df = get_file_from_share()
        
        # Process data and generate predictions
        results = generate_future_predictions(df)
        
        # Save results locally
        results.to_csv('gp_monthly_model_performance.csv', index=False)
        
        # Upload results to blob storage
        upload_to_blob('gp_monthly_model_performance.csv', 'gp_monthly_model_performance.csv')
        
        logging.info('GP Model Predictions completed successfully')
        
    except Exception as e:
        logging.error(f'Error in GP Model Predictions: {str(e)}')
        raise

def get_file_from_share():
    """Get CSV file from Azure File Share."""
    try:
        # Get a reference to the file share
        share_client = service_client.get_share_client(share_name)
        directory_client = share_client.get_directory_client(directory_path)
        file_client = directory_client.get_file_client(file_name)

        # Download file
        with open(file_name, 'wb') as file_handle:
            data = file_client.download_file()
            data.readinto(file_handle)

        # Read CSV into DataFrame
        df = pd.read_csv(file_name)
        return df

    except Exception as e:
        logging.error(f'Error getting file from share: {str(e)}')
        raise

def upload_to_blob(local_file_path, blob_name):
    """
    Upload a file to Azure Blob Storage.
    
    Args:
        local_file_path (str): Full path to the local file (e.g., /tmp/myfile.csv)
        blob_name (str): Name for the blob in storage (e.g., myfile.csv)
    """
    # Azure storage account configuration from environment variables
    connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    container_name = os.environ['CONTAINER_NAME']
    
    try:
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Upload file
        with open(local_file_path, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            
        logging.info(f'Successfully uploaded {blob_name} to blob storage')
        
    except Exception as e:
        logging.error(f'Error uploading to blob: {str(e)}')
        raise

def generate_future_predictions(df):
    """
    Generate future GP predictions using historical data.
    
    Args:
        df (pd.DataFrame): Historical GP data
        
    Returns:
        pd.DataFrame: DataFrame with predictions and performance metrics
    """
    try:
        # Process data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Calculate year-over-year growth rates
        df['YoY_Growth'] = df.groupby(df['Date'].dt.month)['GP'].pct_change(periods=12)
        
        # Calculate moving averages
        df['3M_MA'] = df['GP'].rolling(window=3).mean()
        df['12M_MA'] = df['GP'].rolling(window=12).mean()
        
        # Generate predictions for next 12 months
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=12, freq='M')
        
        predictions = []
        for date in future_dates:
            month = date.month
            
            # Get average growth rate for this month
            avg_growth = df[df['Date'].dt.month == month]['YoY_Growth'].mean()
            
            # Get last year's value for this month
            last_year_value = df[df['Date'].dt.month == month]['GP'].iloc[-1]
            
            # Calculate prediction
            prediction = last_year_value * (1 + avg_growth)
            
            predictions.append({
                'Date': date,
                'GP': prediction,
                'Type': 'Prediction'
            })
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Combine historical and predictions
        df['Type'] = 'Historical'
        results = pd.concat([df, predictions_df], ignore_index=True)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(df)
        
        # Add metrics to results
        for metric, value in metrics.items():
            results[metric] = value
        
        return results
        
    except Exception as e:
        logging.error(f'Error generating predictions: {str(e)}')
        raise

def calculate_performance_metrics(df):
    """
    Calculate model performance metrics.
    
    Args:
        df (pd.DataFrame): Historical data
        
    Returns:
        dict: Dictionary of performance metrics
    """
    try:
        # Calculate metrics
        metrics = {
            'MAE': calculate_mae(df),
            'RMSE': calculate_rmse(df),
            'MAPE': calculate_mape(df)
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f'Error calculating metrics: {str(e)}')
        raise

def calculate_mae(df):
    """Calculate Mean Absolute Error"""
    try:
        # Get actual vs predicted values
        actual = df['GP'].iloc[:-12].values  # All but last 12 months
        predicted = df['GP'].iloc[12:].values  # Skip first 12 months
        
        # Calculate MAE
        mae = np.mean(np.abs(actual - predicted))
        return mae
        
    except Exception as e:
        logging.error(f'Error calculating MAE: {str(e)}')
        raise

def calculate_rmse(df):
    """Calculate Root Mean Square Error"""
    try:
        # Get actual vs predicted values
        actual = df['GP'].iloc[:-12].values
        predicted = df['GP'].iloc[12:].values
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        return rmse
        
    except Exception as e:
        logging.error(f'Error calculating RMSE: {str(e)}')
        raise

def calculate_mape(df):
    """Calculate Mean Absolute Percentage Error"""
    try:
        # Get actual vs predicted values
        actual = df['GP'].iloc[:-12].values
        predicted = df['GP'].iloc[12:].values
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return mape
        
    except Exception as e:
        logging.error(f'Error calculating MAPE: {str(e)}')
        raise

def get_excel_file():
    """Get Excel file from Azure File Share."""
    # Azure File Share connection details from environment variables
    storage_account_url = os.environ['STORAGE_ACCOUNT_URL']
    storage_account_key = os.environ['STORAGE_ACCOUNT_KEY']
    service_client = ShareServiceClient(account_url=storage_account_url, credential=storage_account_key)

    # File share details
    share_name = 'financial'
    directory_path = 'balancegp'
    file_name = 'Balance_GP.xlsx'

    try:
        # Get a reference to the file share
        share_client = service_client.get_share_client(share_name)
        directory_client = share_client.get_directory_client(directory_path)
        file_client = directory_client.get_file_client(file_name)

        # Download file
        with open(file_name, 'wb') as file_handle:
            data = file_client.download_file()
            data.readinto(file_handle)

        # Read Excel into DataFrame
        df = pd.read_excel(file_name)
        return df

    except Exception as e:
        logging.error(f'Error getting Excel file: {str(e)}')
        raise
