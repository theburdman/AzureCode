import io
import logging
import os
from calendar import monthrange
from datetime import datetime, timezone

import azure.functions as func
import numpy as np
import pandas as pd
import requests
from azure.storage.blob import BlobServiceClient
from azure.storage.fileshare import ShareServiceClient
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def generate_future_predictions(model, scaler, feature_cols, combined_data, last_date):
    """Generate predictions for the next 12 months using the trained model."""
    # Get the last 12 months of data for calculating averages
    last_12_months = combined_data.tail(12)
    avg_constGPNW = last_12_months['constGPNW'].mean()
    avg_rollingGPNW = last_12_months['rollingGPNW'].mean()
    current_backlogPct = combined_data['backlogPct'].iloc[-1]
    current_gpBacklog = combined_data['gpBacklog'].iloc[-1]
    current_CPI = combined_data['CPI'].iloc[-1]
    current_GDP = combined_data['GDP'].iloc[-1]
    current_PrimeRate = combined_data['PrimeRate'].iloc[-1]
    
    # Get the last row's values for columns '1'-'12'
    future_gp_cols = [str(i) for i in range(1, 13)]
    future_gp_values = combined_data[future_gp_cols].iloc[-1].values
    
    # Create future dates starting from the last date
    last_date = pd.to_datetime(last_date)
    future_dates = []
    current_date = last_date
    for _ in range(12):
        if current_date.month == 12:
            next_date = pd.Timestamp(year=current_date.year + 1, month=1, day=1)
        else:
            next_date = pd.Timestamp(year=current_date.year, month=current_date.month + 1, day=1)
        future_dates.append(next_date)
        current_date = next_date
    
    predictions = []
    for i in range(12):
        month_gpBacklog = max(0, current_gpBacklog - sum(predictions)) if predictions else current_gpBacklog
        month_CPI = current_CPI * (1 + 0.02/12) ** i
        month_GDP = current_GDP * (1 + 0.02/12) ** i
        
        features = {
            'gpBacklog': month_gpBacklog,
            'backlogPct': current_backlogPct,
            'constGPNW': avg_constGPNW,
            '1': future_gp_values[i] if i < len(future_gp_values) else future_gp_values[-1],
            'rollingGPNW': avg_rollingGPNW,
            'rolling_12MoConst': avg_constGPNW * 12,
            'CPI': month_CPI,
            'GDP': month_GDP,
            'PrimeRate': current_PrimeRate
        }
        X_future = pd.DataFrame([features])
        X_future = X_future[feature_cols]
        X_future_scaled = scaler.transform(X_future)
        pred = model.predict(X_future_scaled)[0]
        predictions.append(pred)
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Projected_GP': predictions
    })
    return future_df



# Azure Storage credentials from environment variables
storage_account_url = os.environ['STORAGE_ACCOUNT_URL']
storage_account_key = os.environ['STORAGE_ACCOUNT_KEY']
service_client = ShareServiceClient(account_url=storage_account_url, credential=storage_account_key)

# File share details
share_name = 'financial'
directory_path = 'balancegp'
file_name = 'BalanceGPHistory.csv'

def analyze_seasonality(data):
    """Analyze monthly seasonality patterns in the GP data"""
    if 'Month' not in data.columns:
        data['Month'] = data['Date'].dt.month

    monthly_stats = data.groupby('Month')['shiftGP'].agg(['mean', 'std', 'count']).round(2)
    monthly_stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    logging.info("\nMonthly Gross Profit Patterns:")
    logging.info("\nMonthly Statistics (mean, std, count):")
    logging.info(monthly_stats)
    
    data['Year'] = data['Date'].dt.year
    yoy_by_month = data.pivot_table(
        index='Year', 
        columns='Month', 
        values='shiftGP',
        aggfunc='mean'
    )
    
    logging.info("\nYear-over-Year Monthly Patterns:")
    logging.info(yoy_by_month.round(2))
    
    cv = (monthly_stats['std'] / monthly_stats['mean']).round(3)
    logging.info("\nCoefficient of Variation by Month (higher values indicate more variability):")
    logging.info(cv)
    
    return monthly_stats, yoy_by_month, cv

def get_excel_tab_info(try_current=True):
    """Determine the correct Excel tab based on current date."""
    currentDay = datetime.now(timezone.utc).day
    currentYear = datetime.now(timezone.utc).year
    currentMonth = datetime.now(timezone.utc).month

    if try_current and currentDay >= 5:
        tabMonth = currentMonth - 1
        tabYear = currentYear
        if currentMonth == 1:
            tabMonth = 12
            tabYear = currentYear - 1
    else:
        tabMonth = currentMonth - 2
        tabYear = currentYear
        if currentMonth in [1, 2]:
            tabMonth = 12 if currentMonth == 2 else 11
            tabYear = currentYear - 1
            
    nextMonth = tabMonth + 1
    nextYear = tabYear
    if nextMonth > 12:
        nextMonth = 1
        nextYear += 1
    
    numDays = monthrange(nextYear, nextMonth)[1]
    tabMonth = f"{int(tabMonth):02d}"
    tabYear = str(tabYear)[2:4]
    tabName = tabMonth + '.' + str(tabYear)
    
    return tabName, nextYear, nextMonth, numDays

def get_balance_gp_projections():
    """Get balance GP projections from Excel file."""
    tab_info = get_excel_tab_info(try_current=True)
    tabName = tab_info[0]
    
    if datetime.now(timezone.utc).day >= 5:
        logging.info(f"\nTrying to use current tab: {tabName}")
        try:
            file_stream = get_excel_file()
            pd.read_excel(file_stream, sheet_name=tabName, header=0, engine='openpyxl')
        except Exception as e:
            logging.info(f"Could not read current tab {tabName}, falling back to previous month")
            tab_info = get_excel_tab_info(try_current=False)
            tabName = tab_info[0]
            logging.info(f"Using previous month's tab: {tabName}")
    else:
        tab_info = get_excel_tab_info(try_current=False)
        tabName = tab_info[0]
        logging.info(f"Using previous month's tab: {tabName}")
    
    nextYear = tab_info[1]
    nextMonth = tab_info[2]
    numDays = tab_info[3]
    dateStart = pd.to_datetime(nextYear * 10000 + nextMonth * 100 + numDays, format='%Y%m%d')
    
    file_stream = get_excel_file()
    file_stream.seek(0)

    yearDrop = pd.read_excel(file_stream, sheet_name=tabName, header=0, engine='openpyxl')
    yearCols = [c for c in yearDrop.columns if str(c).upper()[:4] == 'YEAR']
    dropColumns = [yearDrop.columns.get_loc(i) for i in yearCols]

    file_stream.seek(0)
    balanceGP = pd.read_excel(file_stream, sheet_name=tabName, header=1, engine='openpyxl')
    balanceGP = balanceGP.drop([0])
    balanceGP.drop(balanceGP.columns[dropColumns], axis=1, inplace=True)
    
    colStart = balanceGP.columns.get_loc(dateStart)
    balanceGP['Project'] = balanceGP['Project'].astype(str)
    endActivesRow = balanceGP.loc[balanceGP['Project'].str.upper() == 'ACTIVE PROJECTS G.P.'].index.values.astype(int)[0]
    endMktRow = balanceGP.loc[balanceGP['Project'].str.upper() == 'MARKETING PROJECTS G.P.'].index.values.astype(int)[0]

    logging.info(f"Found Active Projects row at: {endActivesRow}")
    logging.info(f"Found Marketing Projects row at: {endMktRow}")

    return balanceGP, colStart, endActivesRow, endMktRow

def get_excel_file():
    """Get Excel file from Azure File Share."""
    share_client = service_client.get_share_client(share_name)
    directory_client = share_client.get_directory_client(directory_path)
    file_client = directory_client.get_file_client('BalanceGPProjections.xlsx')
    
    file_stream = io.BytesIO()
    file_client.download_file().readinto(file_stream)
    file_stream.seek(0)
    return file_stream

def get_historical_balance_gp():
    """Get historical balance GP data from CSV."""
    share_client = service_client.get_share_client(share_name)
    directory_client = share_client.get_directory_client(directory_path)
    file_client = directory_client.get_file_client(file_name)

    file_stream = io.BytesIO()
    file_client.download_file().readinto(file_stream)
    file_stream.seek(0)

    balanceGP = pd.read_csv(file_stream)
    balanceGP['Date'] = pd.to_datetime(balanceGP.Year*10000+balanceGP.Month*100+1, format='%Y%m%d')
    
    unnamed_cols = [col for col in balanceGP.columns if 'Unnamed' in col]
    balanceGP = balanceGP.drop(columns=unnamed_cols)

    month12 = [str(i) for i in range(1, 13)]
    month13 = [str(i) for i in range(1, 14)]
    month14 = [str(i) for i in range(1, 15)]
    month15 = [str(i) for i in range(1, 16)]

    balanceGP['monthSum12'] = balanceGP[month12].sum(axis=1)
    balanceGP['monthSum13'] = balanceGP[month13].sum(axis=1)
    balanceGP['monthSum14'] = balanceGP[month14].sum(axis=1)
    balanceGP['monthSum15'] = balanceGP[month15].sum(axis=1)

    return balanceGP

def get_financial_data():
    """Retrieve monthly financial general ledger data from API."""
    # Make API call
    response = requests.get(
        'https://vue.korteco.com/korteapi/intranet/GLMonthlyStatus',
        auth=('kortetech', 'F7DV%2%Yn+jn/aq=w2W/')
    )
    financials = response.json()

    # Convert to DataFrame and process
    financials_df = pd.DataFrame.from_dict(financials)
    financials_df['date'] = pd.to_datetime(financials_df['mth'])
    financials_df['year'] = pd.to_numeric(financials_df['mth'].str[:4])
    financials_df['month'] = pd.to_numeric(financials_df['mth'].str[5:7])

    # Create shiftGP as the target variable (next month's constGP)
    financials_df['shiftGP'] = financials_df['constGP'].shift(-1)
    
    # Create 12-month and 15-month forward sums of shiftGP
    # For each row, sum the next 12 months of shiftGP (including current month)
    financials_df['shiftGP12'] = financials_df['shiftGP'].rolling(window=12, min_periods=12).sum().shift(-11)
    # For each row, sum the next 15 months of shiftGP (including current month)
    financials_df['shiftGP15'] = financials_df['shiftGP'].rolling(window=15, min_periods=15).sum().shift(-14)

    # Print rows starting from October 2000
    logging.info("\nRows from October 2000:")
    oct_2000_start = financials_df[financials_df['date'] >= '2000-10-01'].head(10)
    logging.info(oct_2000_start[['date', 'constGP', 'shiftGP', 'shiftGP12', 'shiftGP15']].to_string())

    # Adjusted financial columns as features
    financials_df['rollingGPNW'] = financials_df['constGPNW'].shift().rolling(12).sum()
    financials_df['rolling_12MoConst'] = financials_df['_12MoConst'].shift().rolling(12).sum()

    # Add is_December binary indicator
    financials_df['is_December'] = (financials_df['month'] == 12).astype(int)

    # Filter out rows where constRevenue is 0
    financials_df = financials_df[financials_df.constRevenue != 0]

    # Select required columns
    financials_df = financials_df[['date', 'constGP', 'totalGP', 'shiftGP', 'rollingGPNW', 'rolling_12MoConst', 'gpBacklog', 'constGPNW', 'constGPNWMovingAvg',
                                   'backlog', 'totalGPPct', 'backlogPct', 'gpBudget', 'shiftGP12', 'shiftGP15', 'is_December']]

    return financials_df

def get_fred_data(series_id, column_name):
    """Retrieve data from FRED API for a given series ID."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    df = pd.read_csv(url)
    df.columns = ['Date', column_name]
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_economic_indicators():
    """Retrieve CPI, GDP, and prime rate data from FRED API and interpolate missing values."""
    # Get individual indicators
    cpi_data = get_fred_data('CPIAUCSL', 'CPI')
    gdp_data = get_fred_data('GDP', 'GDP')
    prime_rate_data = get_fred_data('DPRIME', 'PrimeRate')
    
    # Combine all indicators
    economic_data = cpi_data.merge(gdp_data, on='Date', how='outer')
    economic_data = economic_data.merge(prime_rate_data, on='Date', how='outer')
    
    # Forward fill missing values
    economic_data = economic_data.fillna(method='ffill')
    
    # Add year and month columns
    economic_data['Year'] = economic_data['Date'].dt.year
    economic_data['Month'] = economic_data['Date'].dt.month
    
    return economic_data

def combine_all_data(balance_gp, financials_df, economic_indicators):
    """Combine balance GP data, financial data, and economic indicators into a single DataFrame."""
    # Ensure date columns are datetime
    balance_gp['Date'] = pd.to_datetime(balance_gp['Date'])
    financials_df['date'] = pd.to_datetime(financials_df['date'])
    economic_indicators['Date'] = pd.to_datetime(economic_indicators['Date'])
    
    # Check for duplicates before merge
    logging.info("\nChecking for duplicates before merge:")
    logging.info(f"Balance GP duplicated dates: {balance_gp['Date'].duplicated().sum()}")
    logging.info(f"Financials duplicated dates: {financials_df['date'].duplicated().sum()}")
    logging.info(f"Economic duplicated dates: {economic_indicators['Date'].duplicated().sum()}")
    
    # Merge financial data with balance GP
    result = pd.merge(balance_gp, financials_df, left_on='Date', right_on='date', how='left')
    logging.info(f"\nRows after first merge: {len(result)}")
    
    # Merge economic indicators
    result = pd.merge(result, economic_indicators, on='Date', how='left')
    logging.info(f"\nRows after second merge: {len(result)}")
    
    # Forward fill any missing values
    result = result.fillna(method='ffill')
    
    return result

def analyze_column_one(data):
    """Analyze the '1' column (Accounting's 1-month GP prediction)"""
    logging.info("\nVerifying '1' column values after data combination:")
    logging.info("\nSummary statistics for column '1':")
    logging.info(data['1'].describe())
    
    logging.info("\nFirst 10 values of column '1':")
    logging.info(data[['Date', '1']].head(10).to_string())

def upload_to_blob(local_file_path: str, blob_name: str):
    """Upload a file to Azure Blob Storage
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
        
        # Get a reference to the container
        container_client = blob_service_client.get_container_client(container_name)
        
        # Get a reference to the blob
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload the file
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            
        logging.info(f"Successfully uploaded {blob_name} to Azure blob storage")
    except Exception as e:
        logging.error(f"Error uploading {blob_name} to Azure blob: {str(e)}")
        raise


def analyze_feature_importance(model, feature_names, X, y):
    """Analyze and plot feature importance."""
    if isinstance(model, Ridge):
        importance = np.abs(model.coef_)
    else:  # ElasticNet
        importance = np.abs(model.coef_)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    logging.info("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")
    
    return feature_importance

def train_aggregate_models(combined_data):
    """
    Train aggregate models to predict 12-month and 15-month aggregated gross profit using walk-forward validation.
    """
    # Fill missing target values
    combined_data['next_12mo_gp'] = combined_data['next_12mo_gp'].ffill().bfill()
    combined_data['next_15mo_gp'] = combined_data['next_15mo_gp'].ffill().bfill()

    # Create interaction terms if columns exist
    if 'backlogPct' in combined_data.columns and 'rolling_12MoConst' in combined_data.columns:
        combined_data['backlogPct_x_rolling12Mo'] = combined_data['backlogPct'] * combined_data['rolling_12MoConst']
    if 'backlogPct' in combined_data.columns and 'rollingGPNW' in combined_data.columns:
        combined_data['backlogPct_x_rollingGPNW'] = combined_data['backlogPct'] * combined_data['rollingGPNW']
        
    # Define features
    common_features = [
        'CPI', 'PrimeRate', 'GDP',
        'gpBacklog', 'backlogPct', 'constGPNW', 'constGPNWMovingAvg',
        'rolling_12MoConst', 'rollingGPNW'
    ]
    features_12mo = ['monthSum12'] + common_features
    features_15mo = ['monthSum15'] + common_features

    # Handle NaN values in features
    all_features = list(set(features_12mo + features_15mo))
    for col in all_features:
        if col in combined_data.columns:
            combined_data[col] = combined_data[col].ffill().bfill()
        else:
            logging.info(f"Warning: {col} not found in data")

    # Initialize lists to store predictions
    dates = []
    pred_12mo = []
    pred_15mo = []
    actual_12mo = []
    actual_15mo = []

    scaler_12mo = StandardScaler()
    scaler_15mo = StandardScaler()
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}

    train_start = pd.Timestamp('2000-10-01')
    for pred_date in combined_data[combined_data['Date'] >= '2001-10-01']['Date']:
        train_data = combined_data[(combined_data['Date'] >= train_start) & (combined_data['Date'] < pred_date)]
        if len(train_data) < 12:
            continue
        X_train_12mo = train_data[features_12mo].copy().ffill().bfill()
        y_train_12mo = train_data['next_12mo_gp']
        X_train_15mo = train_data[features_15mo].copy().ffill().bfill()
        y_train_15mo = train_data['next_15mo_gp']

        X_train_12mo_scaled = scaler_12mo.fit_transform(X_train_12mo)
        X_train_15mo_scaled = scaler_15mo.fit_transform(X_train_15mo)

        model_12mo = GridSearchCV(Ridge(), ridge_params, cv=5)
        model_12mo.fit(X_train_12mo_scaled, y_train_12mo)

        model_15mo = GridSearchCV(Ridge(), ridge_params, cv=5)
        model_15mo.fit(X_train_15mo_scaled, y_train_15mo)

        pred_row = combined_data[combined_data['Date'] == pred_date].copy()
        X_pred_12mo = pred_row[features_12mo].copy().ffill().bfill()
        X_pred_15mo = pred_row[features_15mo].copy().ffill().bfill()
        X_pred_12mo_scaled = scaler_12mo.transform(X_pred_12mo)
        X_pred_15mo_scaled = scaler_15mo.transform(X_pred_15mo)

        prediction_12 = model_12mo.predict(X_pred_12mo_scaled)[0]
        prediction_15 = model_15mo.predict(X_pred_15mo_scaled)[0]

        dates.append(pred_date)
        pred_12mo.append(prediction_12)
        pred_15mo.append(prediction_15)
        actual_12mo.append(pred_row['next_12mo_gp'].iloc[0])
        actual_15mo.append(pred_row['next_15mo_gp'].iloc[0])

    results = pd.DataFrame({
        'Date': dates,
        'Predicted_12mo_GP': pred_12mo,
        'Actual_12mo_GP': actual_12mo,
        'Predicted_15mo_GP': pred_15mo,
        'Actual_15mo_GP': actual_15mo
    })

    mae_12 = np.mean(np.abs(results['Predicted_12mo_GP'] - results['Actual_12mo_GP']))
    mae_15 = np.mean(np.abs(results['Predicted_15mo_GP'] - results['Actual_15mo_GP']))
    r2_12 = 1 - (np.sum((results['Actual_12mo_GP'] - results['Predicted_12mo_GP'])**2) / np.sum((results['Actual_12mo_GP'] - np.mean(results['Actual_12mo_GP']))**2))
    r2_15 = 1 - (np.sum((results['Actual_15mo_GP'] - results['Predicted_15mo_GP'])**2) / np.sum((results['Actual_15mo_GP'] - np.mean(results['Actual_15mo_GP']))**2))

    logging.info("\nAggregate Model Performance:")
    logging.info(f"12-month MAE: {mae_12:.2f}, R2: {r2_12:.3f}")
    logging.info(f"15-month MAE: {mae_15:.2f}, R2: {r2_15:.3f}")

    # Create performance results with filtered dates
    performance_results = results.copy()
    performance_results['Date'] = pd.to_datetime(performance_results['Date'])
    max_date = pd.to_datetime(combined_data['Date']).max()
    cutoff_date_12mo = max_date - pd.DateOffset(months=12)
    cutoff_date_15mo = max_date - pd.DateOffset(months=15)

    # Filter 12-month predictions
    mask_12mo = performance_results['Date'] <= cutoff_date_12mo
    performance_results.loc[~mask_12mo, ['Predicted_12mo_GP', 'Actual_12mo_GP']] = np.nan

    # Filter 15-month predictions
    mask_15mo = performance_results['Date'] <= cutoff_date_15mo
    performance_results.loc[~mask_15mo, ['Predicted_15mo_GP', 'Actual_15mo_GP']] = np.nan

    # Remove rows where all predictions are NaN
    performance_results = performance_results.dropna(subset=['Predicted_12mo_GP', 'Predicted_15mo_GP'], how='all')
    
    # Remove rows where actuals are 0 or null
    performance_results = performance_results[
        (performance_results['Actual_12mo_GP'].notna() & (performance_results['Actual_12mo_GP'] != 0)) &
        (performance_results['Actual_15mo_GP'].notna() & (performance_results['Actual_15mo_GP'] != 0))
    ]
    
    logging.info(f"\nFiltered out rows with null or zero actuals from aggregate results")
    
    return results, performance_results

def create_long_range_projections(combined_data, results):
    """Create long range projections combining historical actuals and future projections."""
    # Get cutoff dates
    max_date = pd.to_datetime(combined_data['Date']).max()
    cutoff_date_12mo = max_date - pd.DateOffset(months=12)
    cutoff_date_15mo = max_date - pd.DateOffset(months=15)
    
    # Create projections DataFrame
    projections = pd.DataFrame()
    projections['date'] = results['Date']
    projections['date'] = pd.to_datetime(projections['date'])
    
    # Get last 6 months of actuals and model predictions for 12-month aggregates
    mask_12mo = (combined_data['Date'] <= cutoff_date_12mo) & (combined_data['Date'] > cutoff_date_12mo - pd.DateOffset(months=6))
    actuals_12mo = combined_data[mask_12mo][['Date', 'shiftGP12']].copy()
    actuals_12mo.columns = ['date', 'actual_12mo']
    
    model_12mo = results[results['Date'].isin(actuals_12mo['date'])][['Date', 'Predicted_12mo_GP']].copy()
    model_12mo.columns = ['date', 'model_12mo']
    
    # Get last 6 months of actuals and model predictions for 15-month aggregates
    mask_15mo = (combined_data['Date'] <= cutoff_date_15mo) & (combined_data['Date'] > cutoff_date_15mo - pd.DateOffset(months=6))
    actuals_15mo = combined_data[mask_15mo][['Date', 'shiftGP15']].copy()
    actuals_15mo.columns = ['date', 'actual_15mo']
    
    model_15mo = results[results['Date'].isin(actuals_15mo['date'])][['Date', 'Predicted_15mo_GP']].copy()
    model_15mo.columns = ['date', 'model_15mo']
    
    # Get future projections (dates after cutoff)
    future_12mo = results[results['Date'] > cutoff_date_12mo][['Date', 'Predicted_12mo_GP']].copy()
    future_12mo.columns = ['date', 'projected_12mo']
    
    future_15mo = results[results['Date'] > cutoff_date_15mo][['Date', 'Predicted_15mo_GP']].copy()
    future_15mo.columns = ['date', 'projected_15mo']
    
    # Combine all into final projections DataFrame
    projections = pd.merge(actuals_12mo, actuals_15mo, on='date', how='outer')
    projections = pd.merge(projections, model_12mo, on='date', how='outer')
    projections = pd.merge(projections, model_15mo, on='date', how='outer')
    projections = pd.merge(projections, future_12mo, on='date', how='outer')
    projections = pd.merge(projections, future_15mo, on='date', how='outer')
    
    # Sort by date
    projections = projections.sort_values('date')
    
    return projections

def run_aggregate_models(combined_data):
    """Wrapper function to run aggregate models and upload results."""
    results, performance_results = train_aggregate_models(combined_data)
    
    # Create and save long range projections
    projections = create_long_range_projections(combined_data, results)
    
    # Save files
    projections.to_csv('/tmp/gp_long_range_projections.csv', index=False)
    performance_results.to_csv('/tmp/gp_long_range_performance.csv', index=False)
    
    # Upload to Azure blob storage
    upload_to_blob('/tmp/gp_long_range_projections.csv', 'gp_long_range_projections.csv')
    upload_to_blob('/tmp/gp_long_range_performance.csv', 'gp_long_range_performance.csv')
    
    logging.info(f"Successfully uploaded long range projections and performance files")

def main(mytimer: func.TimerRequest) -> None:
    """Azure Function that runs our Gross Profit projections model on a timer."""
    start_time = datetime.utcnow()
    utc_timestamp = start_time.replace(tzinfo=timezone.utc).isoformat()
    run_id = start_time.strftime('%Y%m%d_%H%M%S')

    if mytimer.past_due:
        logging.warning(f'[Run {run_id}] The timer is past due!')

    logging.info(f'[Run {run_id}] Starting Gross Profit Projections Model at {utc_timestamp}')
    
    try:
        # Get historical balance GP data
        balance_gp = get_historical_balance_gp()
        
        # Get financial data
        financials_df = get_financial_data()
        
        # Get economic indicators
        economic_indicators = get_economic_indicators()
        
        # Combine all data
        combined_data = combine_all_data(balance_gp, financials_df, economic_indicators)
        
        # Run aggregate models (this will handle saving and uploading long range files)
        results = run_aggregate_models(combined_data)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        logging.info(f'[Run {run_id}] Successfully completed Gross Profit Projections run in {duration:.2f} seconds')
    except Exception as e:
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        logging.error(f'[Run {run_id}] Error running Gross Profit Projections after {duration:.2f} seconds: {str(e)}')
        raise  # Re-raise to ensure Azure Functions marks this run as failed

    logging.info(f'[Run {run_id}] Python timer trigger function completed at {end_time.replace(tzinfo=timezone.utc).isoformat()}')
    
    file_stream = io.BytesIO()
    file_client.download_file().readinto(file_stream)
    file_stream.seek(0)

    # Read and process the data
    balanceGP = pd.read_csv(file_stream)
    balanceGP['Date'] = pd.to_datetime(balanceGP.Year*10000+balanceGP.Month*100+1, format='%Y%m%d')
    
    # Drop the unnamed column
    unnamed_cols = [col for col in balanceGP.columns if 'Unnamed' in col]
    balanceGP = balanceGP.drop(columns=unnamed_cols)
    
    # Convert currency strings to numeric values for columns 1-15
    numeric_cols = [str(i) for i in range(1, 16)]
    for col in numeric_cols:
        if col in balanceGP.columns:
            balanceGP[col] = balanceGP[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    
    # Calculate sums for different time horizons
    month12 = [str(i) for i in range(1, 13)]
    month13 = [str(i) for i in range(1, 14)]
    month14 = [str(i) for i in range(1, 15)]
    month15 = [str(i) for i in range(1, 16)]
    
    balanceGP['monthSum12'] = balanceGP[month12].sum(axis=1)
    balanceGP['monthSum13'] = balanceGP[month13].sum(axis=1)
    balanceGP['monthSum14'] = balanceGP[month14].sum(axis=1)
    balanceGP['monthSum15'] = balanceGP[month15].sum(axis=1)
    
    return balanceGP

# Define month sequences for different period lengths
MONTH_SEQUENCES = {
    12: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    13: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
    14: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
    15: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
}

def get_previous_month_year(year, month):
    """Get the previous month and year, handling year rollover."""
    if month == 1:
        return year - 1, 12
    return year, month - 1

def format_tab_name(month, year):
    """Format the tab name as MM.YY"""
    return f"{month:02d}.{str(year)[2:4]}"

def get_excel_tab_info(try_current=True):
    """Determine the correct Excel tab based on current date.
    Args:
        try_current (bool): If True, try to get current month's tab first (one month back).
                           If False, use previous month's tab (two months back).
    """
    now = datetime.now(timezone.utc)
    currentDay = now.day
    currentYear = now.year
    currentMonth = now.month
    
    logging.info(f"Current UTC time: {now.isoformat()}, Day: {currentDay}, Month: {currentMonth}, Year: {currentYear}")

    # Determine the starting point for tab search
    if try_current and currentDay >= 5:
        # After the 5th, start looking from one month back
        searchYear, searchMonth = get_previous_month_year(currentYear, currentMonth)
    else:
        # Before the 5th or fallback: start looking from two months back
        searchYear, searchMonth = get_previous_month_year(currentYear, currentMonth)
        searchYear, searchMonth = get_previous_month_year(searchYear, searchMonth)
    
    # Calculate the month after the tab month (for projections)
    nextMonth = searchMonth + 1
    nextYear = searchYear
    if nextMonth > 12:
        nextMonth = 1
        nextYear += 1
    
    # Get the last day of the next month
    numDays = monthrange(nextYear, nextMonth)[1]

    tabName = format_tab_name(searchMonth, searchYear)
    logging.info(f"Calculated tab info - Tab: {tabName}, Next Month: {nextMonth}, Next Year: {nextYear}, Days: {numDays}")
    
    return tabName, nextYear, nextMonth, numDays
    
    return tabName, nextYear, nextMonth, numDays

def find_latest_available_tab(max_attempts=6):
    """Try to find the latest available tab by looking back through previous months.
    Args:
        max_attempts (int): Maximum number of months to look back
    Returns:
        tuple: (tab_info, file_stream) or raises exception if no valid tab found
    """
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    file_stream = None

    # First try the expected tab based on day of month
    try_current = now.day >= 5
    tab_info = get_excel_tab_info(try_current=try_current)
    
    for attempt in range(max_attempts):
        try:
            if file_stream:
                file_stream.close()
            file_stream = get_excel_file()
            
            # Test if we can read this tab
            pd.read_excel(file_stream, sheet_name=tab_info[0], header=0, engine='openpyxl')
            logging.info(f"Successfully found valid tab: {tab_info[0]}")
            return tab_info, file_stream
            
        except Exception as e:
            logging.warning(f"Could not read tab {tab_info[0]}, trying previous month. Error: {str(e)}")
            # Get previous month's info
            year, month = get_previous_month_year(year, month)
            tab_name = format_tab_name(month, year)
            
            # Calculate next month for projections
            next_month = month + 1
            next_year = year
            if next_month > 12:
                next_month = 1
                next_year += 1
            
            num_days = monthrange(next_year, next_month)[1]
            tab_info = (tab_name, next_year, next_month, num_days)
    
    if file_stream:
        file_stream.close()
    raise ValueError(f"No valid tab found after trying {max_attempts} months back")

def get_balance_gp_projections():
    """Get balance GP projections from Excel file."""
    # Try to find the latest available tab
    tab_info, file_stream = find_latest_available_tab()
    tabName = tab_info[0]  # Extract tab name from the tuple
    
    nextYear = tab_info[1]
    nextMonth = tab_info[2]
    numDays = tab_info[3]
    
    # Calculate the start date using the last day of next month
    dateStart = pd.to_datetime(nextYear * 10000 + nextMonth * 100 + numDays, format='%Y%m%d')
    
    # Read Excel file and process data
    yearDrop = pd.read_excel(file_stream, sheet_name=tabName, header=0, engine='openpyxl')
    yearCols = [c for c in yearDrop.columns if str(c).upper()[:4] == 'YEAR']
    dropColumns = [yearDrop.columns.get_loc(i) for i in yearCols]

    # Reset file stream position for second read
    file_stream.seek(0)
    
    # Read main data
    balanceGP = pd.read_excel(file_stream, sheet_name=tabName, header=1, engine='openpyxl')
    balanceGP = balanceGP.drop([0])
    balanceGP.drop(balanceGP.columns[dropColumns], axis=1, inplace=True)
    
    # Find the column index for our start date
    colStart = balanceGP.columns.get_loc(dateStart)

    # Find the rows for Active and Marketing projects
    balanceGP['Project'] = balanceGP['Project'].astype(str)
    endActivesRow = balanceGP.loc[balanceGP['Project'].str.upper() == 'ACTIVE PROJECTS G.P.'].index.values.astype(int)[0]
    endMktRow = balanceGP.loc[balanceGP['Project'].str.upper() == 'MARKETING PROJECTS G.P.'].index.values.astype(int)[0]

    print(f"Found Active Projects row at: {endActivesRow}")
    print(f"Found Marketing Projects row at: {endMktRow}")

    # Get projected gross profit for next 15 months
    print("\nProjected Gross Profit for next 15 months:")
    currentBalanceGP = []
    for i in range(0, 15):
        activeJobs = balanceGP.iloc[:, colStart + i][endActivesRow]
        mktJobs = balanceGP.iloc[:, colStart + i][endMktRow]
        totalGP = activeJobs + mktJobs
        currentBalanceGP.append(totalGP)
    
    # Create label using tab info
    month = int(tabName.split('.')[0])
    year = int('20' + tabName.split('.')[1])
    label = f"{month}, {year}"
    
    # Create and format DataFrame
    currentBalanceGP = pd.DataFrame(currentBalanceGP)
    currentBalanceGP.index += 1
    currentBalanceGP = currentBalanceGP.transpose()
    currentBalanceGP.insert(0, 'Label', label)
    currentBalanceGP.columns = ['Label', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    currentBalanceGP['Label'] = currentBalanceGP['Label'].astype(str)
    currentBalanceGP['Month'] = pd.to_numeric(currentBalanceGP['Label'].str.split(',').str[0])
    currentBalanceGP['Year'] = pd.to_numeric(currentBalanceGP['Label'].str.split(',').str[1])
    currentBalanceGP['Date'] = pd.to_datetime(currentBalanceGP.Year * 10000 + currentBalanceGP.Month * 100 + 1, format='%Y%m%d')

    # Get the existing balanceGPHistory DataFrame
    balanceGPHistory = get_historical_balance_gp()
    
    # Check if we have this label in the existing data
    matching_row = balanceGPHistory[balanceGPHistory['Label'] == label]
    
    should_update = False
    if len(matching_row) > 0:
        # Compare the values (excluding Label, Month, Year, and Date columns)
        numeric_cols = [str(i) for i in range(1, 16)]  # columns '1' through '15'
        print("\nCurrent row values:")
        print(currentBalanceGP[numeric_cols].values)
        print("\nExisting row values:")
        print(matching_row[numeric_cols].values)
        
        if not (currentBalanceGP[numeric_cols].values == matching_row[numeric_cols].values).all():
            should_update = True
            print("\nDifferences found, will update data")
            # Update only the numeric columns in the matching row
            balanceGPHistory.loc[balanceGPHistory['Label'] == label, numeric_cols] = currentBalanceGP.iloc[0][numeric_cols].values
        else:
            print(f"\nNo differences found for label {label}, no update needed")
    else:
        should_update = True
        print("\nNew data found, will update")
        # Append the new row to balanceGPHistory
        balanceGPHistory = pd.concat([balanceGPHistory, currentBalanceGP], ignore_index=True)
    
    # Sort the complete dataset by date
    balanceGPHistory = balanceGPHistory.sort_values('Date').reset_index(drop=True)
    
    return balanceGPHistory, balanceGPHistory, should_update, tabName, matching_row

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

    # Get file from Azure File Share
    share_client = service_client.get_share_client(share_name)
    directory_client = share_client.get_directory_client(directory_path)
    file_client = directory_client.get_file_client(file_name)
    file_stream = io.BytesIO()
    file_client.download_file().readinto(file_stream)
    file_stream.seek(0)
    
    return file_stream

def get_historical_financials():
    """Get historical financial data from the Korte API."""
    # Get financial data from API
    financials = requests.get('https://vue.korteco.com/korteapi/intranet/GLMonthlyStatus', 
                           auth=('kortetech', 'F7DV%2%Yn+jn/aq=w2W/'))
    financials = financials.json()

    # Convert to DataFrame and process
    financials_df = pd.DataFrame.from_dict(financials)
    financials_df['date'] = pd.to_datetime(financials_df['mth'])
    financials_df['year'] = pd.to_numeric(financials_df['mth'].str[:4])
    financials_df['month'] = pd.to_numeric(financials_df['mth'].str[5:7])
    financials_df['rollingGPNW'] = financials_df['constGPNW'].shift().rolling(12).sum()
    financials_df['rolling_12MoConst'] = financials_df['_12MoConst'].shift().rolling(12).sum()

    # Filter out rows where constRevenue is 0
    financials_df = financials_df[financials_df.constRevenue != 0]

    # Select required columns
    financials_df = financials_df[['constGP', 'totalGP', 'rollingGPNW', 'rolling_12MoConst', 'gpBacklog', 'constGPNW', 'constGPNWMovingAvg',
                                'totalGPPct', 'gpBudget', 'date']]
    
    return financials_df

def get_cpi_data():
    """Scrape and process CPI data for inflation proxy."""
    try:
        url = 'https://www.usinflationcalculator.com/inflation/consumer-price-index-and-annual-percent-changes-from-1913-to-2008/'
        cpi = pd.read_html(url)[0]
    except Exception as e:
        print(f"\nWarning: Could not fetch CPI data: {str(e)}")
        print("Using placeholder CPI data for testing...")
        # Create placeholder data
        # Get data for the last 15 years
        current_year = pd.Timestamp.now().year
        years = range(current_year - 14, current_year + 1)
        months = range(1, 13)
        data = [(year, month, 100.0) for year in years for month in months]
        cpi = pd.DataFrame(data, columns=['Year', 'Month', 'CPI'])
        
        # Add required columns
        cpi['Year'] = cpi['Year'].astype(int)
        cpi['Month'] = cpi['Month'].astype(int)
        cpi['Match'] = cpi['Month'].astype(str) + cpi['Year'].astype(str)
        cpi['Date'] = pd.to_datetime(cpi.Year*10000+cpi.Month*100+1,format='%Y%m%d')
        
        return cpi
    
    # Clean and process the data
    cpi = cpi[1:]
    cpi.columns = cpi.iloc[0]
    cpi = cpi.drop(columns=['Avg', 'Dec-Dec', 'Avg-Avg'])
    
    # Melt the DataFrame to get month-wise data
    cpi = pd.melt(cpi, id_vars=['Year'], 
                  value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    cpi.rename(columns={1: 'Month', 'value': 'CPI'}, inplace=True)
    
    # Map month names to numbers
    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
              'July': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    cpi.Month = cpi.Month.map(months)
    
    # Clean numeric data
    cpi = cpi[pd.to_numeric(cpi['CPI'], errors='coerce').notnull()]
    cpi['CPI'] = pd.to_numeric(cpi['CPI'])
    cpi = cpi.sort_values(by=['Year', 'Month'])
    cpi = cpi.dropna()
    
    # Add required number of months to CPI trend
    addMonths = 12 - cpi['Month'].iloc[-1] + 1
    
    startMonth = cpi['Month'].iloc[-1]
    months = []
    for i in range(0, addMonths):
        if startMonth == 12:
            newMonth = 1
        else:
            newMonth = startMonth + 1
        months.append(newMonth)
        startMonth = newMonth
    
    startYear = int(cpi['Year'].iloc[-1])
    years = [startYear + 1 if i == 1 else startYear for i in months]
    
    # Create and append additional months
    adds = pd.DataFrame({
        'Year': years,
        'Month': months,
        'CPI': np.nan
    })
    
    cpi = pd.concat([cpi, adds], ignore_index=True)
    cpi = cpi.reset_index(drop=True)
    
    # Interpolate missing values and create match column
    cpi['CPI'] = cpi['CPI'].interpolate(method='spline', order=1)
    # Create Match column by combining Month and Year as strings
    cpi['Month'] = cpi['Month'].astype(int)
    cpi['Year'] = cpi['Year'].astype(int)
    cpi['Match'] = cpi['Month'].astype(str) + cpi['Year'].astype(str)
    cpi = cpi.drop_duplicates(subset=['Match'])
    cpi['Date'] = pd.to_datetime(cpi.Year*10000+cpi.Month*100+1,format='%Y%m%d')
    
    return cpi

def get_gdp_data():
    """Scrape and process GDP data for economic proxy."""
    try:
        url = 'https://www.multpl.com/us-gdp/table/by-quarter'
        gdp = pd.read_html(url)
        gdp = gdp[0]
    except Exception as e:
        print(f"\nWarning: Could not fetch GDP data: {str(e)}")
        print("Using placeholder GDP data for testing...")
        # Create placeholder data
        dates = []
        values = []
        # Get data for the last 15 years
        current_year = pd.Timestamp.now().year
        for year in range(current_year - 14, current_year + 1):
            for month in range(1, 13):
                dates.append(pd.Timestamp(year=year, month=month, day=1))
                values.append(2.0)
        gdp = pd.DataFrame({'Date': dates, 'GDP': values})
        return gdp
    gdp['Year'] = pd.to_numeric(gdp['Date'].str[-4:])
    gdp['Month'] = gdp['Date'].str[:3]
    gdp.rename(columns={'Value': 'GDP'}, inplace=True)
    gdpMonths = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    gdp.Month = gdp.Month.map(gdpMonths)
    gdp = gdp.sort_values(by=['Year', 'Month'])
    gdp['GDP'] = gdp['GDP'].str.replace(' trillion', '')
    gdp['GDP'] = pd.to_numeric(gdp['GDP'])
    gdp['Date'] = pd.to_datetime(gdp.Year*10000+gdp.Month*100+1,format='%Y%m%d')
    
    return gdp

def get_prime_rate_data():
    """Scrape and process prime interest rate data."""
    try:
        url = 'http://www.fedprimerate.com/wall_street_journal_prime_rate_history.htm'
        prime = pd.read_html(url)
        prime = prime[3]
        prime = prime.drop(columns=[2])
        prime = prime[pd.to_numeric(prime[1], errors='coerce').notnull()]
        prime[1] = pd.to_numeric(prime[1])
        prime[1] = prime[1] / 100
        prime.rename(columns={0: 'Date', 1: 'Prime Rate'}, inplace=True)
        prime['Date'] = pd.to_datetime(prime['Date'])
        return prime
    except Exception as e:
        logging.warning(f"Could not fetch prime rate data: {str(e)}")
        logging.info("Using placeholder prime rate data for testing...")
        # Create placeholder data
        # Get data for the last 15 years
        current_year = pd.Timestamp.now().year
        years = range(current_year - 14, current_year + 1)
        months = range(1, 13)
        data = [(f"{y}-{m:02d}-01", 0.05) for y in years for m in months]
        prime = pd.DataFrame(data, columns=['Date', 'Prime Rate'])
        prime['Date'] = pd.to_datetime(prime['Date'])
        return prime

def combine_all_data(balance_gp, financials_df, economic_indicators):
    """Combine balance GP data, financial data, and economic indicators into a single DataFrame."""
    # Standardize dates in balance_gp (our base dataset)
    balance_gp['Date'] = pd.to_datetime(balance_gp['Year'].astype(str) + '-' + balance_gp['Month'].astype(str) + '-01')
    
    # Standardize dates in financials_df
    financials_df['Date'] = pd.to_datetime(financials_df['date']).dt.to_period('M').dt.to_timestamp()
    
    # Standardize dates in economic_indicators
    economic_indicators['Date'] = pd.to_datetime(economic_indicators['Date']).dt.to_period('M').dt.to_timestamp()
    
    # Filter economic indicators to start from 2000
    min_date = pd.Timestamp('2000-01-01')
    economic_indicators = economic_indicators[economic_indicators['Date'] >= min_date].copy()
    financials_df = financials_df[financials_df['Date'] >= min_date].copy()
    
    # Remove duplicates from economic indicators by keeping the last value for each date
    economic_indicators = economic_indicators.sort_values('Date').drop_duplicates('Date', keep='last')
    
    print("\nChecking for duplicates before merge:")
    print(f"Balance GP duplicated dates: {balance_gp['Date'].duplicated().sum()}")
    print(f"Financials duplicated dates: {financials_df['Date'].duplicated().sum()}")
    print(f"Economic duplicated dates: {economic_indicators['Date'].duplicated().sum()}")
    
    # First merge: balance_gp with financials (left merge)
    combined_data = balance_gp.merge(financials_df, on='Date', how='left')
    print(f"\nRows after first merge: {len(combined_data)}")
    
    # Second merge: result with economic indicators (left merge)
    combined_data = combined_data.merge(economic_indicators, on='Date', how='left')
    print(f"\nRows after second merge: {len(combined_data)}")
    
    # Print available columns
    print("\nAvailable columns after merge:")
    print(combined_data.columns.tolist())
    
    # Sort by date
    combined_data = combined_data.sort_values('Date')
    
    # Create 12-month and 15-month forward sums of shiftGP
    print("\nCreating aggregated GP columns...")
    combined_data['shiftGP12'] = combined_data['shiftGP'].rolling(window=12, min_periods=12).sum().shift(-11)
    combined_data['shiftGP15'] = combined_data['shiftGP'].rolling(window=15, min_periods=15).sum().shift(-14)
    
    # Create target variables for the aggregate model
    combined_data['next_12mo_gp'] = combined_data['shiftGP12']
    combined_data['next_15mo_gp'] = combined_data['shiftGP15']
    
    # Print date ranges and null counts
    print("\nDate ranges in each dataset:")
    print(f"Balance GP: {balance_gp['Date'].min()} to {balance_gp['Date'].max()}")
    print(f"Financials: {financials_df['Date'].min()} to {financials_df['Date'].max()}")
    print(f"Economic Indicators: {economic_indicators['Date'].min()} to {economic_indicators['Date'].max()}")
    
    return combined_data

def generate_rolling_predictions(result, feature_columns):
    """Generate predictions for each row using only prior data."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    predictions = []
    min_samples = 12  # Minimum number of samples needed to train
    
    # Create a copy of the result DataFrame with only required columns
    data = result[feature_columns + ['constGP']].copy()
    
    # Forward fill any missing values in features
    data[feature_columns] = data[feature_columns].fillna(method='ffill')
    
    # If any NaN values remain after forward fill, backward fill them
    data[feature_columns] = data[feature_columns].fillna(method='bfill')
    
    for i in range(min_samples, len(data)-1):
        # Get training data up to current point
        X_train = data[feature_columns].iloc[:i+1].copy()
        y_train = data['constGP'].iloc[:i+1].copy()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = Ridge(alpha=0.1)
        model.fit(X_train_scaled, y_train)
        
        # Get next row for prediction
        X_pred = data[feature_columns].iloc[[i+1]]
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        pred = model.predict(X_pred_scaled)[0]
        
        # Store prediction
        predictions.append(pred)
        
        if (i + 1) % 12 == 0:  # Print progress every 12 months
            print(f"Processed {i + 1} of {len(data)-1} rows")
    
    # Create DataFrame with predictions
    monthly_pred_df = pd.DataFrame(predictions, columns=['MonthlyGPProjections'])
    
    # Create a DataFrame with the same index as the original data
    aligned_predictions = pd.DataFrame(index=data.index, columns=['MonthlyGPProjections'])
    
    # Fill in predictions starting from min_samples
    aligned_predictions.iloc[min_samples:min_samples + len(predictions)] = monthly_pred_df.values
    
    # Combine with original data
    monthly_gp_results = pd.concat([result, aligned_predictions], axis=1).copy()
    
    # Save to CSV
    monthly_gp_results.to_csv('/tmp/monthly_gp_model_performance.csv')
    print(f"\nSaved predictions to /tmp/monthly_gp_model_performance.csv")
    
    return monthly_gp_results

def predict_next_periods(result, feature_columns, balanceGP, monthly_pred_df=None, num_periods=12):
    """Predict multiple future periods' GP using horizon-specific models."""
    import numpy as np
    
    # Train models for each horizon
    horizon_models, horizon_scalers, horizon_metrics = train_and_compare_models(result)
    
    # Base features excluding the month columns
    base_features = ['gpBacklog', 'backlogPct','constGPNW', '1', 'rollingGPNW',
                   'rolling_12MoConst', 'CPI', 'GDP', 'PrimeRate']
    
    # Get latest row for prediction
    last_row = result[base_features].tail(1).copy()
    last_values = last_row.iloc[0].copy()
    
    # Initialize lists for predictions and feature importance
    predictions = []
    feature_importances = []
    
    # Ensure index is datetime and get the last date
    result.index = pd.to_datetime(result.index)
    current_date = result.index[-1]
    
    print("\nPredictions for Future Periods:")
    
    # Create a running state of features that gets updated each period
    current_features = last_values.copy()
    
    # Get the Excel projections for financial features
    excel_projections = {}
    for col in range(1, len(balanceGP.columns)):
        if str(col) in balanceGP.columns:
            excel_projections[col] = float(balanceGP.iloc[-1, balanceGP.columns.get_loc(str(col))])
    
    # Convert current_date to Timestamp if it's not already
    current_date = pd.Timestamp(current_date)
    
    for period in range(1, num_periods + 1):
        # Calculate the date for this period
        future_date = current_date + pd.DateOffset(months=period)
        
        # Only update the month indicator for December
        current_features['is_december'] = 1 if pd.Timestamp(future_date).month == 12 else 0
        
        # Get the model and scaler for this horizon
        model = horizon_models[period]
        scaler = horizon_scalers[period]
        
        # Prepare features for this horizon
        pred_features = current_features.copy()
        pred_features[str(period)] = 1  # Set the corresponding month column to 1
        
        # Create feature row and scale
        feature_cols = base_features + [str(period)]
        pred_row = pd.DataFrame([pred_features])[feature_cols]
        X_pred_scaled = scaler.transform(pred_row)
        
        # Make prediction
        prediction = model.predict(X_pred_scaled)[0]
        
        # Get the Excel projection for this period if available
        if period <= len(balanceGP.columns)-1:
            excel_projection = balanceGP.iloc[-1, period]
        else:
            excel_projection = None
            
        # Store features for this period
        feature_importances.append(pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.coef_
        }).sort_values('Importance', ascending=False))
        
        # Get Excel projection if available
        excel_projection = float(balanceGP.iloc[-1, period]) if period <= len(balanceGP.columns)-1 else None
        
        if excel_projection is not None:
            # Only enforce non-negative constraint if Excel projection is non-negative
            if excel_projection >= 0:
                prediction = max(0, min(prediction, excel_projection * 1.5))
                prediction = max(prediction, excel_projection * 0.5)
            else:
                # For negative Excel projections, just keep within bounds without non-negative constraint
                prediction = min(max(prediction, excel_projection * 1.5), excel_projection * 0.5)
        else:
            # If no Excel projection, allow negative values
            prediction = prediction
            
        # Print prediction details
        print(f"\nMonth {period} ({pd.Timestamp(future_date).strftime('%B %Y')}) Prediction:")
        print(f"Model Prediction: ${prediction:,.2f}")
        if excel_projection is not None:
            print(f"Excel Projection: ${excel_projection:,.2f}")
        
        # Print feature importance
        print(f"\nFeature Importance:")
        for _, row in feature_importances[-1].iterrows():
            print(f"{row['Feature']:<20}: {row['Importance']:>10.4f}")
        
        # Add prediction to list
        predictions.append(prediction)
    
    # Get last 6 months of data for actuals and predictions
    last_6_months = result.tail(6).copy()
    # Print summary of model performance by horizon
    print("\nModel Performance Summary by Horizon:")
    print(f"{'Horizon':<8} {'RMSE':>12} {'MAE':>12} {'R2':>8}")
    print("-" * 42)
    for horizon, metrics in horizon_metrics.items():
        print(f"{horizon:<8} {metrics['rmse']:>12,.2f} {metrics['mae']:>12,.2f} {metrics['r2']:>8.3f}")
    
    # Get the last 6 months of actual data for comparison
    last_6_months = result.tail(6)
    
    if monthly_pred_df is not None:
        past_predictions = monthly_pred_df['MonthlyGPProjections'].iloc[-6:]
        past_predictions.iloc[-1] = balanceGP.iloc[-1]['1']  # Column '1' is 1-month ahead
    else:
        past_predictions = pd.Series([None] * 6)
    
    # For December 2024 actual, use the Excel projection
    last_6_months.iloc[-1, last_6_months.columns.get_loc('shiftGP')] = balanceGP.iloc[-1]['1']  # Column '1' is 1-month ahead
    
    # Create future dates starting from the month after our last data point
    future_dates = pd.date_range(start=current_date + pd.DateOffset(months=1), periods=num_periods, freq='MS')  # MS = Month Start frequency
    
    # Create the combined DataFrame
    combined_df = pd.DataFrame({
        'Date': list(last_6_months.index) + list(future_dates),
        'Predicted_GP': [None] * 6 + predictions,
        'Past_Predicted_GP': list(past_predictions) + [None] * num_periods,
        'Actual_GP': list(last_6_months['shiftGP']) + [None] * num_periods
    })
    
    # Add Excel projections if available
    excel_projections = []
    for period in range(1, num_periods + 1):
        if period <= len(balanceGP.columns)-1:
            excel_projections.append(float(balanceGP.iloc[-1, period]))
        else:
            excel_projections.append(None)
    combined_df['Excel_Projection'] = [None] * 6 + excel_projections
    
    # Set the Date as index and ensure it's datetime
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df.set_index('Date', inplace=True)
    
    return combined_df, feature_importances, horizon_metrics
    

def train_and_compare_models(result):
    """Train and compare different models for GP prediction."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import numpy as np
    
    # Create shifted GP columns for each prediction horizon
    for i in range(1, 16):  # 1 to 15 months ahead
        result[f'shiftGP{i}'] = result['totalGP'].shift(-i)
    
    # Base features excluding the month columns
    base_features = ['gpBacklog', 'constGPNW', 'CPI', 'GDP', 'Prime Rate', 'rollingGPNW', 'rolling_12MoConst', 'is_december']
    
    # Dictionary to store models for each prediction horizon
    horizon_models = {}
    horizon_scalers = {}
    horizon_metrics = {}
    
    # Train a separate model for each prediction horizon
    for horizon in range(1, 16):
        print(f"\nTraining model for {horizon}-month horizon:")
        
        # Features for this horizon include base features plus the corresponding month column
        feature_columns = base_features + [str(horizon)]
        X = result[feature_columns].copy()
        y = result[f'shiftGP{horizon}'].copy()
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = Ridge(alpha=0.1)
        model.fit(X_scaled, y)
        
        # Store model and scaler
        horizon_models[horizon] = model
        horizon_scalers[horizon] = scaler
        
        # Calculate metrics
        y_pred = model.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${mae:,.2f}")
        print(f"R2 Score: {r2:.3f}")
        print(f"Cross-validation R2: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Store metrics
        horizon_metrics[horizon] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
    
    return horizon_models, horizon_scalers, horizon_metrics
    
    # Try different scaling methods
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    
    # RobustScaler is less sensitive to outliers
    robust_scaler = RobustScaler()
    X_robust = robust_scaler.fit_transform(X)
    
    # MinMaxScaler scales to a fixed range [0,1]
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    
    # StandardScaler for comparison
    standard_scaler = StandardScaler()
    X_standard = standard_scaler.fit_transform(X)
    
    # Use RobustScaler as default
    X_scaled = X_robust
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Define models to try
    models = {
        'Ridge Regression': Ridge(alpha=0.1)
    }
    
    # Number of train/test splits to try
    n_splits = 10
    
    # Store results for each model across all splits
    all_results = {name: {'rmse': [], 'mae': [], 'r2': [], 'cv_r2': []} for name in models.keys()}
    
    # Run multiple train/test splits
    for split in range(n_splits):
        # Split data with different random state each time
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=split)
    
        # Train and evaluate each model
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results for this split
            all_results[name]['rmse'].append(rmse)
            all_results[name]['mae'].append(mae)
            all_results[name]['r2'].append(r2)
            
            # Regular cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            all_results[name]['cv_r2'].append(cv_scores.mean())
    
    # Calculate average metrics across all splits
    results = {}
    for name in models.keys():
        results[name] = {
            'RMSE': np.mean(all_results[name]['rmse']),
            'RMSE_std': np.std(all_results[name]['rmse']),
            'MAE': np.mean(all_results[name]['mae']),
            'MAE_std': np.std(all_results[name]['mae']),
            'R2': np.mean(all_results[name]['r2']),
            'R2_std': np.std(all_results[name]['r2']),
            'CV_R2_mean': np.mean(all_results[name]['cv_r2']),
            'CV_R2_std': np.std(all_results[name]['cv_r2']),
            'model': models[name],
            'feature_importance': None
        }
        
        # Get feature importance from the last trained model
        if hasattr(models[name], 'feature_importances_'):
            results[name]['feature_importance'] = dict(zip(feature_columns, models[name].feature_importances_))
        elif hasattr(models[name], 'coef_'):
            results[name]['feature_importance'] = dict(zip(feature_columns, models[name].coef_))
    
    return results, X_scaled, y, robust_scaler, n_splits

def create_prediction_set(combined_data):
    """Create prediction set with rolling training and evaluation using Ridge and Elastic Net models."""
    # Define feature columns for X (removed is_December)
    feature_cols = ['gpBacklog', 'backlogPct','constGPNW', '1', 'rollingGPNW',
                   'rolling_12MoConst', 'CPI', 'GDP', 'PrimeRate']
    
    # Sort data by date
    combined_data = combined_data.sort_values('Date')
    
    # Initialize models with parameter grids
    models = {
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
        },
        'Elastic Net': {
            'model': ElasticNet(random_state=42),
            'params': {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
        }
    }
    
    all_results = {}
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Start after 12 months of data
    min_training_periods = 12
    
    # For each model
    for model_name, model_info in models.items():
        logging.info(f"\nTraining {model_name}...")
        results = pd.DataFrame()
        
        # Prepare data for grid search
        X = combined_data[feature_cols]
        y = combined_data['shiftGP']
        
        # Scale features
        X_scaled = scaler.fit_transform(X)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1
        )
        
        logging.info("Performing grid search...")
        grid_search.fit(X_scaled, y)
        
        logging.info(f"Best parameters: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        
        # Rolling prediction for actual results
        for i in range(min_training_periods, len(combined_data)):
            train_data = combined_data.iloc[:i]
            X_train = train_data[feature_cols]
            y_train = train_data['shiftGP']
            X_pred = combined_data.iloc[i:i+1][feature_cols]
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_pred_scaled = scaler.transform(X_pred)
            
            best_model.fit(X_train_scaled, y_train)
            prediction = best_model.predict(X_pred_scaled)[0]
            
            current_date = combined_data.iloc[i]['Date']
            actual = combined_data.iloc[i]['shiftGP']
            
            results = pd.concat([results, pd.DataFrame({
                'Date': [current_date],
                'Predicted_GP': [prediction],
                'Actual_GP': [actual]
            })])
        
        # Calculate metrics
        mae = mean_absolute_error(results['Actual_GP'], results['Predicted_GP'])
        r2 = r2_score(results['Actual_GP'], results['Predicted_GP'])
        
        logging.info(f"\nFinal Model Performance:")
        logging.info(f"Mean Absolute Error: ${mae:,.2f}")
        logging.info(f"R-squared Score: {r2:.3f}")
        
        # Analyze feature importance
        logging.info(f"\nFeature Importance Analysis for {model_name}:")
        feature_importance = analyze_feature_importance(best_model, feature_cols, X_scaled, y)
        
        # Store results
        all_results[model_name] = {
            'results': results,
            'mae': mae,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
    
    # Find best model based on MAE
    best_model = min(all_results.items(), key=lambda x: x[1]['mae'])
    logging.info(f"\nBest Model: {best_model[0]}")
    logging.info(f"MAE: ${best_model[1]['mae']:,.2f}")
    logging.info(f"R2: {best_model[1]['r2']:.3f}")
    logging.info(f"Best Parameters: {best_model[1]['best_params']}")
    
    # Save best model results to CSV
    best_results = best_model[1]['results']
    
    # Format the results
    best_results['Date'] = pd.to_datetime(best_results['Date'])
    best_results = best_results.sort_values('Date')
    
    # Drop rows where Actual_GP is 0 or null
    best_results = best_results[best_results['Actual_GP'].notna() & (best_results['Actual_GP'] != 0)]
    
    # Save to CSV
    best_results.to_csv('/tmp/gp_monthly_model_performance.csv', index=False)
    upload_to_blob('/tmp/gp_monthly_model_performance.csv', 'gp_monthly_model_performance.csv')
    logging.info(f"Successfully saved and uploaded gp_monthly_model_performance.csv to Azure blob storage")
    
    # Get the best model object and fit it on all data
    best_model_name = min(all_results.items(), key=lambda x: x[1]['mae'])[0]
    best_model_obj = models[best_model_name]['model']
    
    # Fit the model on all data
    X = combined_data[feature_cols]
    y = combined_data['shiftGP']
    X_scaled = scaler.fit_transform(X)
    best_model_obj.fit(X_scaled, y)
    
    # Get historical predictions and actuals
    historical_results = pd.DataFrame({
        'Date': combined_data['Date'],
        'Predicted_GP': best_model_obj.predict(X_scaled),
        'Actual_GP': combined_data['shiftGP']
    })
    historical_results['Date'] = pd.to_datetime(historical_results['Date'])
    
    # Get the actual data from financial_data (using constGP, not shiftGP)
    financial_data = get_financial_data()
    financial_data['date'] = pd.to_datetime(financial_data['date'])
    latest_actual_date = financial_data['date'].max()
    logging.info(f"Latest actual data available: {latest_actual_date}")
    
    # Create DataFrame with actual GP values (constGP)
    financial_gp = pd.DataFrame({
        'Date': financial_data['date'],
        'Actual_GP': financial_data['constGP']
    })
    
    # Get the last 6 months of data
    start_date = latest_actual_date - pd.DateOffset(months=5)  # 6 months including latest
    end_date = latest_actual_date
    
    # Create rows for the last 6 months of historical data
    rows = []
    current_date = start_date
    
    while current_date <= end_date:
        # Get actual GP for this month (constGP)
        actual_gp = financial_gp[financial_gp['Date'] == current_date]['Actual_GP'].iloc[0] if current_date in financial_gp['Date'].values else None
        
        # Get the prediction we made for this month
        # Note: historical_results contains predictions made using shiftGP, which is already aligned
        pred_gp = None
        if current_date in historical_results['Date'].values:
            hist_row = historical_results[historical_results['Date'] == current_date].iloc[0]
            pred_gp = hist_row['Predicted_GP']
        
        rows.append({
            'Date': current_date,
            'Historical_Predicted_GP': pred_gp,
            'Historical_Actual_GP': actual_gp,
            'Future_Projected_GP': None
        })
        
        current_date = current_date + pd.DateOffset(months=1)
    
    # Generate future predictions
    future_predictions = generate_future_predictions(best_model_obj, scaler, feature_cols, combined_data, latest_actual_date)
    
    # Add 12 months of future predictions starting from the month after latest_actual_date
    future_start = latest_actual_date + pd.DateOffset(months=1)
    future_end = future_start + pd.DateOffset(months=11)  # 12 months total
    
    current_date = future_start
    while current_date <= future_end:
        if current_date in future_predictions['Date'].values:
            pred = future_predictions[future_predictions['Date'] == current_date]['Projected_GP'].iloc[0]
            rows.append({
                'Date': current_date,
                'Historical_Predicted_GP': None,
                'Historical_Actual_GP': None,
                'Future_Projected_GP': pred
            })
        current_date = current_date + pd.DateOffset(months=1)
    
    # Create DataFrame
    projections_df = pd.DataFrame(rows)
    
    # Keep raw numeric values and datetime objects
    projections_df['Date'] = pd.to_datetime(projections_df['Date'])
    
    # Save projections to CSV and upload to Azure
    logging.info("\nSaving monthly projections with columns:")
    logging.info(projections_df.columns.tolist())
    logging.info("\nFirst 5 rows of projections:")
    logging.info(projections_df.head())
    
    projections_df.to_csv('/tmp/gp_monthly_model_projections.csv', index=False)
    upload_to_blob('/tmp/gp_monthly_model_projections.csv', 'gp_monthly_model_projections.csv')
    
    logging.info(f"Successfully uploaded gp_monthly_model_projections.csv to Azure blob storage with {len(projections_df)} rows")
    
    return best_model_obj, scaler, feature_cols

def generate_historical_horizon_predictions(data, feature_columns, target_column='totalGP', horizons=[12, 15]):
    """
    Generate historical predictions for specific horizons alongside actuals for comparison.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    
    results = pd.DataFrame(index=data.index)
    results['date'] = data.index
    min_training_rows = 24  # Minimum 2 years of data for training
    
    # Create imputer for handling missing values
    imputer = SimpleImputer(strategy='mean')
    
    for horizon in horizons:
        logging.info(f"\nGenerating predictions for {horizon}-month horizon...")
        
        # Create columns for this horizon
        results[f'actual_{horizon}mo'] = data[target_column]
        results[f'predicted_{horizon}mo'] = np.nan
        results[f'abs_error_{horizon}mo'] = np.nan
        results[f'pct_error_{horizon}mo'] = np.nan
        
        # Generate predictions
        for i in range(min_training_rows, len(data) - horizon):
            if i % 12 == 0:
                logging.info(f"Processed {i} of {len(data)-horizon} rows")
            
            # Get training data up to current point
            train_data = data.iloc[:i]
            future_point = data.iloc[i+horizon]
            
            # Prepare features
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            
            # Remove rows with NaN in target variable
            mask = ~y_train.isna()
            X_train = X_train[mask]
            y_train = y_train[mask]
            
            # Handle NaN in features
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            
            X_future = future_point[feature_columns].to_frame().T
            X_future = pd.DataFrame(imputer.transform(X_future), columns=X_future.columns, index=X_future.index)
            
            # Log data shapes and NaN counts
            logging.info(f"Training data shape after handling NaNs: {X_train.shape}")
            logging.info(f"Any remaining NaNs in features: {X_train.isna().sum().sum() > 0}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_future_scaled = scaler.transform(X_future)
            
            # Train model and predict
            model = Ridge(alpha=0.1)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_future_scaled)[0]
            
            # Store prediction and error metrics
            actual = future_point[target_column]
            results.loc[future_point.name, f'predicted_{horizon}mo'] = pred
            results.loc[future_point.name, f'abs_error_{horizon}mo'] = abs(actual - pred) if not pd.isna(actual) else np.nan
            results.loc[future_point.name, f'pct_error_{horizon}mo'] = abs((actual - pred) / actual) * 100 if actual != 0 and not pd.isna(actual) else np.nan
    
    return results

def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.utcnow().replace(
        tzinfo=timezone.utc).isoformat()

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)
    
    try:
        # Retrieve historical data
        historical_data = get_historical_balance_gp()
        logging.info("\nHistorical Data:")
        logging.info(historical_data.head())

        # Retrieve current projections and update if necessary
        updated_data, _, should_update, tabName, _ = get_balance_gp_projections()
        
        if should_update:
            logging.info(f"\nData was updated with new projections from tab {tabName}.")
        else:
            logging.info(f"\nNo update was necessary. Current projections from tab {tabName} match the historical data.")

        # Retrieve and print financial data
        financial_data = get_financial_data()
        logging.info("\nFinancial Data:")
        logging.info(financial_data.head())

        # Retrieve and print economic indicators
        economic_indicators = get_economic_indicators()
        logging.info("\nEconomic Indicators:")
        logging.info(economic_indicators.head())

        # Combine all data into a single DataFrame
        combined_data = combine_all_data(updated_data, financial_data, economic_indicators)
        logging.info("\nCombined Data:")
        logging.info(combined_data.head())
        
        # Calculate correlation between '1' and shiftGP
        correlation = combined_data['1'].corr(combined_data['shiftGP'])
        logging.info(f"\nCorrelation between '1' and shiftGP: {correlation:.3f}")

        # Analyze seasonality patterns
        logging.info("\nAnalyzing Seasonality Patterns...")
        monthly_stats, yoy_patterns, cv = analyze_seasonality(combined_data)

        # Verify '1' column values
        logging.info("\nVerifying '1' column values after data combination:")
        logging.info("Summary statistics for column '1':")
        logging.info(combined_data['1'].describe())
        logging.info("\nFirst 10 values of column '1':")
        logging.info(combined_data[['Date', '1']].head(10))

        # Create rolling predictions and evaluate performance
        best_model_obj, scaler, feature_cols = create_prediction_set(combined_data)
        
        # Generate predictions for all dates
        X = combined_data[feature_cols]
        X_scaled = scaler.transform(X)
        predictions = best_model_obj.predict(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': combined_data['Date'],
            'Predicted_GP': predictions,
            'Actual_GP': combined_data['shiftGP']
        })
        
        # Format the results
        results['Date'] = pd.to_datetime(results['Date'])
        results = results.sort_values('Date')
        
        # Drop rows where Actual_GP is 0 or null
        performance_results = results[results['Actual_GP'].notna() & (results['Actual_GP'] != 0)].copy()
        
        # Save performance metrics
        performance_results.to_csv('/tmp/gp_monthly_model_performance.csv', index=False)
        upload_to_blob('/tmp/gp_monthly_model_performance.csv', 'gp_monthly_model_performance.csv')
        logging.info(f"\nSaved and uploaded performance results to gp_monthly_model_performance.csv (after removing rows with null or zero Actual_GP)")
        
        logging.info("\nPrediction Results (first 5 rows):")
        logging.info(results.head())
        logging.info("\nPrediction Results (last 5 rows):")
        logging.info(results.tail())

        # Run aggregate models
        run_aggregate_models(combined_data)

        # Attempt to upload projection CSV files with error handling
        projection_files = ['gp_monthly_model_projections.csv', 'gp_monthly_model_performance.csv', 
                          'gp_long_range_projections.csv', 'gp_long_range_performance.csv']
        for proj_file in projection_files:
            if os.path.exists(proj_file):
                try:
                    upload_to_blob(proj_file, os.path.basename(proj_file))
                    logging.info(f"Successfully uploaded {proj_file} to Azure blob storage")
                except Exception as e:
                    logging.error(f"Error uploading {proj_file}: {e}")
            else:
                logging.warning(f"File {proj_file} does not exist, skipping upload.")
                
    except Exception as e:
        logging.error(f'Error in main function: {str(e)}')
        raise e
        raise

if __name__ == "__main__":
    main(None)
