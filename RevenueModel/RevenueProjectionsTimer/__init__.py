import azure.functions as func
import logging
import pandas as pd
import numpy as np
import os
import requests
import xgboost as xgb
from datetime import datetime, timezone
from calendar import monthrange
from azure.storage.fileshare import ShareServiceClient, ShareClient, ShareDirectoryClient, ShareFileClient
from azure.storage.blob import BlobServiceClient
from sklearn.linear_model import LinearRegression
import io
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
currentYear = datetime.now().year
currentMonth = datetime.now().month
bondCeiling = 700000000
temp_dir = tempfile.gettempdir()

def get_temp_path(filename):
    """Helper function to get full path in temp directory"""
    return os.path.join(temp_dir, filename)

def fetch_financial_data() -> dict:
    logging.info("Fetching financial data...")
    try:
        response = requests.get('https://vue.korteco.com/korteapi/intranet/GLMonthlyStatus', auth=('kortetech', 'F7DV%2%Yn+jn/aq=w2W/'))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error fetching financial data: {e}")
        return {}

def process_financial_data(financials: dict) -> pd.DataFrame:
    logging.info("Processing financial data...")
    financials_df = pd.DataFrame.from_dict(financials)
    financials_df['date'] = pd.to_datetime(financials_df['mth'])
    financials_df['year'] = pd.to_numeric(financials_df['mth'].str[:4])
    financials_df['month'] = pd.to_numeric(financials_df['mth'].str[5:7])
    financials_df['rollingBacklog'] = financials_df['backlog'].shift().rolling(12).mean()
    financials_df['rollingRevenue'] = financials_df['constRevenue'].shift().rolling(12).sum()
    financials_df['rollingNW'] = financials_df['constNW'].shift().rolling(12).sum()
    financials_df['lastMthRevenue'] = financials_df['constRevenue'].shift(1)
    financials_df['next12'] = financials_df['constRevenue'].shift(-12).rolling(12).sum()
    financials_df['next15'] = financials_df['constRevenue'].shift(-15).rolling(15).sum()
    financials_df = financials_df[financials_df.constRevenue != 0]

    logging.info("Successfully fetched and processed financial data")

    # Calculate the margin of other work from the financials API call
    margin = financials_df.copy()
    margin = margin[['backlogPct']]
    margin = margin.dropna()
    margin = margin[margin['backlogPct'] != 0]
    otherMargin = margin['backlogPct'].iloc[-1]

    logging.info(f"Calculated other work margin: {otherMargin}")

    # Store the last entry from the financials data for later reference
    recentDate = financials_df['date'].iloc[-1]

    # Fetch real-time backlog data
    try:
        backlogCurrent = requests.get('https://vue.korteco.com/korteapi/intranet/backlogactual', auth=('kortetech', 'F7DV%2%Yn+jn/aq=w2W/'))
        backlogCurrent.raise_for_status()  # Raise an error for bad responses
        j = backlogCurrent.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
        return
    except requests.exceptions.JSONDecodeError as json_err:
        logging.error(f"JSON decode error occurred: {json_err}")
        return

    backlogCurrent = pd.DataFrame.from_dict(j)
    backlogCurrent = backlogCurrent[backlogCurrent['sort'] != 2]
    backlogCurrent = backlogCurrent['backlog'].sum()

    oldBacklog = financials_df.loc[financials_df.index[-1], 'backlog']  # Capture backlog at closing for charting
    financials_df.loc[financials_df.index[-1], 'backlog'] = backlogCurrent  # Replace last month's backlog with actual

    logging.info("Successfully updated financials with real-time backlog data")

    logging.info(f"Stored recent date for reference: {recentDate}")

    logging.info("Successfully fetched and processed financial data")

    # Fetch backlog projections from Salesforce
    try:
        backlogData = requests.get('https://vue.korteco.com/korteapi/intranet/BacklogProjections', auth=('kortetech', 'F7DV%2%Yn+jn/aq=w2W/'))
        backlogData.raise_for_status()  # Raise an error for bad responses
        j = backlogData.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
        return
    except requests.exceptions.JSONDecodeError as json_err:
        logging.error(f"JSON decode error occurred: {json_err}")
        return

    backlogData = pd.DataFrame.from_dict(j)
    backlogData['date'] = pd.to_datetime(backlogData['mth'])
    backlogData['year'] = pd.to_numeric(backlogData['mth'].str[:4])
    backlogData['month'] = pd.to_numeric(backlogData['mth'].str[5:7])
    
    backlogProbables = backlogData.drop_duplicates(subset = ['contract'], keep = 'first').copy() # remove any duplicate contract numbers (e.g. JV jobs, split contracts like BJC, etc.)
    backlogProbables = backlogProbables[['date', 'amount', 'description', 'probability']].copy()
    backlogProbables['probability'] = backlogProbables['probability'] / 100
    backlogProbables['probAmount'] = backlogProbables['probability'] * backlogProbables['amount']
    backlogProbables.dropna(subset = ['probAmount'], inplace = True)
    backlogProbables = backlogProbables.groupby('date')['probAmount'].sum()
    backlogProbables = backlogProbables.reset_index()
    backlogProbables.columns = ['date', 'probAmount']

    logging.info("Successfully fetched and processed backlog projections")

    # Create a small dataset for deriving projections
    out = financials_df[150:] # Remove enough lines to drop 0s in ConstRevenue before it was being tracked
    out = out[out['constRevenue'] != 1] # to acommodate 1 for construction revenue recorded in future December
    out = out[['rollingBacklog', 'rollingRevenue', 'rollingNW', 'lastMthRevenue', 'month', 'year']]
    out = out.loc[financials_df['constRevenue'].shift(1) != financials_df['constRevenue']] # Remove all trailing 0s except for next month
    out = out.tail(1)

    logging.info("Created small dataset for deriving projections")

    # Create master dataset for predictions
    revenue = financials_df[['date', 'constRevenue', 'rollingBacklog', 'rollingRevenue', 'rollingNW', 'lastMthRevenue', 'month', 'year']].copy()
    revenue = revenue[revenue['constRevenue'] != 0]  # Drop rows without constRevenue data
    revenue = revenue[revenue['lastMthRevenue'] != 0]  # Drop rows without lastMthRevenue data

    logging.info("Created master dataset")

    # Create the first output to evaluate model performance over time
    recursiveRev = financials_df[['date', 'constRevenue', 'rollingBacklog', 'rollingRevenue', 'rollingNW', 'lastMthRevenue', 'month', 'year']].copy()
    recursiveRev = recursiveRev[recursiveRev['constRevenue'] != 0]  # Drop rows without constRevenue data
    recursiveRev = recursiveRev[recursiveRev['lastMthRevenue'] != 0]  # Drop rows without lastMthRevenue data
    recursiveRev.dropna(inplace=True)  # Drop missing data

    revenueProjections = []

    for i in range(1, len(recursiveRev)):
        X = recursiveRev[['month', 'year', 'rollingNW', 'rollingBacklog', 'rollingRevenue', 'lastMthRevenue']][0:i].copy()  # Make X past data
        y = recursiveRev[['constRevenue']][0:i].copy()  # Make y past data
        proj = recursiveRev[['month', 'year', 'rollingNW', 'rollingBacklog', 'rollingRevenue', 'lastMthRevenue']].iloc[[i]].copy()  # Make projection unseen data

        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X, y)
        y_predict = model.predict(proj)
        revenueProjections.append(y_predict)

    revenueProjections = pd.DataFrame(revenueProjections)
    revenueProjections.reset_index(drop=True, inplace=True)
    revenueProjections.rename(columns={0: 'projections'}, inplace=True)

    revenueActuals = recursiveRev[['date', 'constRevenue']][1:]
    revenueActuals.reset_index(drop=True, inplace=True)
    revenuePerformance = pd.concat([revenueActuals, revenueProjections], axis=1)
    revenuePerformance.to_csv(get_temp_path('revenuePerformance.csv'))

    logging.info("Generated revenue performance output")

    # Predict 12 months of revenue using recursive model outputs
    X = revenue[['month', 'year', 'rollingNW', 'rollingBacklog', 'rollingRevenue', 'lastMthRevenue']].copy()
    y = revenue[['constRevenue']].copy() 

    model = xgb.XGBRegressor(objective = 'reg:squarederror')
    model.fit(X, y) # Retrain model on entire dataset for output generation

    recursiveOut = out.copy()
    revenueProjections = financials_df[['date', 'constRevenue', 'backlog', 'rollingBacklog', 'rollingRevenue', 'constNW', 'rollingNW', 'lastMthRevenue', 'month', 'year']]
    revenueProjections = revenueProjections [revenueProjections ['constRevenue'] != 0]
    revenueProjections = revenueProjections [revenueProjections ['constRevenue'] != 1] # to acommodate 1 for construction revenue recorded in future December
    revenueProjections.drop(columns = ['date'], inplace = True)

    for i in range(1,13):
        recursiveIn = recursiveOut
        recursiveIn = recursiveIn[['month', 'year', 'rollingNW', 'rollingBacklog', 'rollingRevenue', 'lastMthRevenue']] # Reorder the columns to match the original
        recursivePredict = model.predict(recursiveIn) # Predict from the output df
        
        timeShiftPredict = pd.DataFrame(recursivePredict )[0].iloc[0] # Make predicted revenue into time-shifted revenue variable for use in recursive predictions
        
        revenueProjections['date'] = pd.to_datetime(revenueProjections.year * 10000 + revenueProjections.month * 100 + 1, format = '%Y%m%d') # Create Date column
        
        revenueProjections = revenueProjections.merge(backlogProbables, on = 'date', how = 'left') # Attach future expected backlog additions
        revenueProjections.fillna({'probAmount' : 0}, inplace = True)
        
        recursiveBacklog = revenueProjections['backlog'].iloc[-1] + revenueProjections['probAmount'].iloc[-1] - timeShiftPredict # New backlog is revenue prediction subtracted from old backlog
        
        shiftMonth = revenueProjections['month'].iloc[-1] # New month is last month plus one except if last month was December
        
        if shiftMonth == 12:
            shiftMonth = 1
        else:
            shiftMonth = shiftMonth + 1
        
        shiftYear = revenueProjections['year'].iloc[-1] # New year is same as last row unless last row was December
        
        if shiftMonth == 1:
            shiftYear = shiftYear + 1
        else:
            shiftYear = shiftYear
        
        shiftRow = {'constRevenue' : timeShiftPredict, 'backlog' : recursiveBacklog, 'rollingBacklog' : 0, 'rollingRevenue' : 0, 
                            'constNW' : 0, 'rollingNW' : 0, 'lastMthRevenue' : 0, 'month' : shiftMonth, 'year' : shiftYear} # Make new row from individual pieces
        
        revenueProjections = pd.concat([revenueProjections, pd.DataFrame([shiftRow])], ignore_index = True)
        revenueProjections['rollingBacklog'] = revenueProjections['backlog'].shift().rolling(12).mean()
        revenueProjections['rollingRevenue'] = revenueProjections['constRevenue'].shift().rolling(12).sum()

        revenueProjections['rollingNW'] = revenueProjections['constNW'].shift().rolling(12).sum()
        revenueProjections['lastMthRevenue'] = revenueProjections['constRevenue'].shift(1)
        revenueProjections.drop(columns = 'probAmount', inplace = True)
        
        recursiveOut  = revenueProjections[['month', 'year', 'rollingNW', 'rollingBacklog', 'rollingRevenue', 'lastMthRevenue']].tail(1) # Make output df from last row of newly revised time df

    revenueProjections = revenueProjections[-18:] # Keep only 12 forward-looking and 6 past actuals
    revenueProjections['prediction'] = revenueProjections['constRevenue'].iloc[-12:] # Make 12 future periods into 'Prediction' colum for charting purposes
    revenueProjections['constRevenue'] = revenueProjections['constRevenue'].iloc[0:6] # Make previous 6 periods of actuals available for charting
    revenueProjections['date'] = pd.to_datetime(revenueProjections.year * 10000 + revenueProjections.month * 100 + 1,format = '%Y%m%d')
    revenueProjections = revenueProjections.merge(revenuePerformance[['date', 'projections']], on = 'date', how = 'left')
    revenueProjections.to_csv(get_temp_path('revenueProjections.csv'))

    logging.info("Generated 12-month revenue projections")

    # Project revenue by silo
    revenueSilos = revenueProjections.copy()
    siloSplits = backlogData.copy()
    siloSplits = siloSplits.groupby(['date', 'pillar'])['revenueV2'].sum()
    siloSplits = siloSplits.reset_index()
    dod = siloSplits.copy()
    dc = siloSplits.copy()
    hc = siloSplits.copy()
    lv = siloSplits.copy()
    stl = siloSplits.copy()
    usps = siloSplits.copy()
    other = siloSplits.copy()

    dod = dod[dod['pillar'] == 'Government']
    dc = dc[dc['pillar'] == 'Distribution Center']
    hc = hc[hc['pillar'] == 'Healthcare']
    lv = lv[lv['pillar'] == 'Las Vegas']
    stl = stl[stl['pillar'] == 'St. Louis']
    usps = usps[usps['pillar'] == 'US Postal Service']
    other = other[~other['pillar'].isin(['Government', 'Distribution Center', 'Healthcare', 'Las Vegas', 'St. Louis', 'US Postal Service'])]

    dod.rename(columns={'revenueV2': 'DoD'}, inplace=True)
    dc.rename(columns={'revenueV2': 'DC'}, inplace=True)
    hc.rename(columns={'revenueV2': 'HC'}, inplace=True)
    lv.rename(columns={'revenueV2': 'LV'}, inplace=True)
    stl.rename(columns={'revenueV2': 'STL'}, inplace=True)
    usps.rename(columns={'revenueV2': 'USPS'}, inplace=True)
    other.rename(columns={'revenueV2': 'Other'}, inplace=True)

    siloSplits = dod.copy()
    siloSplits = siloSplits.drop(columns=['pillar'])
    siloSplits = siloSplits.merge(dc[['date', 'DC']], on='date', how='outer')
    siloSplits = siloSplits.merge(hc[['date', 'HC']], on='date', how='outer')
    siloSplits = siloSplits.merge(lv[['date', 'LV']], on='date', how='outer')
    siloSplits = siloSplits.merge(stl[['date', 'STL']], on='date', how='outer')
    siloSplits = siloSplits.merge(usps[['date', 'USPS']], on='date', how='outer')
    siloSplits = siloSplits.merge(other[['date', 'Other']], on='date', how='outer')
    siloSplits = siloSplits.fillna(0)
    siloSplits['total'] = siloSplits['DoD'] + siloSplits['DC'] + siloSplits['HC'] + siloSplits['LV'] + siloSplits['STL'] + siloSplits['USPS'] + siloSplits['Other']

    siloSplits['DoDPercent'] = siloSplits['DoD'] / siloSplits['total']
    siloSplits['DCPercent'] = siloSplits['DC'] / siloSplits['total']
    siloSplits['HCPercent'] = siloSplits['HC'] / siloSplits['total']
    siloSplits['LVPercent'] = siloSplits['LV'] / siloSplits['total']
    siloSplits['STLPercent'] = siloSplits['STL'] / siloSplits['total']
    siloSplits['USPSPercent'] = siloSplits['USPS'] / siloSplits['total']
    siloSplits['OtherPercent'] = siloSplits['Other'] / siloSplits['total']

    revenueSilos = revenueSilos.merge(siloSplits[['date', 'DoDPercent', 'DCPercent', 'HCPercent', 'LVPercent', 'STLPercent', 'USPSPercent', 'OtherPercent']], on='date')

    revenueSilos = revenueSilos[revenueSilos['date'] > recentDate]

    revenueSilos['DoDPrediction'] = revenueSilos['prediction'] * revenueSilos['DoDPercent']
    revenueSilos['DCPrediction'] = revenueSilos['prediction'] * revenueSilos['DCPercent']
    revenueSilos['HCPrediction'] = revenueSilos['prediction'] * revenueSilos['HCPercent']
    revenueSilos['STLPrediction'] = revenueSilos['prediction'] * revenueSilos['STLPercent']
    revenueSilos['LVPrediction'] = revenueSilos['prediction'] * revenueSilos['LVPercent']
    revenueSilos['USPSPrediction'] = revenueSilos['prediction'] * revenueSilos['USPSPercent']
    revenueSilos['OtherPrediction'] = revenueSilos['prediction'] * revenueSilos['OtherPercent']
    revenueSilos = revenueSilos[['date', 'prediction', 'DoDPrediction', 'DCPrediction', 'HCPrediction', 'LVPrediction', 'STLPrediction', 'USPSPrediction', 'OtherPrediction']]
    revenueSilos[['prediction', 'DoDPrediction', 'DCPrediction', 'HCPrediction', 'LVPrediction', 'STLPrediction', 'USPSPrediction', 'OtherPrediction']] = revenueSilos[['prediction', 'DoDPrediction', 'DCPrediction', 'HCPrediction', 'LVPrediction', 'STLPrediction', 'USPSPrediction', 'OtherPrediction']].apply(pd.to_numeric)
    revenueSilos.to_csv(get_temp_path('revenueSilosProjections.csv'))

    logging.info("Generated revenue silo projections")

    # Cut revenue projections by bonding percentages
    revenueBonds = revenueProjections.copy()
    bondProjection = backlogData.copy()
    bondProjection = bondProjection.groupby(['date', 'bond'])['revenueV2'].sum()
    bondProjection = bondProjection.reset_index()

    bondedRevenue = bondProjection.copy()
    unbondedRevenue = bondProjection.copy()
    bondedRevenue = bondedRevenue[bondedRevenue['bond'] != False]
    unbondedRevenue = unbondedRevenue[unbondedRevenue['bond'] != True]
    unbondedRevenue.rename(columns={'revenueV2': 'revenueV2_unbonded'}, inplace=True)
    bondedRevenue = bondedRevenue.merge(unbondedRevenue[['date', 'revenueV2_unbonded']], on='date', how='outer')
    bondedRevenue['total'] = bondedRevenue['revenueV2'] + bondedRevenue['revenueV2_unbonded']
    bondedRevenue['bondPercent'] = bondedRevenue['revenueV2'] / bondedRevenue['total']
    bondedRevenue['unbondedPercent'] = bondedRevenue['revenueV2_unbonded'] / bondedRevenue['total']

    bondedRevenue = bondedRevenue[bondedRevenue['date'] > recentDate]

    revenueBonds = revenueBonds.merge(bondedRevenue[['date', 'bondPercent', 'unbondedPercent']], on='date')
    revenueBonds['bondedPrediction'] = revenueBonds['prediction'] * revenueBonds['bondPercent']
    revenueBonds['unbondedPrediction'] = revenueBonds['prediction'] * revenueBonds['unbondedPercent']
    revenueBonds.to_csv(get_temp_path('revenueBondedProjections.csv'))

    logging.info("Generated revenue bonded projections")

    # Create bonding capacity ceiling model
    bondedWork = backlogData[backlogData['date'] == recentDate].groupby('bond')['backlog'].sum().loc[True]  # Get current bonded work
    revenueBonds['bondedBacklogPrediction'] = bondedWork - revenueBonds['bondedPrediction'].cumsum()
    revenueBonds['bondingCapacity'] = bondCeiling - revenueBonds['bondedBacklogPrediction']
    bondingRoom = revenueBonds[['date', 'bondedBacklogPrediction', 'bondingCapacity']].copy()
    bondingRoom.to_csv(get_temp_path('backlogBondedProjections.csv'))

    logging.info("Generated bonding capacity and backlog projections")

    # Create production staff needs projection
    headCount = backlogData.copy()
    headCount = headCount[headCount['revenueV2'] != 0]
    headCount = headCount.groupby(['date', 'pillar'])['revenueV2'].count()
    headCount = headCount.reset_index()

    # Staffing key
    silos = ['Distribution Center', 'Government', 'Healthcare', 'Las Vegas', 'St. Louis', 'US Postal Service', 'Other']
    silos_PM = [0.5, 1.0, 0.5, 0.5, 0.5, 1.5, 0.5]
    silos_PE = [0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    silos_SU = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    silos_QC = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    silos_SS = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    staffing = pd.DataFrame({'pillar': silos, 'PM Ct': silos_PM, 'PE Ct': silos_PE, 'Super Ct': silos_SU, 'QC Ct': silos_QC, 'Safety Ct': silos_SS})

    # Add staffing rules to jobs
    headCount = headCount.merge(staffing, on='pillar', how='left')
    headCount['PE'] = headCount['PE Ct'] * headCount['revenueV2']
    headCount['PM'] = headCount['PM Ct'] * headCount['revenueV2']
    headCount['Super'] = headCount['Super Ct'] * headCount['revenueV2']
    headCount['QC'] = headCount['QC Ct'] * headCount['revenueV2']
    headCount['Safety'] = headCount['Safety Ct'] * headCount['revenueV2']
    headCount = headCount[headCount['date'] > recentDate]
    headCount.to_csv(get_temp_path('headCountProjections.csv'))

    logging.info("Generated headcount projections")

    # Project backlog
    backlogProjection = financials_df[['date', 'backlog']].copy()
    backlogProjection = backlogProjection.merge(revenueProjections[['date', 'prediction']], on='date', how='right')  # Attach revenue projections
    backlogProjection = backlogProjection[['date', 'backlog', 'prediction']]
    backlogProjection['recBacklog'] = backlogProjection['backlog']
    backlogProjection['recBacklog'] = backlogProjection['recBacklog'].replace(np.nan, 0)  # Remove NaNs so sums work

    backlofRecursive = backlogData.drop_duplicates(subset=['contract'], keep='first').copy()  # Remove duplicate contract numbers
    backlofRecursive = backlofRecursive[['date', 'amount', 'probability']].copy()
    backlofRecursive['probability'] = backlofRecursive['probability'] / 100
    backlofRecursive['probAmount'] = backlofRecursive['probability'] * backlofRecursive['amount']
    backlofRecursive.dropna(subset=['probAmount'], inplace=True)
    backlofRecursive = backlofRecursive.groupby('date')['probAmount'].sum()
    backlofRecursive = backlofRecursive.reset_index()

    backlogProjection = backlogProjection.merge(backlofRecursive[['date', 'probAmount']], on='date', how='left')
    backlogProjection['probAmount'] = backlogProjection['probAmount'].replace(np.nan, 0)
    backlogProjection.loc[6, 'recBacklog'] = backlogProjection.iloc[5]['backlog'] - backlogProjection.iloc[6]['prediction'] + backlogProjection.iloc[6]['probAmount']
    for i in range(7, int(len(backlogProjection))):
        backlogProjection.loc[i, 'recBacklog'] = backlogProjection.iloc[i - 1]['recBacklog'] - backlogProjection.iloc[i]['prediction'] + backlogProjection.iloc[i]['probAmount']
    backlogProjection['recBacklog'] = np.where(backlogProjection['recBacklog'] < 0, 0, backlogProjection['recBacklog'])
    backlogProjection.to_csv(get_temp_path('backlogProjection.csv'), index=False)

    logging.info("Generated backlog projections")

    # Cut backlog projections by silo
    backlogSilos = revenueProjections.copy()

    backlogActuals = backlogData.groupby(['date', 'pillar'])['backlog'].sum()
    backlogActuals = backlogActuals.unstack(level = -1)
    backlogActuals.reset_index(inplace = True)
    backlogActuals.columns.name = None
    backlogActuals = backlogActuals.fillna(0)
    backlogActuals['total'] = backlogActuals['Distribution Center'] + backlogActuals['Government'] + backlogActuals['Healthcare'] + backlogActuals['Las Vegas'] + backlogActuals['St. Louis'] + backlogActuals['US Postal Service'] + backlogActuals['Other']
    backlogSilos = backlogSilos.merge(backlogActuals, on = 'date', how = 'left')
    backlogSilos = backlogSilos.merge(revenueSilos[['date', 'DoDPrediction', 'DCPrediction', 'HCPrediction', 'LVPrediction', 'STLPrediction', 'USPSPrediction']], on = 'date', how = 'left')

    backlogSilos['DCPct'] = backlogSilos['Distribution Center'] / backlogSilos['total']
    backlogSilos['DoDPct'] = backlogSilos['Government'] / backlogSilos['total']
    backlogSilos['HCPct'] = backlogSilos['Healthcare'] / backlogSilos['total']
    backlogSilos['LVPct'] = backlogSilos['Las Vegas'] / backlogSilos['total']
    backlogSilos['STLPct'] = backlogSilos['St. Louis'] / backlogSilos['total']
    backlogSilos['USPSPct'] = backlogSilos['US Postal Service'] / backlogSilos['total']
    backlogSilos['OTHPct'] = backlogSilos['Other'] / backlogSilos['total']

    backlogSilos['Distribution Center'] = backlogSilos['backlog'] * backlogSilos['DCPct']
    backlogSilos['Government'] = backlogSilos['backlog'] * backlogSilos['DoDPct']
    backlogSilos['Healthcare'] = backlogSilos['backlog'] * backlogSilos['HCPct']
    backlogSilos['Las Vegas'] = backlogSilos['backlog'] * backlogSilos['LVPct']
    backlogSilos['St. Louis'] = backlogSilos['backlog'] * backlogSilos['STLPct']
    backlogSilos['US Postal Service'] = backlogSilos['backlog'] * backlogSilos['USPSPct']
    backlogSilos['Other'] = backlogSilos['backlog'] * backlogSilos['OTHPct']

    backlogPillarProjection = backlogData.drop_duplicates(subset = ['contract'], keep = 'first').copy() # remove any duplicate contract numbers (e.g. JV jobs, split contracts like BJC, etc.)
    backlogPillarProjection = backlogPillarProjection[['date', 'amount', 'pillar', 'probability']].copy()
    backlogPillarProjection['probability'] = backlogPillarProjection['probability'] / 100
    backlogPillarProjection['probAmount'] = backlogPillarProjection['probability'] * backlogPillarProjection['amount']
    backlogPillarProjection.dropna(subset = ['probAmount'], inplace = True)
    backlogPillarProjection = backlogPillarProjection.groupby(['date', 'pillar'])['probAmount'].sum()
    backlogPillarProjection = backlogPillarProjection.unstack(level = -1)
    backlogPillarProjection.columns.name = None
    backlogPillarProjection.reset_index(drop = False, inplace = True)
    backlogPillarProjection.rename(columns = {'Distribution Center': 'DC_BL', 'Government': 'GOV_BL', 'Healthcare': 'HC_BL', 'Las Vegas': 'LV_BL', 'Other': 'OTH_BL', 'St. Louis': 'STL_BL', 'US Postal Service': 'USPS_BL'}, inplace = True)
    backlogPillarProjection = backlogPillarProjection.fillna(0)

    backlogSilos = backlogSilos.merge(backlogPillarProjection, on = 'date', how = 'left')

    backlogSilos.fillna(0, inplace = True)

    for i in range(6, 18):
        try:
            backlogSilos.loc[i, 'Distribution Center'] = (backlogSilos.loc[i - 1, 'Distribution Center'] - 
            backlogSilos.loc[i, 'DCPrediction'] + backlogSilos.loc[i, 'DC_BL'])
        except KeyError:
            backlogSilos.loc[i, 'Distribution Center'] = (backlogSilos.loc[i - 1, 'Distribution Center'] - 
            backlogSilos.loc[i, 'DCPrediction'])

        try:
            backlogSilos.loc[i, 'Government'] = (backlogSilos.loc[i - 1, 'Government'] - 
            backlogSilos.loc[i, 'DoDPrediction'] + backlogSilos.loc[i, 'GOV_BL'])
        except KeyError:
            backlogSilos.loc[i, 'Government'] = (backlogSilos.loc[i - 1, 'Government'] - 
            backlogSilos.loc[i, 'DoDPrediction'])

        try:
            backlogSilos.loc[i, 'Healthcare'] = (backlogSilos.loc[i - 1, 'Healthcare'] - 
            backlogSilos.loc[i, 'HCPrediction'] + backlogSilos.loc[i, 'HC_BL'])
        except KeyError:
            backlogSilos.loc[i, 'Healthcare'] = (backlogSilos.loc[i - 1, 'Healthcare'] - 
            backlogSilos.loc[i, 'HCPrediction'])

        try:
            backlogSilos.loc[i, 'St. Louis'] = (backlogSilos.loc[i - 1, 'St. Louis'] - 
            backlogSilos.loc[i, 'STLPrediction'] + backlogSilos.loc[i, 'STL_BL'])
        except KeyError:
            backlogSilos.loc[i, 'St. Louis'] = (backlogSilos.loc[i - 1, 'St. Louis'] - 
            backlogSilos.loc[i, 'STLPrediction'])

        try:
            backlogSilos.loc[i, 'Las Vegas'] = (backlogSilos.loc[i - 1, 'Las Vegas'] - 
            backlogSilos.loc[i, 'LVPrediction'] + backlogSilos.loc[i, 'LV_BL'])
        except KeyError:
            backlogSilos.loc[i, 'Las Vegas'] = (backlogSilos.loc[i - 1, 'Las Vegas'] - 
            backlogSilos.loc[i, 'LVPrediction'])

        try:
            backlogSilos.loc[i, 'US Postal Service'] = (backlogSilos.loc[i - 1, 'US Postal Service'] - 
            backlogSilos.loc[i, 'USPSPrediction'] + backlogSilos.loc[i, 'USPS_BL'])
        except KeyError:
            backlogSilos.loc[i, 'US Postal Service'] = (backlogSilos.loc[i - 1, 'US Postal Service'] - 
            backlogSilos.loc[i, 'USPSPrediction'])

    backlogSilos['OldDC'] = backlogSilos['Distribution Center'][0:6]
    backlogSilos['OldDoD'] = backlogSilos['Government'][0:6]
    backlogSilos['OldHC'] = backlogSilos['Healthcare'][0:6]
    backlogSilos['OldSTL'] = backlogSilos['St. Louis'][0:6]
    backlogSilos['OldLV'] = backlogSilos['Las Vegas'][0:6]
    backlogSilos['OldUSPS'] = backlogSilos['US Postal Service'][0:6]
    backlogSilos.loc[:5, 'Distribution Center'] = np.nan
    backlogSilos.loc[:5, 'Government'] = np.nan
    backlogSilos.loc[:5, 'Healthcare'] = np.nan
    backlogSilos.loc[:5, 'St. Louis'] = np.nan
    backlogSilos.loc[:5, 'Las Vegas'] = np.nan
    backlogSilos.loc[:5, 'US Postal Service'] = np.nan
    backlogSilos.to_csv(get_temp_path('backlogSiloProjections.csv'))

    logging.info("Generated backlog silo projections")

    # Get imputed revenue from historical balance gross profit projections
    from azure.storage.fileshare import ShareServiceClient, ShareClient, ShareDirectoryClient, ShareFileClient
    import io

    storage_account_url = 'https://tkcpowerbidata.file.core.windows.net'
    storage_account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
    service_client = ShareServiceClient(account_url=storage_account_url, credential=storage_account_key)

    share_name = 'financial'
    directory_path = 'balancegp'
    file_name = 'balanceRevenue.csv'

    share_client = service_client.get_share_client(share_name)
    directory_client = share_client.get_directory_client(directory_path)

    file_client = directory_client.get_file_client(file_name)
    file_stream = io.BytesIO()
    file_client.download_file().readinto(file_stream)

    # Set the stream position to the beginning
    file_stream.seek(0)

    # Read the Excel file into a DataFrame
    balanceRevenue = pd.read_csv(file_stream)
    balanceRevenue['date'] = pd.to_datetime(balanceRevenue['Date'], format = 'mixed').dt.date
    balanceRevenue['Date'] = pd.to_datetime(balanceRevenue['Date'], format = 'mixed').dt.date

    # Define month groups
    month12 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    month13 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    month14 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    month15 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

    # Prepare information required to get data from relevant Excel tab in balance gross profit model
    currentDay = datetime.now(timezone.utc).day
    currentYear = datetime.now(timezone.utc).year
    currentMonth = datetime.now(timezone.utc).month

    # Determine the tab month and year
    if currentDay < 5:
        tabMonth = currentMonth - 2
        # Handle year rollover when looking back 2 months
        if currentMonth == 1:  # January
            tabMonth = 11
            tabYear = currentYear - 1
        elif currentMonth == 2:  # February
            tabMonth = 12
            tabYear = currentYear - 1
        else:
            tabYear = currentYear
    else:
        # After the 5th, look back one month
        tabMonth = currentMonth - 1
        # Handle year rollover when looking back 1 month
        if currentMonth == 1:  # January
            tabMonth = 12
            tabYear = currentYear - 1
        else:
            tabYear = currentYear

    # Format month as two digits
    tabMonth = f"{int(tabMonth):02d}"
    tabYear = str(tabYear)[2:4]
    tabName = tabMonth + '.' + str(tabYear)

    logging.info(f"Determined tab name for Excel: {tabName}")

    # Pull down Excel file for annual projections
    storage_account_url = 'https://tkcpowerbidata.file.core.windows.net'
    storage_account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
    service_client = ShareServiceClient(account_url=storage_account_url, credential=storage_account_key)

    share_name = 'financial'
    directory_path = 'balancegp'
    file_name = 'Balance_GP.xlsx'

    share_client = service_client.get_share_client(share_name)
    directory_client = share_client.get_directory_client(directory_path)

    file_client = directory_client.get_file_client(file_name)
    file_stream = io.BytesIO()
    file_client.download_file().readinto(file_stream)

    file_stream.seek(0)

    balanceGP = pd.read_excel(file_stream, sheet_name=tabName, header=1, engine='openpyxl')

    yearDrop = pd.read_excel(file_stream, sheet_name=tabName, header=0, engine='openpyxl')
    yearCols = [c for c in yearDrop.columns if str(c).upper()[:4] == 'YEAR']
    dropColumns = []
    for i in yearCols:
        dropCol = yearDrop.columns.get_loc(i)
        dropColumns.append(dropCol)

    balanceGP = balanceGP.drop([0])
    balanceGP.drop(balanceGP.columns[dropColumns], axis=1, inplace=True)

    numDays = monthrange(currentYear, currentMonth)[1]
    dateStart = pd.to_datetime(currentYear * 10000 + currentMonth * 100 + numDays, format='%Y%m%d')
    colStart = balanceGP.columns.get_loc(dateStart)

    endActivesRow = balanceGP.loc[balanceGP['Project'].str.upper() == 'ACTIVE PROJECTS G.P.'].index.values.astype(int)[0]
    endMktRow = balanceGP.loc[balanceGP['Project'].str.upper() == 'MARKETING PROJECTS G.P.'].index.values.astype(int)[0]

    currentBalanceRevenue = []
    for i in range(0, 15):
        activeJobs = balanceGP.iloc[:, colStart + i][0: endActivesRow - 2]
        activeJobsMargin = balanceGP['Prj fee'][0: endActivesRow - 2]
        activeJobRevenue = (activeJobs / activeJobsMargin).sum()

        otherActiveJobs = balanceGP.iloc[:, colStart + i][endActivesRow - 1]
        otherJobRevenue = otherActiveJobs / otherMargin

        mktJobs = balanceGP.iloc[:, colStart + i][endActivesRow + 1: endMktRow - 2]
        mktJobsMargin = balanceGP['Prj fee'][endActivesRow + 1: endMktRow - 2]
        mktJobRevenue = (mktJobs / mktJobsMargin).sum()

        totalRevenue = activeJobRevenue + otherJobRevenue + mktJobRevenue
        currentBalanceRevenue.append(totalRevenue)

    # Adjust label calculation to handle January correctly
    if currentMonth == 1:
        label = '12, ' + str(currentYear - 1)
    else:
        label = str(currentMonth - 1) + ', ' + str(currentYear)

    currentBalanceRevenue = pd.DataFrame(currentBalanceRevenue)
    currentBalanceRevenue.index += 1
    currentBalanceRevenue = currentBalanceRevenue.transpose()
    currentBalanceRevenue.insert(0, 'Label', label)
    currentBalanceRevenue.columns = ['Label', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    currentBalanceRevenue['Month'] = pd.to_numeric(currentBalanceRevenue['Label'].str.split(',').str[0])
    currentBalanceRevenue['Year'] = pd.to_numeric(currentBalanceRevenue['Label'].str.split(',').str[1])
    currentBalanceRevenue['Date'] = pd.to_datetime(currentBalanceRevenue.Year * 10000 + currentBalanceRevenue.Month * 100 + 1, format='%Y%m%d')

    logging.info("Adjusted date calculation for currentBalanceRevenue")

    currentBalanceRevenue['monthSum12'] = currentBalanceRevenue[month12].sum(axis=1)
    currentBalanceRevenue['monthSum13'] = currentBalanceRevenue[month13].sum(axis=1)
    currentBalanceRevenue['monthSum14'] = currentBalanceRevenue[month14].sum(axis=1)
    currentBalanceRevenue['monthSum15'] = currentBalanceRevenue[month15].sum(axis=1)
    currentBalanceRevenue['date'] = pd.to_datetime(currentBalanceRevenue['Date'])

    logging.info("Calculated balance revenue from accounting spreadsheet")

    # Get imputed revenue from historical balance gross profit projections
    from azure.storage.fileshare import ShareServiceClient, ShareClient, ShareDirectoryClient, ShareFileClient
    import io

    storage_account_url = 'https://tkcpowerbidata.file.core.windows.net'
    storage_account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
    service_client = ShareServiceClient(account_url=storage_account_url, credential=storage_account_key)

    share_name = 'financial'
    directory_path = 'balancegp'
    file_name = 'balanceRevenue.csv'

    share_client = service_client.get_share_client(share_name)
    directory_client = share_client.get_directory_client(directory_path)

    file_client = directory_client.get_file_client(file_name)
    file_stream = io.BytesIO()
    file_client.download_file().readinto(file_stream)

    # Set the stream position to the beginning
    file_stream.seek(0)

    # Read the Excel file into a DataFrame
    balanceRevenue = pd.read_csv(file_stream)
    balanceRevenue['date'] = pd.to_datetime(balanceRevenue['Date'], format='mixed').dt.date
    balanceRevenue['Date'] = pd.to_datetime(balanceRevenue['Date'], format='mixed').dt.date

    # Overwrite old balance revenue data if it needs a new row added
    from azure.storage.fileshare import ShareFileClient

    # Check if the new row exists in balanceRevenue
    if not balanceRevenue['date'].isin([currentBalanceRevenue['date']]).any():
        
        balanceRevenue = pd.concat([balanceRevenue, currentBalanceRevenue], ignore_index=True)
        balanceRevenue.drop_duplicates(subset='Label', keep='first', inplace=True)
        
        # Create csv and prepare to overwrite old data
        file_path = get_temp_path('balanceRevenue.csv')
        balanceRevenue.to_csv(file_path, index=False)
        
        storage_account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
        storage_account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
        share_name = 'financial' 
        file_path_in_share = 'balancegp/balanceRevenue.csv'
        
        file_client = ShareFileClient(
            account_url=f"https://{storage_account_name}.file.core.windows.net/",
            credential=storage_account_key,
            share_name=share_name,
            file_path=file_path_in_share
        )
        
        # Delete the existing file if it exists (optional but ensures overwrite)
        try:
            file_client.delete_file()
            print(f"Existing file '{file_path_in_share}' deleted from the Azure File Share.")
        except Exception as e:
            print(f"File not found or could not be deleted. Proceeding with upload: {str(e)}")

        # Re-create the file before uploading content
        file_client.create_file(size=0)  # Create the file (size=0 because it will be overwritten)
        
        # Upload the new file content
        with open(file_path, 'rb') as source_file:
            file_client.upload_file(source_file)
        
        print('Updated file uploaded to Azure File Share')
    else:
        print('Data already exists, skipping row addition')

    # Prepare data for analysis
    financials_df['date'] = pd.to_datetime(financials_df['date'])
    balanceRevenue['date'] = pd.to_datetime(balanceRevenue['date'])

    balanceRevenue = financials_df.merge(balanceRevenue[['date', 'monthSum12', 'monthSum15']], on='date', how='right')
    balanceRevenue = balanceRevenue[['next12', 'next15', 'date', 'backlog', 'rollingBacklog', 'constRevenue', 'rollingRevenue', 'constNW', 'rollingNW', '_6MoConst', '_12MoConst', 'revenueBudget', 'month', 'year', 'monthSum12', 'monthSum15']].copy()
    annualRevenueProjections = balanceRevenue.copy()  # Make df for future projections
    balanceRevenue = balanceRevenue[:-12]  # Cut off at least 12 months to make sure future looking revenue actually contains 12 months of data

    balanceRevenue = balanceRevenue.reset_index(drop=True)

    # Scrape CPI for inflation proxy and clean df
    url = 'https://www.usinflationcalculator.com/inflation/consumer-price-index-and-annual-percent-changes-from-1913-to-2008/'
    cpi = pd.read_html(url)
    cpi = cpi[0]
    cpi = cpi[1:]
    cpi.columns = cpi.iloc[0]
    cpi = cpi.drop(columns=['Avg', 'Dec-Dec', 'Avg-Avg'])
    cpi = pd.melt(cpi, id_vars=['Year'], value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    cpi.rename(columns={1: 'Month', 'value': 'CPI'}, inplace=True)
    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'July': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    cpi.Month = cpi.Month.map(months)
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
    years = []
    for i in months:
        if i == 1:
            Year = startYear + 1
        else:
            Year = startYear
        years.append(Year)

    adds = pd.DataFrame(zip(years, months))
    adds['CPI'] = np.nan

    adds.rename(columns={0: 'Year', 1: 'Month'}, inplace=True)

    cpi = pd.concat([cpi, adds], ignore_index=True)

    cpi = cpi.reset_index(drop=True)
    cpi['CPI'] = cpi['CPI'].interpolate(method='spline', order=1)

    cpi['Match'] = cpi['Month'].astype(str) + cpi['Year'].astype(str)

    cpi = cpi.drop_duplicates(subset=['Match'])

    # Scrape GDP for economic proxy and clean df
    from urllib.request import Request, urlopen
    url = 'https://www.multpl.com/us-gdp/table/by-quarter'
    gdp = pd.read_html(url)
    gdp = gdp[0]
    gdp['Year'] = pd.to_numeric(gdp['Date'].str[-4:])
    gdp['Month'] = gdp['Date'].str[:3]
    gdp.rename(columns={'Value': 'GDP'}, inplace=True)
    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    gdp.Month = gdp.Month.map(months)
    gdp = gdp.sort_values(by=['Year', 'Month'])
    gdp['GDP'] = gdp['GDP'].str.replace(' trillion', '')
    gdp['GDP'] = pd.to_numeric(gdp['GDP'])
    gdp['Match'] = gdp['Month'].astype(str) + gdp['Year'].astype(str)
    gdp = gdp.drop_duplicates(subset=['Match'])

    # Scrape prime interest rate data and clean df
    url = 'http://www.fedprimerate.com/wall_street_journal_prime_rate_history.htm'
    prime = pd.read_html(url)
    prime = prime[3]
    prime = prime.drop(columns=[2])
    prime = prime[pd.to_numeric(prime[1], errors='coerce').notnull()]
    prime[1] = pd.to_numeric(prime[1])
    prime[1] = prime[1] / 100
    prime.rename(columns={1: 'Prime Rate'}, inplace=True)
    prime['Date'] = pd.to_datetime(prime[0])
    prime['Year'] = pd.DatetimeIndex(prime['Date']).year
    prime['Month'] = pd.DatetimeIndex(prime['Date']).month
    prime['Match'] = prime['Month'].astype(str) + prime['Year'].astype(str)
    prime = prime.drop_duplicates(subset=['Match'])

    # Attach third party data to internal dataset
    balanceRevenue['month'] = balanceRevenue['month'].astype(int).astype(str)
    balanceRevenue['year'] = balanceRevenue['year'].astype(int).astype(str)
    balanceRevenue['Match'] = balanceRevenue['month'].astype(str) + balanceRevenue['year'].astype(str)

    balanceRevenue = balanceRevenue.merge(cpi[['CPI', 'Match']], on='Match', how='left')
    balanceRevenue = balanceRevenue.merge(gdp[['GDP', 'Match']], on='Match', how='left')
    balanceRevenue = balanceRevenue.merge(prime[['Prime Rate', 'Match']], how='left')
    # balanceRevenue = balanceRevenue.merge(dow[['Dow', 'Match']], how='left')

    balanceRevenue['GDP'] = balanceRevenue['GDP'].interpolate(limit_direction='both')
    balanceRevenue['Prime Rate'] = balanceRevenue['Prime Rate'].interpolate(limit_direction='both')
    # balanceRevenue['Dow'] = balanceRevenue['Dow'].interpolate()

    # Prepare data for forward looking projections
    annualRevenueProjections = annualRevenueProjections.dropna(subset=['month'])  # Drop rows that are not yet populated
    annualRevenueProjections['month'] = annualRevenueProjections['month'].astype(int).astype(str)
    annualRevenueProjections['year'] = annualRevenueProjections['year'].astype(int).astype(str)
    annualRevenueProjections['Match'] = annualRevenueProjections['month'].astype(str) + annualRevenueProjections['year'].astype(str)

    annualRevenueProjections = annualRevenueProjections.merge(cpi[['CPI', 'Match']], on='Match', how='left')
    annualRevenueProjections = annualRevenueProjections.merge(gdp[['GDP', 'Match']], on='Match', how='left')
    annualRevenueProjections['GDP'] = annualRevenueProjections['GDP'].interpolate(limit_direction='both')

    annualRevenueLabels = annualRevenueProjections.iloc[-24:].copy()  # Keep last 2 years of data to span period of projection for context

    annualRevenueProjections = annualRevenueProjections.iloc[-24:]
    annualRevenueProjections = annualRevenueProjections[['monthSum12', 'monthSum15', 'CPI', 'GDP']]

    annualRevenueLabels = annualRevenueLabels[['next12', 'next15', 'backlog', 'date']]
    annualRevenueLabels.reset_index(drop = True, inplace = True)

    fifteenRevenueProjections = annualRevenueProjections[['monthSum15', 'CPI', 'GDP']]
    fifteenRevenueLabels = annualRevenueLabels[['next15', 'backlog', 'date']]

    annualRevenueProjections = annualRevenueProjections[['monthSum12', 'CPI', 'GDP']]
    annualRevenueLabels = annualRevenueLabels[['next12', 'backlog', 'date']]

    # For loop to predict each month based on prior 12 months of data to compare predictions and actuals for performance check
    from datetime import date

    recursiveAnnualProjections = pd.DataFrame()

    for i in range(1, len(balanceRevenue)):
        X = balanceRevenue[['monthSum12', 'CPI', 'GDP']].head(len(balanceRevenue)-i)
        y = balanceRevenue[['next12']].head(len(balanceRevenue) - i)
        proj = balanceRevenue[['monthSum12', 'CPI', 'GDP']].iloc[[-i]]
        model = LinearRegression(fit_intercept=True) 
        model.fit(X, y)
        y_predict = pd.DataFrame(model.predict(proj))
        recursiveAnnualProjections = pd.concat([recursiveAnnualProjections, y_predict], ignore_index=True)

    recursiveAnnualProjections = recursiveAnnualProjections.iloc[::-1]  # Invert df to get into chronological order
    recursiveAnnualProjections.reset_index(drop=True, inplace=True)
    recursiveAnnualProjections.rename(columns={0: 'Annual Projections'}, inplace=True)

    actualsAnnualRevenue = balanceRevenue.tail(len(balanceRevenue)-1).copy()
    actualsAnnualRevenue.reset_index(drop=True, inplace=True)
    actualsAnnualRevenue = pd.concat([actualsAnnualRevenue, recursiveAnnualProjections], axis=1)
    actualsAnnualRevenue.to_csv(get_temp_path('annualRevenuePerformance.csv'))

    # For loop to predict each month based on prior 15 months of data to compare predictions and actuals for performance check
    balanceRevenue15 = balanceRevenue.dropna(subset=['next15']).copy()

    recursiveAnnualProjections = pd.DataFrame()

    for i in range(1, len(balanceRevenue15)):
        X = balanceRevenue15[['monthSum15', 'CPI', 'GDP']].head(len(balanceRevenue15)-i)
        y = balanceRevenue15[['next15']].head(len(balanceRevenue15) - i)
        proj = balanceRevenue15[['monthSum15', 'CPI', 'GDP']].iloc[[-i]]
        model = LinearRegression(fit_intercept=True) 
        model.fit(X, y)
        y_predict = pd.DataFrame(model.predict(proj))
        recursiveAnnualProjections = pd.concat([recursiveAnnualProjections, y_predict], ignore_index=True)

    recursiveAnnualProjections = recursiveAnnualProjections.iloc[::-1]  # Invert df to get into chronological order
    recursiveAnnualProjections.reset_index(drop=True, inplace=True)
    recursiveAnnualProjections.rename(columns={0: 'Annual Projections'}, inplace=True)

    actualsAnnualRevenue = balanceRevenue15.tail(len(balanceRevenue15)-1).copy()
    actualsAnnualRevenue.reset_index(drop=True, inplace=True)
    actualsAnnualRevenue = pd.concat([actualsAnnualRevenue, recursiveAnnualProjections], axis=1)
    actualsAnnualRevenue.to_csv(get_temp_path('fifteenMonthRevenuePerformance.csv'))

    try:
        # Make forward looking 12 month model
        logging.info("Starting 12-month model training...")
        X = balanceRevenue[['monthSum12', 'CPI', 'GDP']]
        y = balanceRevenue[['next12']]
        
        model = LinearRegression(fit_intercept=True) 
        model.fit(X, y)
        annualRevenueProjection = model.predict(annualRevenueProjections)
        annualRevenueProjection = pd.DataFrame(annualRevenueProjection)

        annualRevenueProjection = pd.concat([annualRevenueProjection, annualRevenueLabels], axis = 1)
        for i in range(12, 24): # Remove data for which 12 future looking months 
            annualRevenueProjection.at[i, 'next12'] = np.nan
        annualRevenueProjection.rename(columns = {0: 'Future Annual Revenue Projection'}, inplace = True)
        
        # Use tempfile for local storage
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        annual_file_path = os.path.join(temp_dir, 'annualRevenueProjections.csv')
        fifteen_file_path = os.path.join(temp_dir, 'fifteenRevenueProjections.csv')
        
        annualRevenueProjection.to_csv(annual_file_path)
        logging.info(f"Successfully saved annual revenue projections to {annual_file_path}")

        # Make forward looking 15 month model
        logging.info("Starting 15-month model training...")
        fifteenBalance = balanceRevenue.dropna(subset=['next15']).copy()
        X = fifteenBalance[['monthSum15', 'CPI', 'GDP']]
        y = fifteenBalance[['next15']]
        
        model = LinearRegression(fit_intercept=True) 
        model.fit(X, y)
        fifteenRevenueProjection = model.predict(fifteenRevenueProjections)
        fifteenRevenueProjection = pd.DataFrame(fifteenRevenueProjection)

        fifteenRevenueProjection = pd.concat([fifteenRevenueProjection, fifteenRevenueLabels], axis = 1)
        for i in range(12, 24): # Remove data for which 15 future looking months
            fifteenRevenueProjection.at[i, 'next15'] = np.nan
        fifteenRevenueProjection.rename(columns = {0: 'Future 15 Mo. Revenue Projection'}, inplace = True)
        fifteenRevenueProjection.to_csv(fifteen_file_path)
        logging.info(f"Successfully saved fifteen month revenue projections to {fifteen_file_path}")

    except Exception as e:
        logging.error(f"Error processing revenue projections: {str(e)}")
        raise

# Azure Storage settings
storage_account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
storage_account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
container_name = 'tkc-power-bi-data'

def upload_to_blob(local_file_path: str, blob_name: str):
    """Upload a file to Azure Blob Storage"""
    try:
        # Create the BlobServiceClient object using existing connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get a reference to the container
        container_client = blob_service_client.get_container_client(container_name)
        
        # Get a reference to the blob
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload the file
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logging.info(f"Successfully uploaded {blob_name} to blob storage")
    except Exception as e:
        logging.error(f"Failed to upload {blob_name} to Azure Blob Storage: {str(e)}")
        raise

def main(mytimer: func.TimerRequest) -> None:
    try:
        utc_timestamp = datetime.now(timezone.utc).isoformat()

        if mytimer.past_due:
            logging.info('The timer is past due!')

        logging.info("Starting the revenue projections process...")
        try:
            financial_data = fetch_financial_data()
            logging.info("Successfully fetched financial data")
        except Exception as e:
            logging.error(f"Error fetching financial data: {str(e)}")
            raise

        try:
            processed_data = process_financial_data(financial_data)
            logging.info("Successfully processed financial data")
        except Exception as e:
            logging.error(f"Error processing financial data: {str(e)}")
            raise

        try:
            # Upload all generated files to blob storage
            upload_to_blob(get_temp_path('revenuePerformance.csv'), 'revenuePerformance.csv')
            upload_to_blob(get_temp_path('revenueProjections.csv'), 'revenueProjections.csv')
            upload_to_blob(get_temp_path('revenueBondedProjections.csv'), 'revenueBondedProjections.csv')
            upload_to_blob(get_temp_path('backlogBondedProjections.csv'), 'backlogBondedProjections.csv')
            upload_to_blob(get_temp_path('headCountProjections.csv'), 'headCountProjections.csv')
            upload_to_blob(get_temp_path('revenueSilosProjections.csv'), 'revenueSilosProjections.csv')
            upload_to_blob(get_temp_path('backlogSiloProjections.csv'), 'backlogSiloProjections.csv')
            upload_to_blob(get_temp_path('backlogProjection.csv'), 'backlogProjection.csv')
            upload_to_blob(get_temp_path('annualRevenuePerformance.csv'), 'annualRevenuePerformance.csv')
            upload_to_blob(get_temp_path('annualRevenueProjections.csv'), 'annualRevenueProjections.csv')
            upload_to_blob(get_temp_path('fifteenRevenueProjections.csv'), 'fifteenRevenueProjections.csv')
            logging.info("Successfully uploaded all files to blob storage")
        except Exception as e:
            logging.error(f"Error uploading files: {str(e)}")
            raise
        
        logging.info('Python timer trigger function completed at %s', utc_timestamp)
    except Exception as e:
        logging.error(f"Function failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
