import logging
import asyncio
from __init__ import fetch_financial_data, process_financial_data, upload_results

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    financial_data = fetch_financial_data()
    processed_data = process_financial_data(financial_data)
    upload_results(processed_data)
