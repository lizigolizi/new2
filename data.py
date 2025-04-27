# data.py
# Contains functions for fetching data from external APIs
# --- THIS IS THE COMPLETE CODE ---

import requests
import json
import pandas as pd # Needed for F&G processing
from datetime import datetime, timedelta

# Attempt to import the F&G library
try:
    from fear_and_greed import FearAndGreedIndex
    FNG_LIB_AVAILABLE = True
    print("--- 'fear-and-greed-crypto' library loaded successfully. ---")
except ImportError:
    print("="*50)
    print("!!! WARNING: 'fear-and-greed-crypto' library not found. F&G features will be disabled. !!!")
    print("!!! Run: pip install fear-and-greed-crypto !!!")
    print("="*50)
    FearAndGreedIndex = None # Define as None to handle gracefully
    FNG_LIB_AVAILABLE = False


# --- Constants for APIs ---
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
# !! SECURITY WARNING: Keep API keys secret, ideally use environment variables !!
CRYPTOPANIC_API_KEY = "2c2282ebd04b825308aaf96ac92cd98b8e537058" # From your previous code

# --- Data Fetching Functions ---

def fetch_crypto_news(api_key=CRYPTOPANIC_API_KEY, currency="BTC", page=1):
    """Fetches cryptocurrency news for a specific currency from CryptoPanic."""
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={currency}&public=true&page={page}"
    try:
        # print(f"Fetching news from: {url}") # Optional debug print
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.RequestException as e:
        print(f"!!! Error fetching news: {e} !!!")
        return [] # Return empty list on error

def fetch_crypto_price(coin_id="bitcoin", vs_currency="usd"):
    """Fetches the current price of a cryptocurrency from CoinGecko."""
    url = f"{COINGECKO_BASE_URL}/simple/price?ids={coin_id}&vs_currencies={vs_currency}"
    try:
        # print(f"Fetching price from: {url}") # Optional debug print
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Returns the price (float) or None if not found
        return data.get(coin_id, {}).get(vs_currency)
    except requests.exceptions.RequestException as e:
        print(f"!!! Error fetching price: {e} !!!")
        return None # Return None on error

def fetch_gemini_historical_data(symbol, timeframe='1hr'): # Defaulting to hourly
    """Fetches and processes historical candle data (including volume) from Gemini V2 API."""
    base_url = "https://api.gemini.com/v2"
    endpoint = f"/candles/{symbol}/{timeframe}"
    url = f"{base_url}{endpoint}"
    print(f"--- Attempting to fetch Gemini data from: {url} ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        raw_data = json.loads(response.text)
        print(f"--- Successfully fetched {len(raw_data)} raw data points ({timeframe}) from Gemini. ---")

        if not raw_data:
            print("--- Warning: Received empty data from Gemini API. ---")
            return []
        if len(raw_data) < 24:
             print(f"--- Warning: Insufficient hourly data points ({len(raw_data)}). Need > ~24 for prediction. ---")
             if len(raw_data) <= 10: return []

        try:
            raw_data.sort(key=lambda x: x[0]) # Sort ascending by timestamp
        except Exception as sort_e:
            print(f"!!! Error sorting raw data: {sort_e}. Data might be malformed. !!!")
            return None

        processed_data = []
        expected_elements = 6 # ts, o, h, l, c, vol
        for i, entry in enumerate(raw_data):
            if isinstance(entry, list) and len(entry) >= expected_elements:
                try:
                    timestamp = int(entry[0])
                    close_price = float(entry[4])
                    volume = float(entry[5])
                    processed_data.append([timestamp, close_price, volume])
                except (ValueError, TypeError, IndexError) as convert_error:
                    print(f"--- Warning: Skipping entry #{i} due to data error: {entry[:expected_elements+1]}... | Error: {convert_error} ---")
            else:
                print(f"--- Warning: Skipping entry #{i} due to unexpected format: {entry} ---")

        if not processed_data:
             print(f"--- Warning: No valid data points could be processed from {len(raw_data)} raw entries. ---")
             return []

        print(f"--- Processed {len(processed_data)} valid data points (timestamp, close_price, volume). ---")
        return processed_data

    # ... (keep existing except blocks for requests/json errors) ...
    except requests.exceptions.RequestException as e: print(f"!!! Error fetching Gemini data for {symbol} (URL: {url}): {e} !!!"); return None
    except json.JSONDecodeError as e: print(f"!!! Error decoding JSON response for {symbol} (URL: {url}): {e} !!!"); return None
    except Exception as e: print(f"!!! An unexpected error occurred fetching/loading Gemini data (URL: {url}): {e} !!!"); return None

# --- Added Fear & Greed Functions ---

def fetch_current_fear_greed():
    """Fetches the current Fear & Greed Index value."""
    if not FNG_LIB_AVAILABLE: return None # Library not installed check
    try:
        fng = FearAndGreedIndex()
        # Assuming get_current_data returns a dict like {'value': 55, 'classification': 'Neutral', 'timestamp': ...}
        current_data = fng.get_current_data()
        print(f"--- Fetched Current F&G: Value={current_data.get('value')}, Class={current_data.get('classification')} ---")
        return current_data
    except Exception as e:
        print(f"!!! Error fetching current Fear & Greed Index: {e} !!!")
        return None

def fetch_historical_fear_greed(start_date, end_date):
    """Fetches historical F&G index values between two dates and returns a DataFrame."""
    if not FNG_LIB_AVAILABLE: return None # Library not installed check
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        print("!!! Error: Historical F&G fetch requires datetime objects for start/end dates. !!!")
        return None

    try:
        fng = FearAndGreedIndex()
        # NOTE: Adjust based on actual library method for fetching ranges or N days
        print(f"--- Attempting to fetch historical F&G from {start_date.date()} to {end_date.date()} ---")
        # Example assumes get_historical_data exists and takes datetime objects
        # It might return list of dicts: [{'value': V, 'classification': C, 'timestamp': TS_SECONDS}, ...]
        historical_list = fng.get_historical_data(start_date=start_date, end_date=end_date)

        if not historical_list:
            print("--- Warning: Received no historical F&G data for the period. ---")
            return None

        processed_fng = []
        for entry in historical_list:
             try:
                 # Adjust timestamp conversion based on actual library output format!
                 # Assuming entry['timestamp'] is seconds since epoch:
                 ts_seconds = int(entry['timestamp'])
                 ts_datetime = datetime.fromtimestamp(ts_seconds)
                 # We need the date part for merging with hourly data later
                 date_only = pd.to_datetime(ts_datetime.date()) # Use pandas for consistency
                 value = int(entry['value'])
                 processed_fng.append({'ds_date': date_only, 'fear_greed': value})
             except (KeyError, ValueError, TypeError) as proc_e:
                 print(f"--- Warning: Skipping F&G entry due to processing error: {entry} | Error: {proc_e} ---")


        if not processed_fng:
             print("--- Warning: Failed processing historical F&G data. ---")
             return None

        df_fng = pd.DataFrame(processed_fng)
        # Remove potential duplicate dates if API returns multiple per day
        df_fng = df_fng.drop_duplicates(subset=['ds_date'], keep='last').sort_values(by='ds_date')

        print(f"--- Processed {len(df_fng)} unique daily F&G data points. ---")
        return df_fng # Return DataFrame with 'ds_date', 'fear_greed'

    except Exception as e:
        print(f"!!! Error fetching/processing historical Fear & Greed Index: {e} !!!")
        return None