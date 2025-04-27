# data.py
# Contains functions for fetching data from external APIs

import requests
import json
import pandas as pd
from datetime import datetime, timedelta, date # Added date
import time
import math # For pagination calculation

# --- Optional Libraries Check ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("--- 'vaderSentiment' library is available (used in app.py for sentiment). ---")
except ImportError:
    print("="*50); print("!!! WARNING: 'vaderSentiment' library not found. News Sentiment features will be disabled in app.py. !!!"); print("!!! Run: pip install vaderSentiment !!!"); print("="*50)

# --- Constants for APIs ---
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
GEMINI_BASE_URL = "https://api.gemini.com/v2"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
NEWSAPI_ORG_BASE_URL = "https://newsapi.org/v2" # For historical news

# --- API Keys ---
# !! SECURITY WARNING: Storing API keys directly in code is insecure. !!
CRYPTOPANIC_API_KEY = "2c2282ebd04b825308aaf96ac92cd98b8e537058" # For LATEST news
FMP_API_KEY = "yy9dLWwNZLou1neNG4eVX8jNbihIXZPu"
NEWSAPI_ORG_API_KEY = "YOUR_NEWSAPI.ORG_API_KEY" # <<< REPLACE THIS! Get from newsapi.org

# --- Data Fetching Functions ---

def fetch_crypto_news(api_key=CRYPTOPANIC_API_KEY, currency="BTC", page=1):
    """Fetches LATEST cryptocurrency news from CryptoPanic."""
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={currency}&public=true&page={page}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.RequestException as e: print(f"!!! Error fetching LATEST news: {e} !!!"); return []
    except json.JSONDecodeError as e: print(f"!!! Error decoding LATEST news JSON response: {e} !!!"); return []


def fetch_crypto_price(coin_id="bitcoin", vs_currency="usd"):
    """Fetches the current price of a cryptocurrency from CoinGecko."""
    url = f"{COINGECKO_BASE_URL}/simple/price?ids={coin_id}&vs_currencies={vs_currency}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get(coin_id, {}).get(vs_currency)
    except requests.exceptions.RequestException as e: print(f"!!! Error fetching price: {e} !!!"); return None
    except json.JSONDecodeError as e: print(f"!!! Error decoding price JSON response: {e} !!!"); return None

def fetch_gemini_historical_data(symbol, timeframe='1hr'):
    """Fetches and processes historical candle data from Gemini V2 API."""
    endpoint = f"/candles/{symbol}/{timeframe}"
    url = f"{GEMINI_BASE_URL}{endpoint}"
    print(f"--- Attempting to fetch Gemini data from: {url} ---")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        try: raw_data = response.json()
        except json.JSONDecodeError: raw_data = json.loads(response.text)
        print(f"--- Successfully fetched {len(raw_data)} raw data points ({timeframe}) from Gemini. ---")
        if not raw_data or not isinstance(raw_data, list): return []
        try: raw_data.sort(key=lambda x: x[0])
        except Exception as sort_e: print(f"!!! Error sorting raw Gemini data: {sort_e}. !!!"); return None
        processed_data = []
        expected_elements = 6
        for i, entry in enumerate(raw_data):
            if isinstance(entry, list) and len(entry) >= expected_elements:
                try:
                    timestamp_ms = int(entry[0])
                    close_price = float(entry[4])
                    volume = float(entry[5])
                    processed_data.append([timestamp_ms, close_price, volume])
                except (ValueError, TypeError, IndexError) as convert_error: print(f"--- Warning: Skipping Gemini entry #{i} due to data error: {str(entry)[:100]}... | Error: {convert_error} ---")
            else: print(f"--- Warning: Skipping Gemini entry #{i} due to unexpected format or length: {str(entry)[:100]} ---")
        if not processed_data: print(f"--- Warning: No valid data points processed from Gemini. ---"); return []
        print(f"--- Processed {len(processed_data)} valid Gemini data points (timestamp_ms, close_price, volume). ---")
        return processed_data
    except requests.exceptions.RequestException as e: print(f"!!! Error fetching Gemini data for {symbol} (URL: {url}): {e} !!!"); return None
    except json.JSONDecodeError as e: print(f"!!! Error decoding JSON response for {symbol} (URL: {url}): {e} !!!"); return None
    except Exception as e: print(f"!!! An unexpected error occurred fetching/loading Gemini data (URL: {url}): {e} !!!"); return None


def fetch_fmp_historical_data(symbols, start_date, end_date, api_key=FMP_API_KEY):
    """Fetches historical daily closing prices for given symbols from FMP API."""
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime): print("!!! Error: FMP fetch requires datetime objects for start/end dates. !!!"); return None
    if not symbols: print("--- Warning: No symbols provided for FMP fetch. ---"); return None
    start_str = start_date.strftime('%Y-%m-%d'); end_str = end_date.strftime('%Y-%m-%d')
    all_symbols_data = {}; print(f"--- Attempting to fetch FMP historical data for {symbols} from {start_str} to {end_str} ---")
    for symbol in symbols:
        col_name = symbol.replace('^', '').replace('=F', '').replace('-Y.NYB', '').replace('.','')
        print(f"--- Fetching FMP data for: {symbol} (Column: {col_name}) ---")
        url = f"{FMP_BASE_URL}/historical-price-full/{symbol}?from={start_str}&to={end_str}&apikey={api_key}"
        try:
            response = requests.get(url, timeout=15); response.raise_for_status(); data = response.json()
            if not data or 'historical' not in data or not isinstance(data['historical'], list): print(f"--- Warning: No valid 'historical' data found for FMP symbol {symbol}. ---"); all_symbols_data[col_name] = []; time.sleep(0.5); continue
            historical_data = data['historical']; symbol_data = []
            for entry in historical_data:
                try: date_val = pd.to_datetime(entry['date']); close_price = float(entry['close']); symbol_data.append({'ds_date': date_val, col_name: close_price}) # Renamed date to date_val
                except (KeyError, ValueError, TypeError) as proc_e: print(f"--- Warning: Skipping FMP entry for {symbol} due to processing error: {entry} | Error: {proc_e} ---")
            all_symbols_data[col_name] = symbol_data; print(f"--- Processed {len(symbol_data)} data points for FMP symbol {symbol}. ---"); time.sleep(0.5)
        except requests.exceptions.RequestException as e: print(f"!!! Error fetching FMP data for {symbol}: {e} !!!"); all_symbols_data[col_name] = []; time.sleep(1)
        except json.JSONDecodeError as e: print(f"!!! Error decoding FMP JSON for {symbol}: {e} !!!"); all_symbols_data[col_name] = []; time.sleep(1)
        except Exception as e: print(f"!!! An unexpected error occurred fetching FMP data for {symbol}: {e} !!!"); all_symbols_data[col_name] = []; time.sleep(1)
    final_df = None; processed_cols = []
    for col_name, data_list in all_symbols_data.items():
        if not data_list: continue
        try:
            df_symbol = pd.DataFrame(data_list); df_symbol['ds_date'] = pd.to_datetime(df_symbol['ds_date']); df_symbol = df_symbol.drop_duplicates(subset=['ds_date'], keep='last').sort_values(by='ds_date')
            if final_df is None: final_df = df_symbol
            else: final_df = pd.merge(final_df, df_symbol, on='ds_date', how='outer')
            processed_cols.append(col_name)
        except Exception as merge_e: print(f"!!! Error merging FMP data for {col_name}: {merge_e} !!!")
    if final_df is None: print("--- Warning: Failed to fetch or process any valid FMP data. ---"); return None
    if 'ds_date' not in final_df.columns: print("--- Warning: 'ds_date' column missing after FMP merge. ---"); return None
    final_df = final_df[['ds_date'] + processed_cols]; final_df = final_df.sort_values(by='ds_date').reset_index(drop=True)
    print(f"--- Successfully combined FMP data. Shape: {final_df.shape}. Columns: {final_df.columns.tolist()} ---")
    return final_df


# --- Historical News Fetching (Daily) ---
def fetch_historical_news_daily(query, start_date, end_date, api_key=NEWSAPI_ORG_API_KEY):
    """
    Fetches historical news articles matching a query for each day in a date range
    using NewsAPI.org. Returns a dict mapping dates (YYYY-MM-DD strings) to lists of titles.
    """
    if not api_key or api_key == "YOUR_NEWSAPI.ORG_API_KEY":
        print("!!! Error: NewsAPI.org API key not provided in data.py. Cannot fetch historical news. !!!")
        return None
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        print("!!! Error: Historical News fetch requires datetime objects for start/end dates. !!!")
        return None

    today = datetime.now()
    if end_date > today: end_date = today; print(f"--- Warning: Historical news end date adjusted to today ({today.strftime('%Y-%m-%d')}). ---")

    # NewsAPI free tier limit: ~1 month. Adjust start date if necessary.
    max_hist_days = 30
    if (end_date - start_date).days > max_hist_days:
         start_date = end_date - timedelta(days=max_hist_days)
         print(f"--- Warning: Historical news date range exceeds NewsAPI free tier limit (~1 month). Adjusted start date to {start_date.strftime('%Y-%m-%d')}. ---")

    print(f"--- Attempting to fetch historical news for '{query}' from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ---")

    all_news_by_date = {}
    current_date_loop = start_date.date() # Iterate using date objects
    end_date_loop = end_date.date()
    request_count = 0
    max_requests = 90 # Stay under the 100 limit

    while current_date_loop <= end_date_loop and request_count < max_requests:
        date_str = current_date_loop.strftime('%Y-%m-%d')
        print(f"--- Fetching news for date: {date_str} (Request {request_count + 1}) ---")

        url = (f"{NEWSAPI_ORG_BASE_URL}/everything?"
               f"q={query}&"
               f"from={date_str}&"
               f"to={date_str}&"
               f"language=en&"
               f"sortBy=popularity&"
               f"pageSize=100&"
               f"apiKey={api_key}")

        try:
            response = requests.get(url, timeout=15); request_count += 1
            response.raise_for_status(); data = response.json()

            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                titles = [article.get('title') for article in articles if article.get('title')]
                all_news_by_date[date_str] = titles # Store even if empty
                print(f"   - Fetched {len(titles)} articles for {date_str}.")
            else:
                print(f"!!! API Error for {date_str}: {data.get('code')} - {data.get('message')} !!!")
                all_news_by_date[date_str] = [] # Store empty list on API error for this day

            time.sleep(0.7) # Pause

        except requests.exceptions.RequestException as e: print(f"!!! HTTP Error fetching news for {date_str}: {e} !!!"); time.sleep(2)
        except json.JSONDecodeError as e: print(f"!!! Error decoding news JSON for {date_str}: {e} !!!"); time.sleep(1)
        except Exception as e: print(f"!!! Unexpected error fetching news for {date_str}: {e} !!!"); time.sleep(1)

        current_date_loop += timedelta(days=1) # Move to the next day

    if request_count >= max_requests: print(f"--- Warning: Reached approximate daily request limit ({max_requests}). Stopped fetching historical news early. ---")
    print(f"--- Finished fetching historical news. Found articles for {len(all_news_by_date)} dates. ---")
    return all_news_by_date


# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Testing Data Fetching Functions ---")
    # ... (keep tests for LATEST news, price, gemini, fmp) ...
    print("\nTesting News Fetch:")
    news = fetch_crypto_news(currency="BTC", page=1); print(f"Fetched {len(news)} LATEST news items." if news else "LATEST News fetch failed.")
    print("\nTesting Price Fetch:")
    price = fetch_crypto_price(coin_id="bitcoin"); print(f"Current BTC Price: ${price}" if price else "Price fetch failed.")
    print("\nTesting Gemini Historical Data:")
    gemini_data = fetch_gemini_historical_data(symbol='BTCUSD', timeframe='1hr'); print(f"Fetched {len(gemini_data)} Gemini points." if gemini_data else "Gemini fetch failed.")
    print("\nTesting FMP Historical Data:")
    if gemini_data:
        try:
            df_gemini_temp = pd.DataFrame(gemini_data, columns=['timestamp_ms', 'y', 'volume']); df_gemini_temp['ds'] = pd.to_datetime(df_gemini_temp['timestamp_ms'], unit='ms')
            start_dt_fmp = df_gemini_temp['ds'].min(); end_dt_fmp = df_gemini_temp['ds'].max()
            fmp_symbols = ['SPY', '^VIX']; fmp_data_df = fetch_fmp_historical_data(fmp_symbols, start_dt_fmp, end_dt_fmp)
            if fmp_data_df is not None: print(f"Fetched FMP data. Shape: {fmp_data_df.shape}.")
            else: print("FMP fetch failed.")
        except Exception as test_e: print(f"Error during FMP test setup: {test_e}")
    else: print("Skipping FMP test.")

    # Test Historical News
    print("\nTesting Historical News Fetch (NewsAPI.org - Max 1 Month Back):")
    if gemini_data:
        try:
            df_gemini_temp = pd.DataFrame(gemini_data, columns=['timestamp_ms', 'y', 'volume']); df_gemini_temp['ds'] = pd.to_datetime(df_gemini_temp['timestamp_ms'], unit='ms')
            end_dt_news = df_gemini_temp['ds'].max()
            start_dt_news = max(df_gemini_temp['ds'].min(), end_dt_news - timedelta(days=28)) # Limit range for test
            if NEWSAPI_ORG_API_KEY != "YOUR_NEWSAPI.ORG_API_KEY":
                hist_news = fetch_historical_news_daily(query="bitcoin OR btc OR crypto", start_date=start_dt_news, end_date=end_dt_news)
                if hist_news is not None: print(f"Fetched historical news for {len(hist_news)} days.")
                else: print("Historical news fetch failed.")
            else: print("Skipping Historical News test (API Key not set).")
        except Exception as test_e: print(f"Error during Historical News test setup: {test_e}")
    else: print("Skipping Historical News test.")

