# app.py
# Main Flask application for Bitcoin prediction with FMP & HISTORICAL Sentiment

import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from prophet import Prophet
import json
import os
from datetime import datetime, date # Added date
import traceback
import re

# --- Import data fetching functions ---
try:
    from data import (fetch_crypto_news, fetch_crypto_price,
                      fetch_gemini_historical_data,
                      fetch_fmp_historical_data,
                      fetch_historical_news_daily) # Added historical news fetch
    print("--- Successfully imported functions from data.py ---")
except ImportError as e:
    print(f"!!! FATAL ERROR importing from data.py: {e} !!!")
    # Fallback dummy functions...
    def fetch_crypto_news(*args, **kwargs): print("Dummy fetch_crypto_news"); return []
    def fetch_crypto_price(*args, **kwargs): print("Dummy fetch_crypto_price"); return 50000.0
    def fetch_gemini_historical_data(*args, **kwargs): print("Dummy fetch_gemini_historical_data"); return None
    def fetch_fmp_historical_data(*args, **kwargs): print("Dummy fetch_fmp_historical_data"); return None
    def fetch_historical_news_daily(*args, **kwargs): print("Dummy fetch_historical_news_daily"); return None
    VADER_AVAILABLE = False

# --- Optional: Import VADER ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
    print("--- VADER Sentiment Analyzer loaded successfully. ---")
except ImportError:
    analyzer = None
    VADER_AVAILABLE = False
    print("--- VADER Sentiment Analyzer is unavailable. News sentiment features disabled. ---")


app = Flask(__name__)

# --- Configuration ---
FMP_SYMBOLS = ['SPY', '^VIX']
MIN_HISTORICAL_POINTS = 50
HISTORICAL_NEWS_QUERY = "bitcoin OR btc OR crypto" # Query for historical news

# --- Configuration for Dynamic Confidence ---
# Base score for an 'average' news item
ITEM_BASE_CONFIDENCE = 65.0
# Adjustments applied PER ITEM
ITEM_SENTIMENT_EXTREME_THRESHOLD = 0.6 # Higher threshold for stronger impact
ITEM_SENTIMENT_NEUTRAL_THRESHOLD = 0.1
ITEM_SENTIMENT_STRONG_ADJ_MAX = 15.0 # Max points added for strong sentiment
ITEM_SENTIMENT_NEUTRAL_PENALTY = -10.0 # Points subtracted for neutral sentiment
ITEM_KEYWORD_BOOST = 25.0 # Points added if item has high-impact keywords
ITEM_SOURCE_BOOST = 10.0   # Points added if item is from a major source
# Overall caps for the FINAL AVERAGE score
MAX_AVG_CONFIDENCE = 95.0
MIN_AVG_CONFIDENCE = 40.0

HIGH_IMPACT_KEYWORDS = [
    r'\bETF\b', r'SEC', r'approved', r'denied', r'regulation', r'ban\b', r'halving',
    r'government', r'lawsuit', r'adoption', r'institutional', r'major\s+exchange',
    r'BlackRock', r'Fidelity', r'Grayscale', r'Coinbase', r'Binance',
    r'Powell\b', r'Yellen', r'Gensler',
]
KEYWORD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in HIGH_IMPACT_KEYWORDS]
MAJOR_NEWS_SOURCES = [
    "Reuters", "Bloomberg", "Associated Press", "CNBC", "Financial Times",
    "Wall Street Journal", "CoinDesk", "Cointelegraph", "The Block", "Decrypt"
]


# --- Helper Function for Sentiment Calculation (used for both historical and current) ---
def calculate_average_sentiment(titles_list):
    """Calculates average VADER compound sentiment score for a list of titles."""
    if not VADER_AVAILABLE or not titles_list:
        return 0.0 # Return neutral if VADER unavailable or no titles

    total_compound_score = 0
    valid_title_count = 0
    for title in titles_list:
        if title:
            try:
                vs = analyzer.polarity_scores(title)
                total_compound_score += vs['compound']
                valid_title_count += 1
            except Exception as e:
                pass # Ignore VADER errors for single titles

    if valid_title_count == 0:
        return 0.0 # Avoid division by zero

    average_sentiment = total_compound_score / valid_title_count
    return max(-1.0, min(1.0, average_sentiment)) # Clip score


# --- MODIFIED Helper Function for Dynamic Confidence ---
def calculate_dynamic_confidence(latest_news_list):
    """
    Calculates a heuristic confidence score for EACH news item based on its
    sentiment, keywords, and source, then returns the AVERAGE score.
    """
    if not latest_news_list:
        print("--- No news available for dynamic confidence calculation. Returning base confidence. ---")
        # Decide on a default return value if no news exists
        return ITEM_BASE_CONFIDENCE # Or maybe lower? e.g., 50.0

    individual_scores = []
    print(f"--- Calculating Individual Item Confidence for {len(latest_news_list)} items ---")

    for i, news_item in enumerate(latest_news_list):
        item_confidence = ITEM_BASE_CONFIDENCE
        item_sentiment = 0.0
        title = news_item.get('title')
        source_info = news_item.get('source')
        source_name = source_info.get('name') if source_info and isinstance(source_info, dict) else None

        log_prefix = f"   - Item {i+1} ('{title[:30]}...'): Base={item_confidence:.1f}"

        # 1. Calculate and Adjust based on Item Sentiment
        if VADER_AVAILABLE and title:
            try:
                vs = analyzer.polarity_scores(title)
                item_sentiment = vs['compound']
                abs_sentiment = abs(item_sentiment)

                if abs_sentiment >= ITEM_SENTIMENT_EXTREME_THRESHOLD:
                    # Add points for strong sentiment, capped
                    sentiment_adj = min(ITEM_SENTIMENT_STRONG_ADJ_MAX, (abs_sentiment - ITEM_SENTIMENT_EXTREME_THRESHOLD) * 20) # Stronger scaling
                    item_confidence += sentiment_adj
                    log_prefix += f", SentAdj=+{sentiment_adj:.1f}"
                elif abs_sentiment <= ITEM_SENTIMENT_NEUTRAL_THRESHOLD:
                    item_confidence += ITEM_SENTIMENT_NEUTRAL_PENALTY # Subtract points
                    log_prefix += f", SentAdj={ITEM_SENTIMENT_NEUTRAL_PENALTY:.1f}"
                # else: No adjustment for moderate sentiment

            except Exception as e:
                log_prefix += f", SentErr" # Note error
                # print(f"--- VADER Error on item {i+1}: {e} ---") # Optional debug

        # 2. Check for High-Impact Keywords in this item
        item_keyword_found = False
        if title:
            for pattern in KEYWORD_PATTERNS:
                if pattern.search(title):
                    item_confidence += ITEM_KEYWORD_BOOST
                    log_prefix += f", KeyBoost=+{ITEM_KEYWORD_BOOST:.1f}"
                    item_keyword_found = True
                    break # Only boost once per item

        # 3. Check if Source is Major for this item
        if source_name and source_name in MAJOR_NEWS_SOURCES:
            item_confidence += ITEM_SOURCE_BOOST
            log_prefix += f", SrcBoost=+{ITEM_SOURCE_BOOST:.1f}"

        # Cap/Floor individual item score (optional, but prevents extremes)
        # item_confidence = max(MIN_AVG_CONFIDENCE/1.5, min(MAX_AVG_CONFIDENCE*1.1, item_confidence)) # Example wider cap

        print(f"{log_prefix} -> ItemScore={item_confidence:.1f}")
        individual_scores.append(item_confidence)

    # Calculate the average score
    if not individual_scores:
        print("--- No valid news items processed for confidence. Returning base confidence. ---")
        final_avg_confidence = ITEM_BASE_CONFIDENCE
    else:
        final_avg_confidence = sum(individual_scores) / len(individual_scores)
        print(f"--- Average Item Confidence: {final_avg_confidence:.1f}% ({len(individual_scores)} items) ---")

    # Apply overall caps to the final average
    final_avg_confidence = max(MIN_AVG_CONFIDENCE, min(final_avg_confidence, MAX_AVG_CONFIDENCE))
    print(f"--- Final Capped Average Confidence: {final_avg_confidence:.1f}% ---")

    return round(final_avg_confidence, 1)


# --- Main Route ---
@app.route('/')
def index():
    start_time = datetime.now()
    print(f"\n--- Request received at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # Input Parameters / Config
    symbol = 'BTCUSD'; coin_gecko_id = 'bitcoin'; timeframe = '1hr'; prophet_forecast_horizon = 24

    # Data Storage
    historical_data = None; current_price = None; df_prophet = None; df_fmp = None
    df_sentiment_hist = None; df_merged = None; latest_news = []
    historical_news_dict = None; current_sentiment_for_future = 0.0 # Renamed for clarity
    dynamic_confidence_score = ITEM_BASE_CONFIDENCE # Initialize with item base

    # Flags and Results
    fmp_available_hist = False; sentiment_hist_available = False
    sentiment_regressor_added = False
    predicted_price_1h = None; predicted_price_12h = None; predicted_price_24h = None
    percentage_change_1h = None; percentage_change_12h = None; percentage_change_24h = None
    error_message = None

    # --- 1. Fetch Core Data (Price, Volume, LATEST News) ---
    print("--- Fetching core data... ---")
    try:
        historical_data = fetch_gemini_historical_data(symbol, timeframe=timeframe)
        current_price = fetch_crypto_price(coin_id=coin_gecko_id)
        latest_news = fetch_crypto_news(currency="BTC", page=1) # Fetch LATEST news

        # Calculate dynamic confidence based on the LATEST news list
        # This now calculates the AVERAGE confidence of items in latest_news
        dynamic_confidence_score = calculate_dynamic_confidence(latest_news)

        # Calculate overall current sentiment (used for FUTURE prediction input)
        current_sentiment_for_future = calculate_average_sentiment([n.get('title') for n in latest_news if n.get('title')])
        print(f"--- Overall Current Avg Sentiment (for future prediction): {current_sentiment_for_future:.4f} ---")

        if current_price is None: print("!!! Warning: Failed to fetch current price. !!!");

    except Exception as fetch_err:
        print(f"!!! Error during initial data fetching: {fetch_err} !!!")
        error_message = "Failed to fetch essential market data."
        traceback.print_exc()
        # Pass confidence score even on error
        return render_template('index.html', error_message=error_message, dynamic_confidence_score=f"{dynamic_confidence_score:.1f}%")

    # --- 2. Process Historical Price/Volume Data ---
    # ... (keep existing processing logic) ...
    print("--- Processing historical data... ---")
    if historical_data and len(historical_data) >= MIN_HISTORICAL_POINTS:
        try:
            df = pd.DataFrame(historical_data, columns=['timestamp_ms', 'y', 'volume']); df['ds'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            df_prophet = df[['ds', 'y', 'volume']].copy(); df_prophet.dropna(subset=['ds', 'y'], inplace=True)
            if df_prophet.empty: print("!!! Error: No valid historical price/timestamp data after processing. !!!"); df_prophet = None
            else: print(f"--- Created base Prophet DataFrame. Shape: {df_prophet.shape}. Range: {df_prophet['ds'].min()} to {df_prophet['ds'].max()} ---")
        except Exception as e: error_message = "Error processing historical price/volume data."; print(f"!!! {error_message}: {e} !!!"); traceback.print_exc(); df_prophet = None
    elif historical_data is None: current_error = f"Failed to fetch historical data for {symbol}/{timeframe}."; print(f"!!! {current_error} !!!"); error_message = current_error if not error_message else error_message
    else: current_error = f"Insufficient historical data ({len(historical_data)} points fetched, need {MIN_HISTORICAL_POINTS})."; print(f"!!! {current_error} !!!"); error_message = current_error if not error_message else error_message
    if df_prophet is None and not error_message: error_message = "Historical data unavailable or processing failed."


    # --- 3. Fetch/Process Additional HISTORICAL Regressors (FMP, News Sentiment) ---
    if df_prophet is not None:
        start_date = df_prophet['ds'].min()
        end_date = df_prophet['ds'].max()

        # a) Fetch Historical FMP Data
        # ... (keep existing FMP fetching logic) ...
        print("--- Fetching historical FMP data... ---")
        df_fmp = fetch_fmp_historical_data(FMP_SYMBOLS, start_date, end_date)
        if df_fmp is not None and not df_fmp.empty: fmp_available_hist = True
        else: print("--- Proceeding without historical FMP data. ---"); fmp_available_hist = False


        # b) Fetch & Process Historical News Sentiment
        # ... (keep existing historical news fetching and processing logic) ...
        print("--- Fetching & Processing Historical News Sentiment... ---")
        if VADER_AVAILABLE:
            historical_news_dict = fetch_historical_news_daily(HISTORICAL_NEWS_QUERY, start_date, end_date)
            if historical_news_dict:
                sentiment_data = []
                for date_str, titles in historical_news_dict.items():
                    daily_sentiment = calculate_average_sentiment(titles) if titles else 0.0
                    sentiment_data.append({'ds_date': pd.to_datetime(date_str), 'historical_sentiment': daily_sentiment})
                if sentiment_data:
                    df_sentiment_hist = pd.DataFrame(sentiment_data); df_sentiment_hist = df_sentiment_hist.sort_values(by='ds_date')
                    sentiment_hist_available = True; print(f"--- Processed historical sentiment for {len(df_sentiment_hist)} days. ---")
                else: print("--- No historical sentiment data could be processed. ---")
            else: print("--- Historical news fetch failed or returned no data. ---")
        else: print("--- Skipping historical sentiment processing (VADER unavailable). ---")


        # --- 4. Merge DataFrames for Prophet ---
        # ... (keep existing merging logic) ...
        print("--- Merging data for Prophet model... ---")
        try:
            df_merged = df_prophet.copy()
            if df_merged['ds'].dt.tz is not None: df_merged['ds'] = df_merged['ds'].dt.tz_localize(None) # Ensure naive for merge
            df_merged['ds_date'] = df_merged['ds'].dt.normalize()
            if fmp_available_hist:
                print("--- Merging FMP data... ---"); df_fmp['ds_date'] = pd.to_datetime(df_fmp['ds_date']); df_merged = pd.merge(df_merged, df_fmp, on='ds_date', how='left')
                fmp_cols = [col for col in df_fmp.columns if col != 'ds_date']; print(f"--- Filling NaNs for FMP columns: {fmp_cols} ---")
                for col in fmp_cols:
                    if col in df_merged.columns: df_merged[col] = df_merged[col].ffill().bfill().fillna(0)
            if sentiment_hist_available:
                print("--- Merging Historical Sentiment data... ---"); df_sentiment_hist['ds_date'] = pd.to_datetime(df_sentiment_hist['ds_date']); df_merged = pd.merge(df_merged, df_sentiment_hist, on='ds_date', how='left')
                df_merged['historical_sentiment'] = df_merged['historical_sentiment'].ffill().fillna(0) # Fill NaNs
            elif VADER_AVAILABLE: df_merged['historical_sentiment'] = 0.0 # Add neutral if VADER exists but hist fetch failed
            if 'ds_date' in df_merged.columns: df_merged = df_merged.drop(columns=['ds_date'])
            essential_cols = ['ds', 'y', 'volume']; initial_rows = len(df_merged); df_merged.dropna(subset=essential_cols, inplace=True)
            if len(df_merged) < initial_rows: print(f"--- Warning: Dropped {initial_rows - len(df_merged)} rows due to NaN in essential columns post-merge. ---")
            if len(df_merged) < MIN_HISTORICAL_POINTS: error_message = f"Insufficient data ({len(df_merged)} points) remaining after merging."; df_merged = None
            else: print(f"--- Final DataFrame for Prophet ready. Shape: {df_merged.shape}. Columns: {df_merged.columns.tolist()} ---")
        except Exception as merge_err: current_error = "Error merging dataframes for Prophet."; print(f"!!! {current_error}: {merge_err} !!!"); error_message = current_error if not error_message else error_message; traceback.print_exc(); df_merged = None


    # --- 5. Train Prophet Model and Predict ---
    if df_merged is not None:
        try:
            print("--- Initializing Prophet model... ---")
            model = Prophet(interval_width=0.95)
            print("--- Adding regressors... ---")
            regressors_added = []
            # Add regressors (Volume, FMP, Historical Sentiment)
            if 'volume' in df_merged.columns and not df_merged['volume'].isnull().all(): model.add_regressor('volume'); regressors_added.append('volume')
            if fmp_available_hist:
                fmp_cols = [col for col in df_fmp.columns if col != 'ds_date']
                for col in fmp_cols:
                    if col in df_merged.columns and not df_merged[col].isnull().all(): model.add_regressor(col); regressors_added.append(col)
            if VADER_AVAILABLE and 'historical_sentiment' in df_merged.columns:
                 if not df_merged['historical_sentiment'].isnull().all():
                      model.add_regressor('historical_sentiment'); regressors_added.append('historical_sentiment'); sentiment_regressor_added = True
                 else: print("--- Skipping 'historical_sentiment' regressor (all NaN or missing). ---")
            print(f"--- Regressors added to model: {regressors_added} ---")
            print(f"--- Fitting Prophet model on data shape: {df_merged.shape} ---")
            # Ensure numeric regressors
            for reg in regressors_added:
                 if reg not in ['ds', 'y']: df_merged[reg] = pd.to_numeric(df_merged[reg], errors='coerce').fillna(0)
            model.fit(df_merged)
            print("--- Model fitting complete. ---")

            # --- 6. Make Future DataFrame and Predict ---
            print("\n--- Making future dataframe for prediction... ---")
            future = model.make_future_dataframe(periods=prophet_forecast_horizon, freq='h')
            print("--- Populating future regressor values... ---")
            # Populate future regressors (Volume, FMP, Historical Sentiment -> using CURRENT sentiment)
            if 'volume' in regressors_added: future['volume'] = df_merged['volume'].iloc[-1] if not df_merged['volume'].empty else 0
            if fmp_available_hist:
                fmp_cols = [col for col in df_fmp.columns if col != 'ds_date']
                for col in fmp_cols:
                    if col in regressors_added: future[col] = df_merged[col].iloc[-1] if col in df_merged.columns and not df_merged[col].empty else 0
            if sentiment_regressor_added:
                 future['historical_sentiment'] = current_sentiment_for_future # Use overall avg sentiment of LATEST news
                 print(f"   - Future 'historical_sentiment' regressor set to OVERALL CURRENT sentiment: {current_sentiment_for_future:.4f}")

            print("--- Predicting future prices... ---")
            forecast = model.predict(future)
            print("--- Prediction complete. ---")
            # Extract predictions
            if len(forecast) >= prophet_forecast_horizon:
                idx_1h = len(df_merged); predicted_price_1h = forecast.loc[idx_1h, 'yhat']
                idx_12h = len(df_merged) + 11; predicted_price_12h = forecast.loc[idx_12h, 'yhat']
                idx_24h = len(forecast) - 1; predicted_price_24h = forecast.loc[idx_24h, 'yhat']
                print(f"--- Predicted Prices Extracted (1h, 12h, 24h) ---")
            else: current_error = "Prediction generated incomplete forecast."; print(f"!!! {current_error} !!!"); error_message = current_error if not error_message else error_message
        except Exception as model_err:
            current_error = "Prediction model failed during training or forecasting."; print(f"!!! {current_error}: {model_err} !!!"); error_message = current_error if not error_message else error_message; traceback.print_exc()
            predicted_price_1h = predicted_price_12h = predicted_price_24h = None
    else:
        print("--- Skipping model training and prediction due to unavailable/insufficient merged data. ---")
        if not error_message: error_message = "Cannot generate prediction due to data issues."

    # --- 7. Calculate Percentage Changes --- ## <<<< CORRECTED SECTION >>>> ##
    print("--- Calculating percentage changes... ---")
    if current_price is not None and current_price > 0:
        # Calculate for 1h
        if predicted_price_1h is not None:
            try:
                percentage_change_1h = ((predicted_price_1h - current_price) / current_price) * 100
            except Exception as e:
                print(f"Error calc % change 1h: {e}")
                percentage_change_1h = None # Ensure it's None on error

        # Calculate for 12h
        if predicted_price_12h is not None:
            try:
                percentage_change_12h = ((predicted_price_12h - current_price) / current_price) * 100
            except Exception as e:
                print(f"Error calc % change 12h: {e}")
                percentage_change_12h = None # Ensure it's None on error

        # Calculate for 24h
        if predicted_price_24h is not None:
            try:
                percentage_change_24h = ((predicted_price_24h - current_price) / current_price) * 100
            except Exception as e:
                print(f"Error calc % change 24h: {e}")
                percentage_change_24h = None # Ensure it's None on error

        print("--- Percentage changes calculated. ---")
    elif current_price is None:
         print("--- Skipping percentage change calculation (current price unavailable). ---")
    else: # current_price is 0 or negative
         print(f"--- Skipping percentage change calculation (invalid current price: {current_price}). ---")


    # --- 8. Render Template ---
    # ... (keep existing rendering logic) ...
    end_time = datetime.now(); processing_time = (end_time - start_time).total_seconds()
    print(f"--- Request processing finished in {processing_time:.2f} seconds. Rendering template... ---")
    def format_price(p): return f"${p:,.2f}" if p is not None else "N/A"
    def format_perc(p): return f"{p:.2f}%" if p is not None else "N/A"
    return render_template('index.html',
                           symbol=symbol,
                           current_price_display=format_price(current_price),
                           predicted_price_1h_display=format_price(predicted_price_1h),
                           predicted_price_12h_display=format_price(predicted_price_12h),
                           predicted_price_24h_display=format_price(predicted_price_24h),
                           percentage_change_1h_display=format_perc(percentage_change_1h),
                           percentage_change_12h_display=format_perc(percentage_change_12h),
                           percentage_change_24h_display=format_perc(percentage_change_24h),
                           dynamic_confidence_score=f"{dynamic_confidence_score:.1f}%", # Pass the average score
                           error_message=error_message,
                           processing_time=f"{processing_time:.2f} sec",
                           last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
                           )

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
