# app.py

# --- Keep imports and Flask app setup as before ---
from prophet.diagnostics import cross_validation, performance_metrics
from flask import Flask, render_template
import pandas as pd
from prophet import Prophet
# ... other imports ...
try:
    from data import (fetch_crypto_news, fetch_crypto_price,
                      fetch_gemini_historical_data, fetch_current_fear_greed,
                      fetch_historical_fear_greed)
    print("--- Successfully imported functions from data.py ---") # Success Message
except ImportError as e:
    # ... (Keep your fallback dummy functions) ...
    print(f"!!! ERROR importing from data.py: {e} !!!")


app = Flask(__name__)

@app.route('/')
def index():
    # --- Configuration ---
    symbol = 'BTCUSD'
    coin_gecko_id = 'bitcoin'
    timeframe = '1hr'

    # --- 1. FETCH DATA ---
    historical_data = fetch_gemini_historical_data(symbol, timeframe=timeframe)
    current_price = fetch_crypto_price(coin_id=coin_gecko_id)
    current_fng_data = fetch_current_fear_greed()
    current_fng_value = current_fng_data.get('value') if current_fng_data else None

    # --- Initialize variables ---
    df_prophet = None # Holds price/volume initially
    df_merged = None  # Will hold the final data for fitting
    fng_available = False # Flag to track if F&G data was successfully merged
    predicted_price_1h = None
    predicted_price_12h = None
    predicted_price_24h = None
    percentage_change_1h = None
    percentage_change_12h = None
    percentage_change_24h = None
    error_message = None

    # --- 2. PROCESS PRICE/VOLUME DATA ---
    min_data_points = 25
    if historical_data and len(historical_data) >= min_data_points:
        try:
            print("Processing Price/Volume data...")
            df = pd.DataFrame(historical_data, columns=['timestamp_ms', 'y', 'volume'])
            df['ds'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            df_prophet = df[['ds', 'y', 'volume']] # Base dataframe
        except Exception as e:
            error_message = "Error processing historical price/volume data."
            print(f"!!! {error_message}: {e} !!!")
            df_prophet = None
    # ... (keep elif/else blocks for handling insufficient/failed price data fetch) ...
    elif historical_data is None: error_message = f"Failed to fetch historical data from API for {symbol}/{timeframe}."
    else: error_message = f"Insufficient historical data ({len(historical_data)} points)."


    # --- 2b. FETCH & ATTEMPT MERGE FEAR/GREED DATA ---
    if df_prophet is not None: # Only try if price/volume data is OK
        start_date = df_prophet['ds'].min()
        end_date = df_prophet['ds'].max()
        df_fng = fetch_historical_fear_greed(start_date, end_date) # Fetch F&G history

        if df_fng is not None:
            try:
                print("Attempting to merge Fear/Greed data...")
                df_prophet['ds_date'] = df_prophet['ds'].dt.normalize() # Date for merging
                # Left merge keeps all price/vol rows, adds F&G where date matches
                temp_merged = pd.merge(df_prophet, df_fng, on='ds_date', how='left')
                # Apply daily F&G value forward to fill hourly gaps/weekends
                temp_merged['fear_greed'] = temp_merged['fear_greed'].ffill()
                # Fill any remaining NaNs at the beginning
                temp_merged['fear_greed'] = temp_merged['fear_greed'].bfill()

                if temp_merged['fear_greed'].isnull().any():
                     print("!!! Warning: Could not fill all missing Fear/Greed values. Filling remaining with 50. !!!")
                     temp_merged['fear_greed'].fillna(50, inplace=True)

                # Select final columns for the model fitting dataframe
                df_merged = temp_merged[['ds', 'y', 'volume', 'fear_greed']]
                fng_available = True # Set flag indicating F&G is included
                print(f"Successfully merged F&G data. Fitting DataFrame has columns: {list(df_merged.columns)}")

            except Exception as e:
                if not error_message: error_message = "Error merging Fear/Greed data."
                print(f"!!! {error_message}: {e} !!!")
                df_merged = df_prophet # Fallback to use only price/volume
                fng_available = False
        else:
             print("--- Proceeding without Fear/Greed data (fetch failed or no data). ---")
             df_merged = df_prophet # Use original price/volume df
             fng_available = False
    else:
         # df_prophet was None initially, so df_merged remains None
         df_merged = None


    # --- 3. MODEL TRAINING & PREDICTION ---
    # Check the DataFrame THAT WILL BE USED FOR FITTING
    if df_merged is not None:
        try:
            print("Instantiating Prophet model...")
            model = Prophet()
            print("Adding 'volume' regressor...")
            model.add_regressor('volume') # Always add volume

            # *** CRITICAL FIX: Only add fear_greed regressor if it's actually available ***
            if fng_available and 'fear_greed' in df_merged.columns:
                 print("Adding 'fear_greed' regressor...")
                 model.add_regressor('fear_greed')
            else:
                 print("Fitting model WITHOUT 'fear_greed' regressor.")

            print(f"Fitting model on DataFrame with columns: {list(df_merged.columns)}")
            model.fit(df_merged) # Fit on df_merged (may or may not have fear_greed)

            print("Making future dataframe...")
            future = model.make_future_dataframe(periods=24, freq='h') # Use 'h'

            # --- Add FUTURE values for ALL active regressors ---
            if not df_merged['volume'].empty: future['volume'] = df_merged['volume'].iloc[-1]
            else: future['volume'] = 0

            # Only add future fear_greed if the regressor was added to the model
            if fng_available and 'fear_greed' in df_merged.columns:
                if current_fng_value is not None:
                    future['fear_greed'] = float(current_fng_value)
                    print(f"Added current F&G value ({current_fng_value}) to future dataframe.")
                elif 'fear_greed' in df_merged.columns and not df_merged['fear_greed'].empty: # Check if column exists and isn't empty
                    future['fear_greed'] = df_merged['fear_greed'].iloc[-1] # Fallback to last known
                    print(f"Warning: Current F&G unavailable. Using last historical value ({future['fear_greed'].iloc[0]}) for future.")
                else:
                    future['fear_greed'] = 50 # Absolute fallback
                    print("Warning: Current & historical F&G unavailable. Using 50 for future.")
            # --- End Adding Future Regressors ---

            print("Predicting...")
            forecast = model.predict(future)

            # Extract predictions (check length)
            if len(forecast) >= 24:
                 predicted_price_1h = forecast['yhat'].iloc[-24]
                 predicted_price_12h = forecast['yhat'].iloc[-13]
                 predicted_price_24h = forecast['yhat'].iloc[-1]
                 # ... print statements ...
            else:
                 if not error_message: error_message = "Prediction generated incomplete forecast."
                 print(f"!!! {error_message} Forecast length: {len(forecast)} !!!")


        except Exception as e:
            if not error_message: error_message = "Prediction model failed."
            # Print the specific exception from Prophet/Pandas
            print(f"!!! {error_message}: {e} !!!")
            # Print traceback for detailed debugging if needed
            import traceback
            traceback.print_exc()
            predicted_price_1h = predicted_price_12h = predicted_price_24h = None


    # --- 4. CALCULATE PERCENTAGE CHANGES ---
    # ... (Keep existing percentage change logic) ...
    print("Calculating percentage changes...")
    if current_price is not None and current_price != 0:
        # ... calculations ...
        if predicted_price_1h is not None:
            try: percentage_change_1h = ((predicted_price_1h - current_price) / current_price) * 100
            except Exception as e: print(f"Error calc % change 1h: {e}")
        # etc for 12h, 24h
        if predicted_price_12h is not None:
            try: percentage_change_12h = ((predicted_price_12h - current_price) / current_price) * 100
            except Exception as e: print(f"Error calc % change 12h: {e}")
        if predicted_price_24h is not None:
            try: percentage_change_24h = ((predicted_price_24h - current_price) / current_price) * 100
            except Exception as e: print(f"Error calc % change 24h: {e}")
    # ... (keep handling for current_price being 0 or None) ...


    # --- 5. RENDER TEMPLATE ---
    print("Rendering template...")
    return render_template('index.html',
                           # ... pass all variables ...
                           symbol=symbol,
                           current_price=current_price,
                           current_fng_value=current_fng_value, # Pass F&G value if you want to display it
                           predicted_price_1h=predicted_price_1h,
                           predicted_price_12h=predicted_price_12h,
                           predicted_price_24h=predicted_price_24h,
                           percentage_change_1h=percentage_change_1h,
                           percentage_change_12h=percentage_change_12h,
                           percentage_change_24h=percentage_change_24h,
                           error_message=error_message
                           )
# --- (Existing code: Fit the main model) ---
    if df_merged is not None:
        try:
            print("Fitting Prophet model...")
            model = Prophet()
            model.add_regressor('volume')
            if fng_available and 'fear_greed' in df_merged.columns:
                 model.add_regressor('fear_greed')
            model.fit(df_merged) # Fit on the full available history

            # --- 3b. ACCURACY TESTING (Prophet Cross-Validation) ---
            accuracy_metrics = None # Initialize
            # Only run CV if you have a reasonable amount of data (e.g., > 3x horizon)
            # Let's say minimum ~72 hours (3 days) for a 24h horizon test
            if len(df_merged) > 72:
                try:
                    # --- Define CV Parameters (ADJUST THESE based on your data!) ---
                    # 'initial': How much data to use for the FIRST training set.
                    # 'period': How often to shift the cutoff point forward.
                    # 'horizon': How far ahead to predict from each cutoff.
                    # *Units should match your data frequency (hours)*

                    # Example: If you have ~1400 hours (~60 days) of data
                    # Train on first 1000 hours (~41 days)
                    initial_train_period = '1000 hours'
                    # Make a new prediction every 168 hours (7 days)
                    period_between_forecasts = '168 hours'
                    # Predict 24 hours ahead each time
                    forecast_horizon = '24 hours'

                    print(f"Running Prophet cross-validation (initial='{initial_train_period}', period='{period_between_forecasts}', horizon='{forecast_horizon}')...")
                    # This part can take some time to run!
                    # parallel="processes" might speed it up if you have multiple CPU cores
                    df_cv = cross_validation(model, initial=initial_train_period, period=period_between_forecasts, horizon=forecast_horizon, parallel="processes", disable_tqdm=True)

                    # Calculate standard performance metrics (MAPE, MAE, RMSE etc.)
                    df_p = performance_metrics(df_cv)
                    print("\nCross-Validation Performance Metrics (Sample):")
                    # Display metrics for different horizons up to 24h
                    print(df_p[['horizon', 'mape', 'mae', 'rmse']].head())

                    # Let's calculate overall average MAPE and MAE to display simply
                    # MAPE: Mean Absolute Percentage Error (lower is better)
                    # MAE: Mean Absolute Error (in price units, lower is better)
                    avg_mape = df_p['mape'].mean() * 100 # Show as percentage
                    avg_mae = df_p['mae'].mean()
                    accuracy_metrics = {'mape': avg_mape, 'mae': avg_mae} # Store metrics
                    print(f"Average MAPE across horizons: {avg_mape:.2f}%")
                    print(f"Average MAE across horizons: {avg_mae:.2f}")

                except Exception as cv_e:
                    print(f"!!! Cross-validation failed: {cv_e} !!!")
                    if not error_message: error_message = "Accuracy cross-validation failed."
                    accuracy_metrics = None # Ensure it's None on failure
            else:
                print(f"--- Skipping cross-validation due to insufficient data ({len(df_merged)} rows) ---")
                if not error_message: error_message = "Not enough historical data for accuracy test."


            # --- NOW continue with predicting the actual future ---
            print("Making future dataframe for final prediction...")
            future = model.make_future_dataframe(periods=24, freq='h')
            # ... (Add future regressors: volume, fear_greed) ...
            # ... (forecast = model.predict(future)) ...
            # ... (Extract predicted_price_1h, _12h, _24h) ...

        except Exception as e:
            # ... (Existing error handling for fitting/prediction) ...
            pass # Ensure accuracy_metrics remains None if fit fails

    # --- (Existing code: Calculate percentage changes) ---
    # ...

    # --- 5. RENDER TEMPLATE (Pass the metrics) ---
    print("Rendering template...")
    return render_template('index.html',
                           # ... (pass other variables: symbol, current_price, predictions, etc.) ...
                           accuracy_metrics=accuracy_metrics, # Pass the calculated metrics dict
                           error_message=error_message
                           )


# --- Main execution block ---
if __name__ == '__main__':
    app.run(debug=True)