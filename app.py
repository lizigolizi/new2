# app.py

# --- Imports ---
import pandas as pd
import traceback # For detailed error printing
from flask import Flask, render_template
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import datetime # Added for timestamping

# --- Import data fetching functions ---
try:
    # Assuming data.py is in the same directory
    from data import (fetch_crypto_price,
                      fetch_gemini_historical_data, fetch_current_fear_greed,
                      fetch_historical_fear_greed)
    print("--- Successfully imported functions from data.py ---")
except ImportError as e:
    print(f"!!! ERROR importing from data.py: {e} !!!")
    # Define dummy functions if import fails, so the app doesn't crash immediately
    def fetch_crypto_price(*args, **kwargs): return None
    def fetch_gemini_historical_data(*args, **kwargs): return None
    def fetch_current_fear_greed(*args, **kwargs): return None
    def fetch_historical_fear_greed(*args, **kwargs): return None

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Main Route ---
@app.route('/')
def index():
    start_time = datetime.now() # For timing request
    print(f"\n--- Request received at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # --- Configuration ---
    symbol = 'BTCUSD'
    coin_gecko_id = 'bitcoin'
    timeframe = '1hr'
    min_data_points = 25 # Minimum historical points needed

    # --- Initialize variables ---
    historical_data = None
    current_price = None
    current_fng_data = None
    current_fng_value = None
    df_prophet = None # Holds price/volume initially
    df_merged = None  # Will hold the final data for fitting
    df_fng = None     # Holds historical F&G data
    fng_available = False # Flag to track if F&G data was successfully merged
    model = None      # Prophet model instance
    forecast = None   # Prophet forecast DataFrame
    future = None     # Prophet future DataFrame
    predicted_price_1h = None
    predicted_price_12h = None
    predicted_price_24h = None
    percentage_change_1h = None
    percentage_change_12h = None
    percentage_change_24h = None
    accuracy_metrics = None # For CV results
    historical_accuracy_percent = None # Derived from CV MAPE
    data_confidence_score = 0 # Initialize confidence score
    prediction_interval_penalty = 0 # Initialize penalty
    used_current_fng_for_future = False # Flag to track F&G usage for future
    plot_data = None # Initialize plot data dictionary
    error_message = None # Collects user-facing errors

    # --- 1. FETCH DATA ---
    try:
        print("--- Fetching Data ---")
        historical_data = fetch_gemini_historical_data(symbol, timeframe=timeframe)
        current_price = fetch_crypto_price(coin_id=coin_gecko_id)
        current_fng_data = fetch_current_fear_greed()
        current_fng_value = current_fng_data.get('value') if current_fng_data else None
        print(f"Fetched: Current Price={current_price}, Current F&G={current_fng_value}")
    except Exception as fetch_e:
        error_message = "Error during initial data fetching."
        print(f"!!! {error_message}: {fetch_e} !!!")
        # Allow proceeding, but some features might fail later

    # --- 2. PROCESS PRICE/VOLUME DATA ---
    if historical_data and len(historical_data) >= min_data_points:
        try:
            print("--- Processing Price/Volume data ---")
            df = pd.DataFrame(historical_data, columns=['timestamp_ms', 'y', 'volume'])
            df['ds'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            # Ensure essential columns exist and drop rows with NaNs in them BEFORE calculating score
            df.dropna(subset=['ds', 'y'], inplace=True)
            if df.empty:
                 raise ValueError("No valid rows remaining after dropping NaNs in 'ds' or 'y'.")
            df_prophet = df[['ds', 'y', 'volume']].copy() # Use .copy()

            # Calculate base confidence score ONLY if data is valid
            data_confidence_score = 50 # Base score for having minimum required data
            if len(df_prophet) > 100: # Bonus for having more substantial history
                 data_confidence_score += 10
            print(f"Data Confidence: Base score {data_confidence_score} after price/vol check.")
        except Exception as e:
            error_message = "Error processing historical price/volume data."
            print(f"!!! {error_message}: {e} !!!")
            traceback.print_exc()
            df_prophet = None # Ensure it's None on failure
            data_confidence_score = 0 # Reset confidence if core data fails
    else:
        # Handle insufficient or failed price data fetch
        if historical_data is None:
            error_message = f"Failed to fetch historical data from API for {symbol}/{timeframe}."
        else: # len(historical_data) < min_data_points
            error_message = f"Insufficient historical data ({len(historical_data)} points) for prediction."
        print(f"!!! {error_message} Confidence Score: {data_confidence_score} !!!")
        df_prophet = None # Ensure it's None
        # Render early if core data is missing/insufficient - cannot proceed
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"--- Request processing finished early in {processing_time:.2f} seconds due to data issue. Rendering template... ---")
        return render_template('dashboard.html',
                               symbol=symbol,
                               error_message=error_message,
                               data_confidence_score=data_confidence_score, # Pass 0 score
                               plot_data=None, # Ensure plot_data is None
                               processing_time=f"{processing_time:.2f} sec",
                               last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
                               )


    # --- 2b. FETCH & ATTEMPT MERGE FEAR/GREED DATA ---
    # This block only runs if df_prophet was successfully created above
    if df_prophet is not None:
        try:
            start_date = df_prophet['ds'].min()
            end_date = df_prophet['ds'].max()
            df_fng = fetch_historical_fear_greed(start_date, end_date) # Fetch F&G history

            if df_fng is not None and not df_fng.empty: # Check if df_fng is not None AND not empty
                print("--- Attempting to merge Fear/Greed data ---")
                # Use .copy() to avoid modifying df_prophet directly if merge fails
                df_prophet_with_date = df_prophet.copy()
                df_prophet_with_date['ds_date'] = df_prophet_with_date['ds'].dt.normalize() # Date for merging

                # Perform the merge
                temp_merged = pd.merge(df_prophet_with_date, df_fng, on='ds_date', how='left')

                # Apply daily F&G value forward/backward to fill gaps
                temp_merged['fear_greed'] = temp_merged['fear_greed'].ffill().bfill()

                # Final check and fill for any remaining NaNs (e.g., if all history was NaN)
                if temp_merged['fear_greed'].isnull().any():
                     print("!!! Warning: Could not fill all missing Fear/Greed values. Filling remaining with 50. !!!")
                     temp_merged['fear_greed'].fillna(50, inplace=True)

                # Select final columns for the model fitting dataframe
                df_merged = temp_merged[['ds', 'y', 'volume', 'fear_greed']].copy()
                fng_available = True # Set flag indicating F&G is included
                data_confidence_score += 25 # Major boost for using F&G regressor
                print(f"Data Confidence: Added F&G bonus. Score = {data_confidence_score}")
                print(f"Successfully merged F&G data. Fitting DataFrame has columns: {list(df_merged.columns)}")
            else:
                 print("--- Proceeding without Fear/Greed data (fetch failed or no data). ---")
                 df_merged = df_prophet.copy() # Use original price/volume df
                 fng_available = False
                 print(f"Data Confidence: No F&G bonus applied. Score = {data_confidence_score}")

        except Exception as e:
            if not error_message: error_message = "Error merging Fear/Greed data."
            print(f"!!! {error_message}: {e} !!!")
            traceback.print_exc()
            df_merged = df_prophet.copy() # Fallback to use only price/volume
            fng_available = False
            print(f"Data Confidence: F&G merge failed. Score = {data_confidence_score}")
    # If df_prophet was None initially, df_merged remains None here


    # --- 3. MODEL TRAINING & PREDICTION ---
    # This block only runs if df_merged was successfully created
    if df_merged is not None:
        try:
            print("--- Instantiating Prophet model ---")
            model = Prophet(interval_width=0.95) # Consider adding seasonality options if needed
            print("Adding 'volume' regressor...")
            # Ensure volume column exists and is not all NaN before adding
            if 'volume' in df_merged.columns and not df_merged['volume'].isnull().all():
                 df_merged['volume'] = pd.to_numeric(df_merged['volume'], errors='coerce').fillna(0) # Ensure numeric, fill NaNs
                 model.add_regressor('volume')
            else:
                 print("--- Skipping 'volume' regressor (column missing or all NaN) ---")


            # Conditionally add fear_greed regressor
            if fng_available and 'fear_greed' in df_merged.columns and not df_merged['fear_greed'].isnull().all():
                 print("Adding 'fear_greed' regressor...")
                 df_merged['fear_greed'] = pd.to_numeric(df_merged['fear_greed'], errors='coerce').fillna(50) # Ensure numeric, fill NaNs
                 model.add_regressor('fear_greed')
            else:
                 print("Fitting model WITHOUT 'fear_greed' regressor (not available or all NaN).")
                 fng_available = False # Ensure flag is false if regressor not added

            # Drop rows with NaN in 'y' before fitting
            initial_rows = len(df_merged)
            df_merged.dropna(subset=['y'], inplace=True)
            if len(df_merged) < initial_rows:
                print(f"--- Warning: Dropped {initial_rows - len(df_merged)} rows due to NaN in 'y' before fitting. ---")

            if len(df_merged) < min_data_points: # Check again after potential drops
                 raise ValueError(f"Insufficient data ({len(df_merged)} rows) remaining for model fitting after processing regressors.")

            print(f"Fitting model on DataFrame with columns: {list(df_merged.columns)} ({len(df_merged)} rows)")
            model.fit(df_merged) # Fit on df_merged

            print("Making future dataframe...")
            future = model.make_future_dataframe(periods=24, freq='h') # Hourly frequency

            # --- Add FUTURE values for ALL active regressors ---
            # Volume: Use last known value or 0 as fallback
            if 'volume' in model.regressors: # Check if regressor was actually added
                if not df_merged['volume'].empty:
                     future['volume'] = df_merged['volume'].iloc[-1]
                else:
                     future['volume'] = 0
                     print("Warning: Volume data was empty, using 0 for future volume.")

            # Fear & Greed: Only add if the regressor was added to the model
            if 'fear_greed' in model.regressors: # Check if regressor was actually added
                if current_fng_value is not None:
                    future['fear_greed'] = float(current_fng_value)
                    used_current_fng_for_future = True # Set flag
                    print(f"Added current F&G value ({current_fng_value}) to future. Confidence flag set.")
                elif not df_merged['fear_greed'].empty: # Check if historical column exists and isn't empty
                    last_hist_fng = df_merged['fear_greed'].iloc[-1]
                    future['fear_greed'] = last_hist_fng # Fallback to last known
                    print(f"Warning: Current F&G unavailable. Using last historical value ({last_hist_fng}) for future.")
                else:
                    future['fear_greed'] = 50 # Absolute fallback
                    print("Warning: Current & historical F&G unavailable. Using 50 for future.")

                # Apply Bonus/Penalty based on current F&G usage for future
                if used_current_fng_for_future:
                    data_confidence_score += 5 # Small bonus for having up-to-date F&G
                    print(f"Data Confidence: Added Current F&G bonus. Score = {data_confidence_score}")
                else:
                    # Optional penalty removed for simplicity, just no bonus
                    print("Data Confidence: No Current F&G bonus (used fallback).")
            # --- End Adding Future Regressors ---

            print("Predicting...")
            forecast = model.predict(future)

            # --- Calculate Penalty based on Prediction Interval Width ---
            if forecast is not None and len(forecast) >= 24:
                pred_24h_row = forecast.iloc[-1] # Get the last row (24h prediction)
                yhat = pred_24h_row.get('yhat')
                yhat_lower = pred_24h_row.get('yhat_lower')
                yhat_upper = pred_24h_row.get('yhat_upper')

                # Check if interval values are valid numbers
                if pd.notna(yhat) and yhat != 0 and pd.notna(yhat_lower) and pd.notna(yhat_upper):
                    interval_width = yhat_upper - yhat_lower
                    relative_width = abs(interval_width / yhat)
                    print(f"Prediction Interval (24h): Width={interval_width:.2f}, Relative Width={relative_width:.3f}")

                    # Define penalty thresholds (adjust these based on typical results)
                    if relative_width > 0.20: prediction_interval_penalty = 20
                    elif relative_width > 0.10: prediction_interval_penalty = 10
                    elif relative_width > 0.05: prediction_interval_penalty = 5
                    else: prediction_interval_penalty = 0

                    data_confidence_score -= prediction_interval_penalty
                    print(f"Data Confidence: Applied Interval Penalty: -{prediction_interval_penalty}. Score = {data_confidence_score}")
                else:
                    print("Data Confidence: Could not calculate interval penalty (missing forecast/interval values).")

            # --- Extract specific point predictions ---
            if forecast is not None:
                # Use .get() for safety in case forecast is shorter than expected
                predicted_price_1h = forecast.iloc[-24].get('yhat') if len(forecast) >= 24 else None
                predicted_price_12h = forecast.iloc[-13].get('yhat') if len(forecast) >= 13 else None
                predicted_price_24h = forecast.iloc[-1].get('yhat') if len(forecast) >= 1 else None # Use last row

                if predicted_price_1h is None or predicted_price_12h is None or predicted_price_24h is None:
                     if not error_message: error_message = "Prediction generated incomplete forecast."
                     print(f"!!! {error_message} Forecast length: {len(forecast)} !!!")
            else:
                 # Forecast is None, prediction failed earlier
                 predicted_price_1h = predicted_price_12h = predicted_price_24h = None


        except Exception as e:
            if not error_message: error_message = "Prediction model fitting or forecasting failed."
            print(f"!!! {error_message}: {e} !!!")
            traceback.print_exc() # Print detailed error
            # Reset predictions if model failed
            predicted_price_1h = predicted_price_12h = predicted_price_24h = None
            forecast = None # Ensure forecast is None if prediction fails
            model = None # Ensure model is None
            data_confidence_score = max(0, data_confidence_score - 30) # Penalize heavily

    # --- Clamp final confidence score ---
    data_confidence_score = max(0, min(data_confidence_score, 100))
    print(f"FINAL Data Confidence Score: {data_confidence_score}")


    # --- 3b. ACCURACY TESTING (Prophet Cross-Validation) ---
    # Run only if model fitting was successful and we have enough data
    if model is not None and df_merged is not None and len(df_merged) > 72:
        try:
            # Define CV Parameters (ADJUST THESE!)
            # These should be strings representing time periods (e.g., 'X days', 'Y hours')
            initial_train_period = f'{max(30, len(df_merged) - 3*24)} hours' # Use most data, leave 3x horizon
            period_between_forecasts = '168 hours' # Example: 7 days
            forecast_horizon = '24 hours' # Predict 24h ahead

            print(f"\n--- Running Prophet cross-validation (initial='{initial_train_period}', period='{period_between_forecasts}', horizon='{forecast_horizon}') ---")
            # Ensure model used for CV is the one just trained
            df_cv = cross_validation(model, initial=initial_train_period, period=period_between_forecasts, horizon=forecast_horizon, parallel="processes", disable_tqdm=True)
            df_p = performance_metrics(df_cv)
            print("\nCross-Validation Performance Metrics (Sample):")
            print(df_p[['horizon', 'mape', 'mae', 'rmse']].head())

            # Calculate overall average MAPE and MAE
            avg_mape = df_p['mape'].mean() # Keep as fraction for calculation
            avg_mae = df_p['mae'].mean()

            # Store display-ready metrics
            if pd.notna(avg_mape) and pd.notna(avg_mae):
                 accuracy_metrics = {'mape': avg_mape * 100, 'mae': avg_mae}
            else:
                 accuracy_metrics = None

            # Calculate Historical Accuracy Percentage
            if pd.notna(avg_mape):
                 clamped_mape = max(0, min(avg_mape, 1.0)) # Clamp MAPE between 0 and 1
                 historical_accuracy_percent = (1 - clamped_mape) * 100
                 print(f"Calculated Historical Accuracy: {historical_accuracy_percent:.2f}% (based on Avg MAPE: {avg_mape:.4f})")
            else:
                 print("--- Could not calculate historical accuracy (MAPE unavailable) ---")
                 historical_accuracy_percent = None

        except Exception as cv_e:
            print(f"!!! Cross-validation failed: {cv_e} !!!")
            traceback.print_exc()
            if not error_message: error_message = "Accuracy cross-validation failed."
            accuracy_metrics = None
            historical_accuracy_percent = None
    elif model is not None: # Model exists but not enough data for CV
        print(f"--- Skipping cross-validation due to insufficient data ({len(df_merged)} rows) ---")
        if not error_message: error_message = "Not enough historical data for accuracy test."
        accuracy_metrics = None
        historical_accuracy_percent = None


    # --- 4. CALCULATE PERCENTAGE CHANGES ---
    print("--- Calculating percentage changes ---")
    if current_price is not None and current_price != 0:
        try:
            if predicted_price_1h is not None and pd.notna(predicted_price_1h):
                percentage_change_1h = ((predicted_price_1h - current_price) / current_price) * 100
            if predicted_price_12h is not None and pd.notna(predicted_price_12h):
                percentage_change_12h = ((predicted_price_12h - current_price) / current_price) * 100
            if predicted_price_24h is not None and pd.notna(predicted_price_24h):
                percentage_change_24h = ((predicted_price_24h - current_price) / current_price) * 100
        except Exception as e:
            print(f"!!! Error calculating percentage changes: {e} !!!")
            if not error_message: error_message = "Error calculating prediction changes."
            percentage_change_1h = percentage_change_12h = percentage_change_24h = None # Reset on error
    elif current_price == 0:
        print("Warning: Current price is 0, cannot calculate percentage change.")
    else: # current_price is None
        print("Warning: Current price not available, cannot calculate percentage change.")


    # --- 5. PREPARE DATA FOR PLOTTING ---
    plot_data = None # Initialize
    # Check if we have the necessary dataframes from model fitting and prediction
    if df_merged is not None and forecast is not None:
        try:
            print("--- Preparing data for plotting ---")
            history_points_to_plot = 72 # How many recent actual points to show
            history_to_plot = df_merged.tail(history_points_to_plot)

            # Forecast includes 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
            # We need the part of the forecast that corresponds to the future periods
            # Prophet adds the original history rows to the forecast df, then the future rows
            # Ensure forecast has more rows than df_merged before slicing
            if len(forecast) > len(df_merged):
                 future_plot_points = forecast.iloc[len(df_merged):][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy() # Select only future rows
            else:
                 print("--- Warning: Forecast length not greater than merged data length. No future points to plot. ---")
                 future_plot_points = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper']) # Empty df


            # Combine historical dates and future dates for the X-axis labels
            # Ensure we use datetime objects first, then format
            hist_dates = history_to_plot['ds']
            future_dates = future_plot_points['ds']
            # Handle potential empty future_dates
            if not future_dates.empty:
                 all_dates_dt = pd.concat([hist_dates, future_dates], ignore_index=True).sort_values()
            else:
                 all_dates_dt = hist_dates.sort_values()

            all_dates_str = all_dates_dt.dt.strftime('%Y-%m-%dT%H:%M:%S').unique().tolist() # Unique sorted strings

            # Prepare historical values mapped to the string dates
            hist_values_dict = history_to_plot.set_index(history_to_plot['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'))['y']
            hist_values = [hist_values_dict.get(dt) for dt in all_dates_str]

            # Prepare predicted values mapped to the string dates
            if not future_plot_points.empty:
                 pred_values_dict = future_plot_points.set_index(future_plot_points['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S'))
            else:
                 pred_values_dict = {} # Empty dict if no future points

            pred_values = [pred_values_dict.get(dt, {}).get('yhat') for dt in all_dates_str]
            pred_lower = [pred_values_dict.get(dt, {}).get('yhat_lower') for dt in all_dates_str]
            pred_upper = [pred_values_dict.get(dt, {}).get('yhat_upper') for dt in all_dates_str]

            # Helper function to round safely
            def round_if_not_none(val, digits=2):
                return round(val, digits) if pd.notna(val) else None # Use pd.notna for robustness

            plot_data = {
                "labels": all_dates_str,
                "hist_values": [round_if_not_none(v) for v in hist_values],
                "pred_values": [round_if_not_none(v) for v in pred_values],
                "pred_lower": [round_if_not_none(v) for v in pred_lower],
                "pred_upper": [round_if_not_none(v) for v in pred_upper]
            }
            print(f"--- Prepared plot_data with {len(all_dates_str)} labels. ---")

        except Exception as plot_e:
            print(f"!!! Error preparing plot data: {plot_e} !!!")
            traceback.print_exc()
            plot_data = None # Ensure it's None if prep fails
    else:
        print("--- Skipping plot data preparation (missing model data or forecast) ---")
        plot_data = None


    # --- 6. RENDER TEMPLATE ---
    processing_time = (datetime.now() - start_time).total_seconds()
    print(f"--- Request processing finished in {processing_time:.2f} seconds. Rendering template... ---")
    return render_template('dashboard.html', # Use the correct template name
                           # Pass all calculated variables
                           symbol=symbol,
                           current_price=current_price, # Pass raw value if needed elsewhere
                           current_fng_value=current_fng_value,
                           predicted_price_1h=predicted_price_1h, # Pass raw value
                           predicted_price_12h=predicted_price_12h, # Pass raw value
                           predicted_price_24h=predicted_price_24h, # Pass raw value
                           percentage_change_1h=percentage_change_1h, # Pass raw value
                           percentage_change_12h=percentage_change_12h, # Pass raw value
                           percentage_change_24h=percentage_change_24h, # Pass raw value
                           accuracy_metrics=accuracy_metrics,
                           historical_accuracy_percent=historical_accuracy_percent,
                           data_confidence_score=data_confidence_score,
                           plot_data=plot_data, # Pass the plot data dictionary
                           error_message=error_message,
                           processing_time=f"{processing_time:.2f} sec", # Pass processing time
                           last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z") # Pass update time
                           )


# --- Main execution block ---
if __name__ == '__main__':
    # Set debug=True for development - shows errors in browser and auto-reloads
    # Use host='0.0.0.0' to make accessible on local network if needed
    # Change port if 5000 is occupied
    app.run(debug=True, host='0.0.0.0', port=5000)
