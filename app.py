from flask import Flask, render_template
<<<<<<< Updated upstream
from textblob import TextBlob
import requests
import json

CRYPTOPANIC_API_KEY = "2c2282ebd04b825308aaf96ac92cd98b8e537058"  # Replace with your actual API key

def fetch_crypto_news(api_key, currency="BTC", page=1):
    """Fetches cryptocurrency news for a specific currency from CryptoPanic."""
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={currency}&public=true&page={page}"
    try:
        print(f"Fetching news from: {url}")  # This will show you the URL being requested
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful (status code 200)
        data = response.json()
        # print(json.dumps(data, indent=4)) # Uncomment this to see the raw JSON response
        return data.get('results', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

def fetch_crypto_price(coin_id="bitcoin", vs_currency="usd"):
    """Fetches the current price of a cryptocurrency from CoinGecko."""
    url = f"{COINGECKO_BASE_URL}/simple/price?ids={coin_id}&vs_currencies={vs_currency}"
    try:
        print(f"Fetching price from: {url}") # Show the price API URL
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # print(json.dumps(data, indent=4)) # Uncomment to see the raw JSON response
        return data.get(coin_id, {}).get(vs_currency)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching price: {e}")
        return None
=======
import requests
import json
import pandas as pd
from datetime import datetime
>>>>>>> Stashed changes

app = Flask(__name__)

def fetch_gemini_historical_data(symbol, timeframe='1day'):
    base_url = "https://api.gemini.com/v2"
    endpoint = f"/candles/{symbol}/hist/{timeframe}"
    url = f"{base_url}{endpoint}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = json.loads(response.text)
        # Gemini API returns data as [timestamp_ms, open, high, low, close]
        # Format for Chart.js: [timestamp_ms, close_price]
        prices = [[entry[0], float(entry[4])] for entry in data]
        # Sort by timestamp (Gemini usually returns in descending order)
        prices.sort(key=lambda x: x[0])
        return prices
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Gemini data for {symbol}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for {symbol}: {e}")
        return None

def simple_moving_average_prediction(prices_data, window=5):
    if not prices_data or len(prices_data) < window:
        return None
    prices = pd.Series([price for _, price in prices_data])
    sma = prices.rolling(window=window).mean().iloc[-1]
    return sma

@app.route('/')
def index():
<<<<<<< Updated upstream
    news_articles = fetch_crypto_news(CRYPTOPANIC_API_KEY, currency="BTC")
    btc_price = fetch_crypto_price(coin_id="bitcoin")
    processed_news = []

    if news_articles:
        for article in news_articles:
            title = article['title']
            source = article['source']['title']
            analysis = TextBlob(title)
            sentiment_polarity = analysis.sentiment.polarity
            sentiment_label = "neutral"
            if sentiment_polarity > 0.01:
                sentiment_label = "positive"
            elif sentiment_polarity < -0.01:
                sentiment_label = "negative"
            processed_news.append({'title': title, 'source': source, 'sentiment': sentiment_label})

    return render_template('index.html', news=processed_news, btc_price=btc_price)
=======
    symbol = 'BTCUSD'  # Example: Bitcoin/USD on Gemini
    historical_data = fetch_gemini_historical_data(symbol, timeframe='1day')
    print(f"Historical Data in Flask: {historical_data}") # For debugging
    predicted_price = None
    if historical_data:
        predicted_price = simple_moving_average_prediction(historical_data)

    return render_template('index.html', historical_data=historical_data, predicted_price=predicted_price, symbol=symbol)
>>>>>>> Stashed changes

if __name__ == '__main__':
    app.run(debug=True)