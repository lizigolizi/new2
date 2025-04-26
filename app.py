from flask import Flask, render_template
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

app = Flask(__name__)

@app.route('/')
def index():
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

def fetch_and_save_news():
    api_key = "2c2282ebd04b825308aaf96ac92cd98b8e537058"
    news_data = fetch_crypto_news(api_key, currency="BTC")
    if news_data:
        with open('news_data.json', 'w') as f:
            json.dump(news_data, f, indent=4)
        print("News data saved to news_data.json")
    else:
        print("Failed to fetch and save news data.")

if __name__ == '__main__':
    fetch_and_save_news() # Save the news data when app.py is run directly
    app.run(debug=True)