import ccxt
import pandas as pd
import time

class DataFetcher:
    def __init__(self, config):
        self.config = config
        self.exchange = ccxt.binanceus({
            'apiKey': config['api_key'],
            'secret': config['api_secret'],
        })

    def get_latest_data(self, trade_pair, granularity='1m', data_limit=50):
        try:
            # Ensure 'granularity' uses Binance-compatible intervals like '1m' for 1-minute data
            data = self.exchange.fetch_ohlcv(trade_pair, timeframe=granularity, limit=data_limit)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            return None
