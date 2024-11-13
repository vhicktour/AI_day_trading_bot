import pandas as pd
from datetime import datetime
import os

class MarketDataExporter:
    def __init__(self, data_fetcher):
        self.fetcher = data_fetcher
        self.export_dir = 'market_data'
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def export_current_market_data(self):
        """Export current market data for all USDT pairs"""
        try:
            tickers = self.fetcher.exchange.fetch_tickers()
            market_data = []
            
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT'):
                    market_data.append({
                        'symbol': symbol,
                        'price': ticker['last'],
                        'change_24h': ticker['percentage'],
                        'volume': ticker['quoteVolume'],
                        'high_24h': ticker['high'],
                        'low_24h': ticker['low'],
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            df = pd.DataFrame(market_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{self.export_dir}/market_data_{timestamp}.csv'
            df.to_csv(filename, index=False)
            print(f"Market data exported to {filename}")
            return df
            
        except Exception as e:
            print(f"Error exporting market data: {e}")
            return None

    def export_historical_data(self, symbol, timeframe='1d', limit=100):
        """Export historical data for a specific symbol"""
        try:
            ohlcv = self.fetcher.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{self.export_dir}/{symbol.replace("/", "_")}_{timeframe}_{timestamp}.csv'
            df.to_csv(filename, index=False)
            print(f"Historical data exported to {filename}")
            return df
            
        except Exception as e:
            print(f"Error exporting historical data: {e}")
            return None