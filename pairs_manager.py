import pandas as pd
import numpy as np
from datetime import datetime

class PairsManager:
    def __init__(self, data_fetcher, config):
        """
        Initialize PairsManager with data fetcher and configuration.
        
        Args:
            data_fetcher: DataFetcher instance for market data retrieval
            config: Configuration dictionary containing pairs settings
        """
        self.fetcher = data_fetcher
        self.config = config
        self.available_pairs = []
        self.pairs_data = {}
        self.volume_threshold = config['pairs_config'].get('min_volume', 10000)
        self.quote_currencies = config['pairs_config'].get('quote_currencies', ['USDT'])
        self.update_interval = config['pairs_config'].get('update_interval', 60)  # seconds
        self.last_update = 0

    def update_available_pairs(self):
        """Update the list of available trading pairs and their market data."""
        current_time = datetime.now().timestamp()
        
        # Only update if enough time has passed since last update
        if current_time - self.last_update < self.update_interval:
            return
            
        try:
            # Fetch all tickers from the exchange
            tickers = self.fetcher.exchange.fetch_tickers()
            
            pairs_data = []
            for symbol, ticker in tickers.items():
                # Check if pair ends with configured quote currency
                if any(symbol.endswith(f"/{quote}") for quote in self.quote_currencies):
                    # Calculate basic metrics
                    base_currency = symbol.split('/')[0]
                    quote_currency = symbol.split('/')[1]
                    volume = ticker.get('quoteVolume', 0)
                    
                    # Only include pairs meeting minimum volume requirement
                    if volume >= self.volume_threshold:
                        pair_data = {
                            'symbol': symbol,
                            'base_currency': base_currency,
                            'quote_currency': quote_currency,
                            'price': ticker.get('last', 0),
                            'change_24h': ticker.get('percentage', 0),
                            'volume': volume,
                            'high': ticker.get('high', 0),
                            'low': ticker.get('low', 0),
                            'spread': ticker.get('ask', 0) - ticker.get('bid', 0) if ticker.get('ask') and ticker.get('bid') else 0,
                            'min_amount': self.fetcher.exchange.markets.get(symbol, {}).get('limits', {}).get('amount', {}).get('min'),
                            'max_amount': self.fetcher.exchange.markets.get(symbol, {}).get('limits', {}).get('amount', {}).get('max')
                        }
                        pairs_data.append(pair_data)
            
            # Update class variables
            self.pairs_data = {pair['symbol']: pair for pair in pairs_data}
            self.available_pairs = [pair['symbol'] for pair in pairs_data]
            self.last_update = current_time
            
        except Exception as e:
            print(f"Error updating pairs: {e}")

    def get_pair_info(self, symbol):
        """
        Get detailed information for a specific trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
        
        Returns:
            dict: Pair information or None if not found
        """
        return self.pairs_data.get(symbol)

    def check_pair_validity(self, symbol):
        """
        Check if a trading pair is valid and available.
        
        Args:
            symbol: Trading pair symbol to check
        
        Returns:
            bool: True if pair is valid, False otherwise
        """
        return symbol in self.available_pairs

    def get_sorted_pairs(self, sort_by='volume'):
        """
        Get list of pairs sorted by specified metric.
        
        Args:
            sort_by: Metric to sort by ('volume', 'change', or 'price')
        
        Returns:
            list: Sorted list of pair dictionaries
        """
        pairs_list = list(self.pairs_data.values())
        
        if sort_by == 'volume':
            return sorted(pairs_list, key=lambda x: x['volume'], reverse=True)
        elif sort_by == 'change':
            return sorted(pairs_list, key=lambda x: abs(x['change_24h']), reverse=True)
        elif sort_by == 'price':
            return sorted(pairs_list, key=lambda x: x['price'], reverse=True)
        
        return pairs_list

    def get_top_movers(self, limit=5):
        """
        Get top gaining and losing pairs.
        
        Args:
            limit: Number of pairs to return for each category
        
        Returns:
            dict: Dictionary containing lists of top gainers and losers
        """
        pairs_list = list(self.pairs_data.values())
        sorted_pairs = sorted(pairs_list, key=lambda x: x['change_24h'], reverse=True)
        
        return {
            'gainers': sorted_pairs[:limit],
            'losers': sorted_pairs[-limit:]
        }

    def get_high_volume_pairs(self, threshold_multiplier=2):
        """
        Get pairs with unusually high volume.
        
        Args:
            threshold_multiplier: Multiplier for volume threshold
        
        Returns:
            list: List of pairs with high volume
        """
        pairs_list = list(self.pairs_data.values())
        avg_volume = np.mean([pair['volume'] for pair in pairs_list])
        threshold = avg_volume * threshold_multiplier
        
        high_volume_pairs = [pair for pair in pairs_list if pair['volume'] > threshold]
        return sorted(high_volume_pairs, key=lambda x: x['volume'], reverse=True)

    def export_pairs_data(self, filename):
        """
        Export current pairs data to CSV file.
        
        Args:
            filename: Path to save the CSV file
        """
        df = pd.DataFrame(list(self.pairs_data.values()))
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df.to_csv(filename, index=False)
        print(f"Pairs data exported to {filename}")

    def get_volatility_metrics(self, symbol, timeframe='1h', periods=24):
        """
        Calculate volatility metrics for a specific pair.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval for data
            periods: Number of periods to analyze
        
        Returns:
            dict: Volatility metrics
        """
        try:
            ohlcv = self.fetcher.exchange.fetch_ohlcv(symbol, timeframe, limit=periods)
            if not ohlcv:
                return None
                
            prices = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            metrics = {
                'volatility': np.std(prices['close'].pct_change().dropna()) * 100,
                'high_low_range': ((prices['high'].max() - prices['low'].min()) / prices['close'].mean()) * 100,
                'avg_volume': prices['volume'].mean(),
                'price_range_percent': ((prices['close'].max() - prices['close'].min()) / prices['close'].min()) * 100
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating volatility metrics: {e}")
            return None