import logging
from typing import List, Dict, Optional
import pandas as pd

class PairsManager:
    def __init__(self, fetcher, config):
        self.fetcher = fetcher
        self.config = config
        self.pairs_config = config.get('pairs_config', {})
        self.available_pairs = {}
        self.active_pairs = set()
        self.logger = logging.getLogger(__name__)

    def update_available_pairs(self) -> Dict:
        """Update and return available trading pairs with their current data"""
        try:
            markets = self.fetcher.exchange.load_markets()
            tickers = self.fetcher.exchange.fetch_tickers()
            
            self.available_pairs = {}
            quote_currencies = self.pairs_config.get('quote_currencies', ['USDT'])
            min_volume = self.pairs_config.get('min_volume', 10000)
            excluded_pairs = set(self.pairs_config.get('excluded_pairs', []))
            
            for symbol, ticker in tickers.items():
                # Check if the pair ends with any of our quote currencies
                if any(symbol.endswith(f"/{quote}") for quote in quote_currencies):
                    try:
                        volume = ticker.get('quoteVolume')
                        if volume is None or ticker.get('last') is None:
                            continue  # Skip if essential data is missing
                        if volume >= min_volume and symbol not in excluded_pairs:
                            self.available_pairs[symbol] = {
                                'symbol': symbol,
                                'price': ticker.get('last', 0),
                                'change_24h': ticker.get('percentage', 0),
                                'volume': volume,
                                'high': ticker.get('high', 0),
                                'low': ticker.get('low', 0)
                            }
                    except (KeyError, TypeError):
                        continue
            
            return self.available_pairs
            
        except Exception as e:
            self.logger.error(f"Error updating pairs: {str(e)}")
            return {}

    def get_sorted_pairs(self, sort_by: str = 'volume') -> List[Dict]:
        """Get pairs sorted by specified criterion"""
        pairs_list = list(self.available_pairs.values())
        
        if sort_by == 'volume':
            return sorted(pairs_list, key=lambda x: x.get('volume') or 0, reverse=True)
        elif sort_by == 'change':
            return sorted(pairs_list, key=lambda x: x.get('change_24h') or 0, reverse=True)
        elif sort_by == 'price':
            return sorted(pairs_list, key=lambda x: x.get('price') or 0, reverse=True)
        
        return pairs_list

    def export_pairs_data(self, filename: str = 'pairs_data.csv'):
        """Export current pairs data to CSV"""
        df = pd.DataFrame(self.available_pairs.values())
        df.to_csv(filename, index=False)
        return df

    def get_pair_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed information for a specific pair"""
        try:
            if symbol in self.available_pairs:
                ticker = self.fetcher.exchange.fetch_ticker(symbol)
                market = self.fetcher.exchange.market(symbol)
                
                return {
                    **self.available_pairs[symbol],
                    'bid': ticker.get('bid'),
                    'ask': ticker.get('ask'),
                    'spread': (ticker.get('ask', 0) - ticker.get('bid', 0)) if ticker.get('ask') and ticker.get('bid') else None,
                    'base_currency': market['base'],
                    'quote_currency': market['quote'],
                    'min_amount': market.get('limits', {}).get('amount', {}).get('min'),
                    'max_amount': market.get('limits', {}).get('amount', {}).get('max'),
                    'min_price': market.get('limits', {}).get('price', {}).get('min'),
                    'max_price': market.get('limits', {}).get('price', {}).get('max'),
                }
        except Exception as e:
            self.logger.error(f"Error getting pair info for {symbol}: {str(e)}")
        
        return None

    def check_pair_validity(self, symbol: str) -> bool:
        """Check if a pair is valid and meets minimum requirements"""
        try:
            if symbol not in self.available_pairs:
                self.update_available_pairs()
            
            return symbol in self.available_pairs
        except Exception as e:
            self.logger.error(f"Error checking pair validity: {str(e)}")
            return False

    def get_top_movers(self, limit: int = 10) -> Dict[str, List[Dict]]:
        """Get top gainers and losers"""
        pairs = list(self.available_pairs.values())
        sorted_pairs = sorted(pairs, key=lambda x: x.get('change_24h') or 0, reverse=True)
        
        return {
            'gainers': sorted_pairs[:limit],
            'losers': sorted_pairs[-limit:]
        }

    def get_high_volume_pairs(self, threshold_multiplier: float = 2.0) -> List[Dict]:
        """Get pairs with unusually high volume"""
        pairs = list(self.available_pairs.values())
        avg_volume = sum(p.get('volume', 0) for p in pairs) / len(pairs) if pairs else 0
        
        return [
            pair for pair in pairs
            if pair.get('volume', 0) > avg_volume * threshold_multiplier
        ]
