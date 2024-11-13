import threading
import time
import numpy as np
from collections import deque
from datetime import datetime

class MarketAnalyzer:
    def __init__(self, data_fetcher):
        self.fetcher = data_fetcher
        self.running = False
        self.analysis_thread = None
        self.market_data = {}
        self.alerts = deque(maxlen=100)  # Keep last 100 alerts
        self.analysis_interval = 60  # seconds
        
    def start_analysis(self):
        """Start background analysis"""
        if not self.running:
            self.running = True
            self.analysis_thread = threading.Thread(target=self._analysis_loop)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            print("Background market analysis started")
            
    def stop_analysis(self):
        """Stop background analysis"""
        self.running = False
        if self.analysis_thread:
            self.analysis_thread.join()
            print("Background market analysis stopped")

    def _analyze_volume_spikes(self, ticker_data):
        """Detect unusual volume activity"""
        try:
            avg_volume = ticker_data.get('quoteVolume')
            if avg_volume is None:
                return None
            if avg_volume > 1000000:  # Significant volume threshold
                return f"High volume detected: ${avg_volume:,.2f}"
        except:
            pass
        return None

    def _analyze_price_movement(self, ticker_data):
        """Detect significant price movements"""
        try:
            change = ticker_data.get('percentage')
            if change is None:
                return None
            if abs(change) > 5:  # 5% threshold
                direction = "up" if change > 0 else "down"
                return f"Significant price movement {direction}: {change:+.2f}%"
        except:
            pass
        return None

    def _analyze_patterns(self, symbol, timeframe='1h'):
        """Analyze price patterns"""
        try:
            ohlcv = self.fetcher.exchange.fetch_ohlcv(symbol, timeframe, limit=24)
            closes = [x[4] for x in ohlcv]
            if not closes:
                return None
                
            # Simple trend analysis
            sma_short = np.mean(closes[-5:])
            sma_long = np.mean(closes[-20:])
            
            if sma_short > sma_long * 1.02:  # 2% threshold
                return f"Bullish trend detected: Short MA > Long MA"
            elif sma_short < sma_long * 0.98:
                return f"Bearish trend detected: Short MA < Long MA"
        except:
            pass
        return None

    def _analysis_loop(self):
        """Main analysis loop"""
        while self.running:
            try:
                # Fetch current market data
                tickers = self.fetcher.exchange.fetch_tickers()
                
                for symbol, ticker in tickers.items():
                    if symbol.endswith('/USDT'):
                        # Store current data
                        self.market_data[symbol] = ticker
                        
                        # Run analyses
                        volume_alert = self._analyze_volume_spikes(ticker)
                        price_alert = self._analyze_price_movement(ticker)
                        pattern_alert = self._analyze_patterns(symbol)
                        
                        # Add significant alerts to the queue
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        for alert in [volume_alert, price_alert, pattern_alert]:
                            if alert:
                                self.alerts.append({
                                    'timestamp': timestamp,
                                    'symbol': symbol,
                                    'alert': alert
                                })
                
                # Sleep before next analysis
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                print(f"Analysis error: {e}")
                time.sleep(self.analysis_interval)

    def get_latest_alerts(self, limit=10):
        """Get the most recent alerts"""
        return list(self.alerts)[-limit:]

    def get_market_summary(self):
        """Get current market summary"""
        summary = {
            'top_gainers': [],
            'top_losers': [],
            'high_volume': [],
            'trending': []
        }
        
        for symbol, data in self.market_data.items():
            change = data.get('percentage')
            volume = data.get('quoteVolume')
            if change is None or volume is None:
                continue  # Skip if data is missing
            if change > 5:
                summary['top_gainers'].append({
                    'symbol': symbol,
                    'change': change
                })
            elif change < -5:
                summary['top_losers'].append({
                    'symbol': symbol,
                    'change': change
                })
                
            if volume > 1000000:
                summary['high_volume'].append({
                    'symbol': symbol,
                    'volume': volume
                })
        
        return summary
