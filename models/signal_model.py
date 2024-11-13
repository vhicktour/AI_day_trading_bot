import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib
import logging
from datetime import datetime

class SignalModel:
    def __init__(self, config):
        """Initialize SignalModel with profit-focused parameters."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()
        
        # Profit-focused parameters
        self.profit_threshold = 0.005  # 0.5% minimum potential profit
        self.volume_threshold = 1.5    # 50% above average volume
        self.trend_strength_threshold = 25  # ADX threshold
        self.volatility_threshold = 0.02  # 2% price volatility
        
        # Technical indicator parameters
        self.rsi_period = 14
        self.rsi_overbought = 75  # More aggressive overbought level
        self.rsi_oversold = 25    # More aggressive oversold level
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Price action parameters
        self.support_resistance_periods = 20
        self.breakout_threshold = 0.02  # 2% breakout threshold
        
    def calculate_support_resistance(self, data, periods=20):
        """Calculate dynamic support and resistance levels."""
        try:
            highs = data['high'].rolling(window=periods).max()
            lows = data['low'].rolling(window=periods).min()
            
            # Calculate pivot points
            pivot = (data['high'] + data['low'] + data['close']) / 3
            r1 = 2 * pivot - data['low']
            s1 = 2 * pivot - data['high']
            
            return pd.DataFrame({
                'resistance': highs,
                'support': lows,
                'pivot': pivot,
                'r1': r1,
                's1': s1
            })
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return None

    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators for profit opportunities."""
        try:
            df = data.copy()
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            open_prices = df['open'].values  # Added open_prices to avoid undefined error
            volumes = df['volume'].values
            
            # Core momentum indicators
            df['rsi'] = talib.RSI(close_prices, timeperiod=self.rsi_period)
            macd, signal, hist = talib.MACD(close_prices, 
                                            fastperiod=self.macd_fast, 
                                            slowperiod=self.macd_slow, 
                                            signalperiod=self.macd_signal)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # Trend and volatility indicators
            df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Volume analysis
            df['obv'] = talib.OBV(close_prices, volumes)
            df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=14)
            
            # Volatility bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            
            # Price patterns (using open_prices)
            df['engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            df['morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            df['evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            
            # Additional profit-focused indicators
            df['momentum'] = close_prices - np.roll(close_prices, 10)
            df['rate_of_change'] = (close_prices - np.roll(close_prices, 10)) / np.roll(close_prices, 10) * 100
            
            # Add support/resistance levels
            sr_levels = self.calculate_support_resistance(df)
            if sr_levels is not None:
                df = pd.concat([df, sr_levels], axis=1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return None

    def identify_price_patterns(self, data):
        """Identify profitable price patterns."""
        try:
            df = data.copy()
            patterns = []
            
            # Check for bullish engulfing
            if df['engulfing'].iloc[-1] > 0:
                patterns.append(('buy', 0.8))  # High confidence pattern
                
            # Check for morning star
            if df['morning_star'].iloc[-1] > 0:
                patterns.append(('buy', 0.9))  # Very high confidence pattern
                
            # Check for evening star
            if df['evening_star'].iloc[-1] > 0:
                patterns.append(('sell', 0.9))
                
            # Check for breakouts
            if df['close'].iloc[-1] > df['resistance'].iloc[-1]:
                patterns.append(('buy', 0.85))
                
            if df['close'].iloc[-1] < df['support'].iloc[-1]:
                patterns.append(('sell', 0.85))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error identifying patterns: {e}")
            return []

    def calculate_profit_potential(self, data):
        """Calculate potential profit based on technical setup."""
        try:
            df = data.tail(20)  # Look at recent data
            
            # Calculate volatility
            atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]
            volatility = atr / current_price
            
            # Calculate trend strength
            adx = df['adx'].iloc[-1]
            
            # Calculate volume strength
            volume_ratio = df['volume'].iloc[-1] / df['volume'].mean()
            
            # Calculate momentum
            momentum = df['momentum'].iloc[-1]
            
            # Score different aspects
            volatility_score = min(volatility / self.volatility_threshold, 1)
            trend_score = min(adx / self.trend_strength_threshold, 1)
            volume_score = min(volume_ratio / self.volume_threshold, 1)
            momentum_score = abs(momentum) / current_price
            
            # Combine scores
            total_score = (volatility_score + trend_score + volume_score + momentum_score) / 4
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error calculating profit potential: {e}")
            return 0

    def generate_signal(self, data):
        """Generate trading signal with profit potential."""
        try:
            df = self.calculate_technical_indicators(data)
            if df is None:
                return 'hold', 0
            
            # Get price patterns
            patterns = self.identify_price_patterns(df)
            
            # Calculate profit potential
            profit_potential = self.calculate_profit_potential(df)
            
            latest = df.iloc[-1]
            signals = []
            
            # Strong trend signals
            if latest['adx'] > self.trend_strength_threshold:
                if latest['macd'] > latest['macd_signal'] and latest['rsi'] < 60:
                    signals.append(('buy', 0.7))
                elif latest['macd'] < latest['macd_signal'] and latest['rsi'] > 40:
                    signals.append(('sell', 0.7))
            
            # Volume breakout signals
            volume_ma = pd.Series(df['volume']).rolling(window=20).mean().iloc[-1]
            if latest['volume'] > volume_ma * self.volume_threshold:
                if latest['close'] > latest['bb_upper']:
                    signals.append(('buy', 0.8))
                elif latest['close'] < latest['bb_lower']:
                    signals.append(('sell', 0.8))
            
            # Add pattern signals
            signals.extend(patterns)
            
            # Calculate final signal
            if not signals:
                return 'hold', 0
            
            # Weight signals by their confidence
            buy_confidence = sum(conf for sig, conf in signals if sig == 'buy')
            sell_confidence = sum(conf for sig, conf in signals if sig == 'sell')
            
            # Adjust confidence based on profit potential
            buy_confidence *= (1 + profit_potential)
            sell_confidence *= (1 + profit_potential)
            
            if buy_confidence > sell_confidence and buy_confidence > 0.7:
                return 'buy', min(buy_confidence, 1)
            elif sell_confidence > buy_confidence and sell_confidence > 0.7:
                return 'sell', min(sell_confidence, 1)
            
            return 'hold', 0
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return 'hold', 0

    def calculate_profit_potential(self, data):
        """Calculate potential profit based on technical setup."""
        try:
            df = data.tail(20)  # Look at recent data
            
            # Calculate volatility
            atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]
            volatility = atr / current_price
            
            # Calculate trend strength
            adx = df['adx'].iloc[-1]
            
            # Calculate volume strength using pandas Series
            volume_ratio = df['volume'].iloc[-1] / pd.Series(df['volume']).mean()
            
            # Calculate momentum
            momentum = df['momentum'].iloc[-1]
            
            # Score different aspects
            volatility_score = min(volatility / self.volatility_threshold, 1)
            trend_score = min(adx / self.trend_strength_threshold, 1)
            volume_score = min(volume_ratio / self.volume_threshold, 1)
            momentum_score = abs(momentum) / current_price
            
            # Combine scores
            total_score = (volatility_score + trend_score + volume_score + momentum_score) / 4
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error calculating profit potential: {e}")
            return 0

    def log_trade_opportunity(self, data, signal, confidence):
        """Log detailed analysis of trade opportunity."""
        try:
            df = self.calculate_technical_indicators(data)
            latest = df.iloc[-1]
            profit_potential = self.calculate_profit_potential(df)
            
            # Fix volume ratio calculation
            volume_ma = pd.Series(df['volume']).rolling(window=20).mean().iloc[-1]
            volume_ratio = latest['volume'] / volume_ma
            
            analysis_log = f"""
    Trade Opportunity Analysis ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    ---------------------------
    Signal: {signal.upper()}
    Confidence: {confidence:.2f}
    Profit Potential: {profit_potential:.2f}

    Technical Setup:
    - Trend Strength (ADX): {latest['adx']:.2f}
    - RSI: {latest['rsi']:.2f}
    - MACD Histogram: {latest['macd_hist']:.2f}
    - Volume Ratio: {volume_ratio:.2f}
    - ATR: {latest['atr']:.2f}
    - CCI: {latest['cci']:.2f}

    Price Levels:
    - Current Price: {latest['close']:.2f}
    - Resistance: {latest['resistance']:.2f}
    - Support: {latest['support']:.2f}
    - BB Upper: {latest['bb_upper']:.2f}
    - BB Lower: {latest['bb_lower']:.2f}

    Risk Assessment:
    - Stop Loss: ${latest['support']:.2f} (Support Level)
    - Take Profit: ${latest['resistance']:.2f} (Resistance Level)
    - Risk/Reward Ratio: {(latest['resistance'] - latest['close']) / (latest['close'] - latest['support']):.2f}
            """
            
            self.logger.info(analysis_log)
            print(analysis_log)  # Also print for real-time monitoring
            
        except Exception as e:
            self.logger.error(f"Error logging trade opportunity: {e}")