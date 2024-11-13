import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import openai
import logging

class TradingChat:
    def __init__(self, ai_signal_generator, data_fetcher, pairs_manager):
        """Initialize trading chat with required components."""
        self.ai = ai_signal_generator
        self.fetcher = data_fetcher
        self.pairs_manager = pairs_manager
        self.current_context = {}
        self.logger = logging.getLogger(__name__)
        
    def get_market_context(self, symbol):
        """Get comprehensive market context for better chat responses."""
        try:
            # Get current market data
            data = self.fetcher.get_latest_data(symbol, granularity='1m', data_limit=50)
            pair_info = self.pairs_manager.get_pair_info(symbol)
            
            if data is None or pair_info is None:
                return None
                
            # Calculate basic metrics
            closing_prices = data['close'].values
            price_changes = np.diff(closing_prices) / closing_prices[:-1] * 100
            
            context = {
                'symbol': symbol,
                'current_price': float(closing_prices[-1]),
                'price_change_24h': pair_info['change_24h'],
                'volume_24h': pair_info['volume'],
                'volatility': float(np.std(price_changes)),
                'trend': 'upward' if np.mean(price_changes) > 0 else 'downward',
                'recent_high': float(pair_info['high']),
                'recent_low': float(pair_info['low']),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add moving averages
            context['sma_20'] = float(pd.Series(closing_prices).rolling(window=20).mean().iloc[-1])
            context['sma_50'] = float(pd.Series(closing_prices).rolling(window=50).mean().iloc[-1])
            
            # Add support and resistance levels
            context['support'] = float(np.percentile(closing_prices, 25))
            context['resistance'] = float(np.percentile(closing_prices, 75))
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {e}")
            return None

    def generate_response(self, user_query, symbol):
        """Generate detailed trading-focused chat response."""
        try:
            # Get fresh market context
            context = self.get_market_context(symbol)
            if not context:
                return "I'm having trouble getting current market data. Please try again."
            
            # Get technical analysis
            analysis = self.ai.get_technical_analysis(
                self.fetcher.get_latest_data(symbol, granularity='1m', data_limit=50)
            )
            
            # Create comprehensive prompt
            prompt = f"""Trading Chat Context:
            Symbol: {symbol}
            Current Price: ${context['current_price']:.4f}
            24h Change: {context['price_change_24h']:+.2f}%
            24h Volume: ${context['volume_24h']:,.2f}
            Recent High: ${context['recent_high']:.4f}
            Recent Low: ${context['recent_low']:.4f}
            Current Trend: {context['trend']}
            Volatility: {context['volatility']:.2f}%
            
            Technical Indicators:
            SMA20: ${context['sma_20']:.4f}
            SMA50: ${context['sma_50']:.4f}
            Support Level: ${context['support']:.4f}
            Resistance Level: ${context['resistance']:.4f}
            
            Analysis Summary:
            Signal: {analysis['primary_signal']}
            Confidence: {analysis['confidence']}%
            Risk Level: {analysis['risk_level']}
            
            User Question: {user_query}
            
            Provide a detailed, trading-focused response that:
            1. Directly addresses the user's question
            2. References relevant market data and technical indicators
            3. Explains any trading signals or patterns
            4. Discusses potential risks and considerations
            5. Avoids definitive predictions
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experienced trading assistant providing data-driven market insights. Be clear, specific, and reference market data. Avoid generic advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=250,
                temperature=0.7
            )
            
            # Format the response for better readability
            chat_response = response['choices'][0]['message']['content'].strip()
            
            # Add current market snapshot
            market_snapshot = f"""
Current Market Snapshot ({datetime.now().strftime('%H:%M:%S')}):
- Price: ${context['current_price']:.4f}
- 24h Change: {context['price_change_24h']:+.2f}%
- Signal: {analysis['primary_signal'].upper()}
- Confidence: {analysis['confidence']}%
- Risk: {analysis['risk_level'].upper()}

Technical Levels:
- Support: ${context['support']:.4f}
- Resistance: ${context['resistance']:.4f}
- SMA20: ${context['sma_20']:.4f}
- SMA50: ${context['sma_50']:.4f}
            """
            
            return f"{chat_response}\n{market_snapshot}"
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"

    def handle_specific_queries(self, query, symbol):
        """Handle specific trading-related queries with detailed responses."""
        query_lower = query.lower()
        
        try:
            if "should i buy" in query_lower or "should i sell" in query_lower:
                analysis = self.ai.get_technical_analysis(
                    self.fetcher.get_latest_data(symbol, granularity='1m', data_limit=50)
                )
                context = self.get_market_context(symbol)
                return f"""
Based on current analysis for {symbol}:
Signal: {analysis['primary_signal'].upper()}
Confidence: {analysis['confidence']}%
Risk Level: {analysis['risk_level'].upper()}

Market Position:
- Price vs SMA20: {'Above' if context['current_price'] > context['sma_20'] else 'Below'}
- Price vs SMA50: {'Above' if context['current_price'] > context['sma_50'] else 'Below'}
- Distance to Support: ${(context['current_price'] - context['support']):.4f}
- Distance to Resistance: ${(context['resistance'] - context['current_price']):.4f}

Analysis: {analysis['reasoning']}

Remember: This is algorithmic analysis, not financial advice. Always manage your risk carefully.
                """
                
            elif "trend" in query_lower:
                data = self.fetcher.get_latest_data(symbol, granularity='1h', data_limit=24)
                if data is not None:
                    prices = data['close'].values
                    trend = np.polyfit(np.arange(len(prices)), prices, 1)[0]
                    trend_str = "upward" if trend > 0 else "downward"
                    strength = abs(trend) / np.mean(prices) * 100
                    
                    # Calculate momentum
                    momentum = (prices[-1] - prices[0]) / prices[0] * 100
                    
                    return f"""
Trend Analysis for {symbol}:
Direction: {trend_str.upper()}
Strength: {'Strong' if strength > 1 else 'Moderate' if strength > 0.5 else 'Weak'}
Momentum: {momentum:+.2f}%

Price Levels:
- 24h High: ${np.max(prices):.4f}
- 24h Low: ${np.min(prices):.4f}
- Current: ${prices[-1]:.4f}

Trend Statistics:
- Consistency: {(np.sum(np.diff(prices) > 0) / (len(prices)-1) * 100):.1f}%
- Volatility: {(np.std(prices) / np.mean(prices) * 100):.2f}%
- Price Range: ${(np.max(prices) - np.min(prices)):.4f}
                    """
                    
            elif "volume" in query_lower:
                context = self.get_market_context(symbol)
                avg_volume = context['volume_24h'] / 24  # Average hourly volume
                current_hour_volume = self.fetcher.get_latest_data(
                    symbol, granularity='1h', data_limit=1
                )['volume'].iloc[0]
                
                return f"""
Volume Analysis for {symbol}:
24h Statistics:
- Total Volume: ${context['volume_24h']:,.2f}
- Average Hourly: ${avg_volume:,.2f}
- Current Hour: ${current_hour_volume:,.2f}

Volume Trends:
- Current vs Average: {'Above' if current_hour_volume > avg_volume else 'Below'} average
- Ratio: {(current_hour_volume / avg_volume * 100):.1f}% of average
- Volume Change: {((current_hour_volume - avg_volume) / avg_volume * 100):+.1f}%

{f'High volume alert! Current volume is {(current_hour_volume / avg_volume):.1f}x average.' if current_hour_volume > avg_volume * 1.5 else ''}
                """
                
            elif "risk" in query_lower:
                context = self.get_market_context(symbol)
                analysis = self.ai.get_technical_analysis(
                    self.fetcher.get_latest_data(symbol, granularity='1m', data_limit=50)
                )
                
                return f"""
Risk Assessment for {symbol}:
Overall Risk Level: {analysis['risk_level'].upper()}

Market Conditions:
- Volatility: {context['volatility']:.2f}%
- Price Stability: {'Stable' if context['volatility'] < 2 else 'Moderate' if context['volatility'] < 5 else 'Volatile'}
- Distance to Support: {((context['current_price'] - context['support']) / context['current_price'] * 100):.1f}%
- Distance to Resistance: {((context['resistance'] - context['current_price']) / context['current_price'] * 100):.1f}%

Risk Factors:
- Market Trend: {context['trend'].upper()}
- Volume Pattern: {'Normal' if 0.8 < context['volume_24h']/(context['volume_24h']/24) < 1.2 else 'Irregular'}
- Price Position: {'Overbought' if context['current_price'] > context['resistance'] else 'Oversold' if context['current_price'] < context['support'] else 'Normal'}

Recommendation:
{analysis['reasoning']}
                """
            
            # Default to regular chat response
            return self.generate_response(query, symbol)
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return f"Error processing query: {e}"

    def get_trade_suggestion(self, symbol):
        """Get actionable trade suggestion with supporting data."""
        try:
            analysis = self.ai.get_technical_analysis(
                self.fetcher.get_latest_data(symbol, granularity='1m', data_limit=50)
            )
            context = self.get_market_context(symbol)
            
            return f"""
Trade Analysis for {symbol} at {datetime.now().strftime('%H:%M:%S')}:

Price Analysis:
Current Price: ${context['current_price']:.4f}
24h Change: {context['price_change_24h']:+.2f}%
Volatility: {context['volatility']:.2f}%

Technical Levels:
- Support: ${context['support']:.4f}
- Resistance: ${context['resistance']:.4f}
- SMA20: ${context['sma_20']:.4f}
- SMA50: ${context['sma_50']:.4f}

Trade Signal:
Recommendation: {analysis['primary_signal'].upper()}
Confidence Level: {analysis['confidence']}%
Risk Level: {analysis['risk_level'].upper()}

Market Conditions:
- Trend: {context['trend'].upper()}
- Volume: ${context['volume_24h']:,.2f}
- Price Position: {'Above' if context['current_price'] > context['sma_20'] else 'Below'} SMA20

Analysis:
{analysis['reasoning']}

Key Levels:
- Entry: ${context['current_price']:.4f}
- Stop Loss: ${context['current_price'] * 0.98:.4f} (-2%)
- Take Profit: ${context['current_price'] * 1.04:.4f} (+4%)

Remember: This is algorithmic analysis. Always use proper risk management.
            """
            
        except Exception as e:
            self.logger.error(f"Error generating trade suggestion: {e}")
            return f"Error generating trade suggestion: {e}"