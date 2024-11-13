import openai
import numpy as np
import json

class AIBasedSignal:
    def __init__(self, openai_key, data_fetcher, trade_pair):
        """Initialize AI signal generator with structured prompts."""
        openai.api_key = openai_key
        self.data_fetcher = data_fetcher
        self.trade_pair = trade_pair
        
    def get_ai_signal(self, data):
        """Get trading signal using structured prompt."""
        closing_prices = data['close'][-50:].tolist()
        
        # Create a structured prompt that forces a specific response format
        prompt = f"""Analyze these closing prices and respond ONLY with one of these three words: 'buy', 'sell', or 'hold'.
        Recent closing prices: {closing_prices}
        
        Rules:
        - If prices show a clear upward trend and buying conditions, respond with: buy
        - If prices show a clear downward trend and selling conditions, respond with: sell
        - If the trend is unclear or neutral, respond with: hold
        
        Response (just the word):"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are a trading bot that only responds with single-word trading signals: 'buy', 'sell', or 'hold'."
                },
                {
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=5,
                temperature=0.0
            )
            
            signal = response['choices'][0]['message']['content'].strip().lower()
            if signal in ['buy', 'sell', 'hold']:
                return signal
            return 'hold'
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return 'hold'

    def get_confidence(self, data):
        """Get confidence score using structured prompt."""
        closing_prices = data['close'][-50:].tolist()
        
        # Create a structured prompt that forces a numerical response
        prompt = f"""Analyze these closing prices and respond ONLY with a number between 0 and 100 representing your confidence level.
        Recent closing prices: {closing_prices}
        
        Rules:
        - Respond only with a number between 0 and 100
        - Higher confidence (closer to 100) means stronger signals
        - Lower confidence (closer to 0) means weaker signals
        - Consider factors like trend strength, volatility, and price momentum
        
        Response (just the number):"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are a trading bot that only responds with numerical confidence scores between 0 and 100."
                },
                {
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=3,
                temperature=0.0
            )
            
            confidence_text = response['choices'][0]['message']['content'].strip()
            try:
                confidence = float(confidence_text) / 100  # Convert percentage to decimal
                return min(max(confidence, 0), 1)  # Ensure it's between 0 and 1
            except ValueError:
                return 0.5  # Default confidence if conversion fails
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return 0.5

    def get_technical_analysis(self, data):
        """Get detailed technical analysis with structured format."""
        closing_prices = data['close'][-50:].tolist()
        price_changes = np.diff(closing_prices) / closing_prices[:-1] * 100
        volatility = np.std(price_changes)
        trend = np.mean(price_changes)
        
        prompt = f"""Analyze these market metrics and provide a technical analysis response in JSON format:
        Closing Prices (last 50): {closing_prices}
        Recent Volatility: {volatility:.2f}%
        Price Trend: {trend:.2f}%
        
        Respond with a JSON object containing:
        - primary_signal: "buy", "sell", or "hold"
        - confidence: number between 0 and 100
        - reasoning: brief explanation
        - risk_level: "low", "medium", or "high"
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are a trading bot that responds with JSON-formatted technical analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=150,
                temperature=0.0
            )
            
            analysis = json.loads(response['choices'][0]['message']['content'])
            return analysis
            
        except Exception as e:
            print(f"Error in technical analysis: {e}")
            return {
                "primary_signal": "hold",
                "confidence": 50,
                "reasoning": "Analysis error - defaulting to hold",
                "risk_level": "medium"
            }

    def chat(self, query):
        """Enhanced chat function with structured market insights."""
        data = self.data_fetcher.get_latest_data(self.trade_pair, granularity='1m', data_limit=50)
        if data is None:
            return "Unable to retrieve market data."
            
        latest_price = self.get_latest_price()
        recent_trend = data['close'][-5:].tolist()
        
        # Get technical analysis for context
        analysis = self.get_technical_analysis(data)
        
        prompt = f"""Context:
        Trading Pair: {self.trade_pair}
        Current Price: {latest_price}
        Recent Trend: {recent_trend}
        Technical Analysis: {json.dumps(analysis, indent=2)}
        
        User Query: {query}
        
        Provide a clear, data-driven response addressing the user's query."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are a professional trading assistant providing data-driven market insights."
                },
                {
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=150,
                temperature=0.5
            )
            
            return response['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            return f"Error generating response: {e}"

    def get_latest_price(self):
        """Get latest price with error handling."""
        try:
            data = self.data_fetcher.get_latest_data(
                self.trade_pair, 
                granularity='1m', 
                data_limit=1
            )
            if data is not None and 'close' in data.columns:
                return data['close'].iloc[-1]
            return None
        except Exception as e:
            print(f"Error fetching price: {e}")
            return None