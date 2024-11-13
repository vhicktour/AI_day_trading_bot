import openai
import numpy as np

class AIBasedSignal:
    def __init__(self, openai_key, data_fetcher, trade_pair):
        openai.api_key = openai_key
        self.data_fetcher = data_fetcher
        self.trade_pair = trade_pair

    def get_ai_signal(self, data):
        closing_prices = data['close'][-50:].tolist()
        input_text = f"Given the following closing prices: {closing_prices}. Should I 'buy', 'sell', or 'hold'?"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": input_text}],
                max_tokens=5,
                temperature=0.0
            )
            signal = response['choices'][0]['message']['content'].strip().lower()
            if signal in ['buy', 'sell', 'hold']:
                return signal
            else:
                return 'hold'
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return 'hold'

    def get_confidence(self, data):
        closing_prices = data['close'][-50:].tolist()
        input_text = f"Given the following closing prices: {closing_prices}. What confidence level (0-100) would you assign to your decision to 'buy', 'sell', or 'hold'?"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": input_text}],
                max_tokens=10,
                temperature=0.0
            )
            confidence_text = response['choices'][0]['message']['content'].strip()
            confidence = float(confidence_text) / 100  # Convert percentage to decimal
            return min(max(confidence, 0), 1)  # Ensure it's between 0 and 1
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return 0.5  # Default confidence if there's an error

    def get_latest_price(self):
        """Retrieve the latest real-time price for the current trading pair."""
        try:
            # Change granularity to '1m' for real-time data in Binance-compatible format
            data = self.data_fetcher.get_latest_data(self.trade_pair, granularity='1m', data_limit=1)
            if data is not None and 'close' in data.columns:
                return data['close'].iloc[-1]  # Return the most recent closing price
            return "No data available."
        except Exception as e:
            print(f"Error fetching real-time price: {e}")
            return "Unable to retrieve real-time data."

    def chat(self, query):
        """Chat with AI for market insights and advice, with recent market data included."""
        # Fetch the latest market data to use as context
        data = self.data_fetcher.get_latest_data(self.trade_pair, granularity='1m', data_limit=50)
        if data is None:
            return "I couldn't retrieve market data at the moment. Please try again later."

        # Get the latest price for more accurate responses
        latest_price = self.get_latest_price()
        recent_trend = data['close'][-5:].tolist()
        input_text = (
            f"{query} Here is the latest market data for {self.trade_pair}: "
            f"The **current price** of {self.trade_pair} is {latest_price}. "
            f"Recent closing prices over the last five intervals are: {recent_trend}."
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": input_text}],
                max_tokens=100,
                temperature=0.5
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error in AI response: {e}"

