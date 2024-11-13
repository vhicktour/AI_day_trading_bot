import ccxt
from datetime import datetime, time

class TradeExecutor:
    def __init__(self, config):
        self.config = config
        self.exchange = ccxt.binanceus({
            'apiKey': config['api_key'],
            'secret': config['api_secret'],
            'enableRateLimit': True,
        })

    def is_market_open(self):
        """Check if the U.S. stock market is open (9:30 AM - 4:00 PM Eastern Time)."""
        now = datetime.now()
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= now.time() <= market_close

    def execute_trade(self, signal):
        symbol = self.config['trade_pair']
        trade_amount = self.config['trade_amount']

        # Determine if the market is open for stock/ETF pairs, allowing 24/7 trading for crypto pairs
        if not symbol.endswith("USDT") and not self.is_market_open():
            print(f"Market is closed for {symbol}. Skipping trade.")
            return None

        try:
            if signal == "buy":
                # Execute a buy order
                order = self.exchange.create_market_buy_order(symbol, trade_amount)
                print(f"Placed buy order: {order}")
                return order
            elif signal == "sell":
                # Execute a sell order
                order = self.exchange.create_market_sell_order(symbol, trade_amount)
                print(f"Placed sell order: {order}")
                return order
            else:
                print("No trade executed. Signal was 'hold'.")
                return None
        except ccxt.ExchangeError as e:
            if "Market is closed" in str(e):
                print(f"Trade execution error: Market is closed for {symbol}. Skipping trade.")
            else:
                print(f"Trade execution error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
