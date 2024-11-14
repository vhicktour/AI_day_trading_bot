import ccxt
from datetime import datetime, time
import json
import logging
from decimal import Decimal, ROUND_DOWN

class TradeExecutor:
    def __init__(self, config):
        """Initialize trade executor with specific configuration settings."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize exchange with rate limiting and precise calculation
        self.exchange = ccxt.binanceus({
            'apiKey': config['api_key'],
            'secret': config['api_secret'],
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'createMarketBuyOrderRequiresPrice': False,
                'precision': {
                    'price': 8,
                    'amount': 8
                }
            }
        })
        
        # Load risk management settings
        self.max_position_size = config['risk_management']['max_position_size']
        self.stop_loss_percentage = config['risk_management']['stop_loss_percentage']
        self.take_profit_percentage = config['risk_management']['take_profit_percentage']
        self.max_daily_trades = config['risk_management']['max_daily_trades']
        self.max_daily_loss = config['risk_management']['max_daily_loss']
        
        # Load trading rules
        self.min_order_size = config['trading_rules']['min_order_size']
        self.max_order_size = config['trading_rules']['max_order_size']
        self.default_slippage = config['trading_rules']['default_slippage']
        
        # Initialize trade tracking
        self.daily_trades = 0
        self.daily_loss = 0
        self.initial_balance = None

    def check_daily_limits(self):
        """Check if we've exceeded daily trading limits."""
        if self.daily_trades >= self.max_daily_trades:
            print("⚠️ Daily trade limit reached")
            return False
            
        if self.daily_loss >= self.max_daily_loss:
            print("⚠️ Maximum daily loss reached")
            return False
            
        return True

    def get_precise_quantity(self, quantity, symbol):
        """Get quantity with correct precision for the symbol."""
        try:
            market = self.exchange.market(symbol)
            precision = market['precision']['amount']
            return self.exchange.decimal_to_precision(
                quantity, 
                rounding_mode=ROUND_DOWN,
                precision=precision
            )
        except Exception as e:
            self.logger.error(f"Error calculating precise quantity: {e}")
            return None

    def calculate_trade_amount(self, symbol, side):
        """Calculate trade amount respecting all configured limits."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = float(ticker['last'])
            
            # Convert USDT minimums to asset quantity
            min_order_size_quantity = self.min_order_size / current_price
            max_order_size_quantity = self.max_order_size / current_price
            
            balance = self.exchange.fetch_balance()
            
            if side == 'buy':
                available_usdt = float(balance['free'].get('USDT', 0))
                
                # Calculate maximum quantity based on available USDT
                max_possible_quantity = available_usdt / current_price
                
                # Use configured trade amount, but ensure it meets minimums
                desired_quantity = float(self.config['trade_amount'])
                quantity = min(desired_quantity, max_possible_quantity)
                
                # Ensure quantity meets USDT minimums
                if (quantity * current_price) < self.min_order_size:
                    quantity = min_order_size_quantity
                
                if (quantity * current_price) > self.max_order_size:
                    quantity = max_order_size_quantity
                    
            else:  # sell
                asset = symbol.split('/')[0]
                available_asset = float(balance['free'].get(asset, 0))
                quantity = min(available_asset, float(self.config['trade_amount']))
                
                # Check if sell amount meets minimum USDT value
                if (quantity * current_price) < self.min_order_size:
                    print(f"⚠️ Sell amount too small (${quantity * current_price:.2f} USDT)")
                    return None
            
            # Get precise quantity
            precise_quantity = self.get_precise_quantity(quantity, symbol)
            if not precise_quantity:
                return None
                
            # Verify final amount meets requirements
            final_usdt_value = float(precise_quantity) * current_price
            print(f"\n📊 Trade Amount Check:")
            print(f"Quantity: {precise_quantity} {symbol.split('/')[0]}")
            print(f"Value: ${final_usdt_value:.2f} USDT")
            print(f"Minimum Required: ${self.min_order_size:.2f} USDT")
            
            if final_usdt_value < self.min_order_size:
                print("❌ Amount too small")
                return None
                
            return precise_quantity
                
        except Exception as e:
            self.logger.error(f"Error calculating trade amount: {e}")
            return None

    def execute_trade(self, signal):
        """Execute trade with all safety checks and limits."""
        if not self.check_daily_limits():
            print("⚠️ Daily trading limits reached")
            return None
            
        symbol = self.config['trade_pair']
        
        try:
            print(f"\n🔍 DEBUG: Trade Execution Start")
            print(f"Signal: {signal}")
            print(f"Symbol: {symbol}")
            
            # Get current market price first
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = float(ticker['last'])
            print(f"Current Market Price: ${current_price:.8f}")
            
            # Calculate minimum order size in USDT
            market = self.exchange.market(symbol)
            min_amount = float(market['limits']['amount']['min'])
            min_cost_usdt = min_amount * current_price
            print(f"Minimum Order Size: {min_amount} {symbol.split('/')[0]} (${min_cost_usdt:.2f} USDT)")
            
            # Get balance with explicit error checking
            try:
                balance = self.exchange.fetch_balance()
                available_usdt = float(balance['free'].get('USDT', 0))
                print(f"Available USDT Balance: ${available_usdt:.2f}")
            except Exception as e:
                print(f"❌ Error fetching balance: {e}")
                return None

            # Calculate trade amount (in base currency)
            quantity = self.config['trade_amount']  # Direct amount from config
            required_usdt = quantity * current_price
            
            print(f"\n📊 Trade Calculation:")
            print(f"Attempting to trade: {quantity} {symbol.split('/')[0]}")
            print(f"Estimated cost: ${required_usdt:.2f} USDT")
            
            if signal == "buy":
                if available_usdt < required_usdt:
                    print("❌ Insufficient USDT for buy order")
                    return None
                    
                try:
                    print("\n🚀 Executing market buy order...")
                    order = self.exchange.create_market_buy_order(
                        symbol=symbol,
                        amount=quantity,
                        params={
                            'quoteOrderQty': required_usdt  # This ensures we use exact USDT amount
                        }
                    )
                    
                    print("\n✅ Buy order result:")
                    print(f"Status: {order['status']}")
                    print(f"Filled: {order.get('filled', 0)}")
                    print(f"Cost: ${order.get('cost', 0):.2f}")
                    
                    # Verify the order was actually executed
                    if order['status'] == 'closed' and order.get('filled', 0) > 0:
                        print("Order successfully executed!")
                        return order
                    else:
                        print("❌ Order was not filled")
                        return None
                        
                except Exception as e:
                    print(f"❌ Buy order failed: {str(e)}")
                    return None
                    
            elif signal == "sell":
                # Check if we have the asset to sell
                base_currency = symbol.split('/')[0]
                available_asset = float(balance['free'].get(base_currency, 0))
                
                if available_asset < quantity:
                    print(f"❌ Insufficient {base_currency} for sell order")
                    return None
                    
                try:
                    print("\n🚀 Executing market sell order...")
                    order = self.exchange.create_market_sell_order(
                        symbol=symbol,
                        amount=quantity
                    )
                    
                    print("\n✅ Sell order result:")
                    print(f"Status: {order['status']}")
                    print(f"Filled: {order.get('filled', 0)}")
                    print(f"Cost: ${order.get('cost', 0):.2f}")
                    
                    if order['status'] == 'closed' and order.get('filled', 0) > 0:
                        print("Order successfully executed!")
                        return order
                    else:
                        print("❌ Order was not filled")
                        return None
                        
                except Exception as e:
                    print(f"❌ Sell order failed: {str(e)}")
                    return None
                    
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return None

    # Add this new method to test trade execution
    def test_market_order(self, signal):
        """Test market order execution with minimum amount."""
        symbol = self.config['trade_pair']
        
        try:
            # Get market info
            market = self.exchange.market(symbol)
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Calculate minimum order
            min_amount = float(market['limits']['amount']['min'])
            current_price = float(ticker['last'])
            min_cost = min_amount * current_price
            
            print(f"\n🧪 Testing market order:")
            print(f"Symbol: {symbol}")
            print(f"Signal: {signal}")
            print(f"Minimum amount: {min_amount} {symbol.split('/')[0]}")
            print(f"Current price: ${current_price:.8f}")
            print(f"Minimum cost: ${min_cost:.2f} USDT")
            
            # Get balance
            balance = self.exchange.fetch_balance()
            available_usdt = float(balance['free'].get('USDT', 0))
            print(f"Available USDT: ${available_usdt:.2f}")
            
            if available_usdt < min_cost and signal == 'buy':
                print("❌ Insufficient funds for test")
                return None
                
            # Execute test order
            try:
                if signal == 'buy':
                    print("\n🚀 Testing market buy...")
                    order = self.exchange.create_market_buy_order(
                        symbol,
                        min_amount,
                        params={'quoteOrderQty': min_cost}
                    )
                else:
                    print("\n🚀 Testing market sell...")
                    order = self.exchange.create_market_sell_order(
                        symbol,
                        min_amount
                    )
                
                print("\n✅ Test order result:")
                print(json.dumps(order, indent=2))
                return order
                
            except Exception as e:
                print(f"❌ Test order failed: {str(e)}")
                return None
                
        except Exception as e:
            print(f"❌ Test setup failed: {str(e)}")
            return None

    def verify_trade_execution(self, order, symbol):
        """Verify that a trade was actually executed."""
        try:
            if not order:
                return False
                
            # Check order status
            order_id = order['id']
            fetched_order = self.exchange.fetch_order(order_id, symbol)
            
            if fetched_order['status'] == 'closed':
                # Verify the trade in recent trades
                trades = self.exchange.fetch_my_trades(symbol, limit=1)
                if trades and trades[0]['order'] == order_id:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error verifying trade: {e}")
            return False

    def cancel_all_orders(self, symbol):
        """Cancel all open orders for a symbol."""
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                self.exchange.cancel_order(order['id'], symbol)
            self.logger.info(f"Cancelled all open orders for {symbol}")
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")