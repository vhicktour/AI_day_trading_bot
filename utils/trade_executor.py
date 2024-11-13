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
            self.logger.warning("Maximum daily trades reached")
            return False
            
        if self.daily_loss >= self.max_daily_loss:
            self.logger.warning("Maximum daily loss reached")
            return False
            
        return True
        
    def get_precise_quantity(self, quantity, symbol):
        """Get quantity with correct precision for the symbol."""
        try:
            market = self.exchange.market(symbol)
            precision = market['precision']['amount']
            # Convert to string to avoid float precision issues
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
            current_price = Decimal(str(ticker['last']))
            
            balance = self.exchange.fetch_balance()
            
            if side == 'buy':
                usdt_balance = Decimal(str(balance['free'].get('USDT', 0)))
                # Respect initial investment limit
                max_usdt = min(
                    usdt_balance,
                    Decimal(str(self.config['initial_investment']))
                )
                
                # Calculate quantity based on trade_amount setting
                base_quantity = Decimal(str(self.config['trade_amount']))
                
                # Ensure we don't exceed position size limits
                max_position_value = usdt_balance * Decimal(str(self.max_position_size))
                max_quantity = max_position_value / current_price
                
                quantity = min(base_quantity, max_quantity)
                
            else:  # sell
                asset = symbol.split('/')[0]
                asset_balance = Decimal(str(balance['free'].get(asset, 0)))
                quantity = min(
                    asset_balance,
                    Decimal(str(self.config['trade_amount']))
                )
            
            # Get market limits
            market = self.exchange.market(symbol)
            min_amount = Decimal(str(market['limits']['amount']['min']))
            max_amount = Decimal(str(market['limits']['amount']['max']))
            
            # Apply limits
            quantity = max(min(quantity, max_amount), min_amount)
            
            # Get precise quantity string
            precise_quantity = self.get_precise_quantity(float(quantity), symbol)
            
            if not precise_quantity:
                return None
                
            return precise_quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating trade amount: {e}")
            return None

    def execute_trade(self, signal):
        """Execute trade with all safety checks and limits."""
        if not self.check_daily_limits():
            return None
            
        symbol = self.config['trade_pair']
        
        try:
            # Verify exchange connection
            self.exchange.check_required_credentials()
            
            # Calculate trade amount
            quantity = self.calculate_trade_amount(symbol, signal)
            if not quantity:
                self.logger.error("Failed to calculate valid trade amount")
                return None
            
            # Execute the trade
            if signal == "buy":
                order = self.exchange.create_market_buy_order(
                    symbol,
                    quantity,
                    params={
                        'type': 'market'
                    }
                )
                
                if order['status'] == 'closed':
                    entry_price = Decimal(str(order['price']))
                    stop_loss_price = entry_price * (1 - Decimal(str(self.stop_loss_percentage)))
                    take_profit_price = entry_price * (1 + Decimal(str(self.take_profit_percentage)))
                    
                    # Place stop loss order
                    self.exchange.create_order(
                        symbol,
                        'stop_loss_limit',
                        'sell',
                        quantity,
                        self.get_precise_quantity(float(stop_loss_price), symbol),
                        {'stopPrice': float(stop_loss_price)}
                    )
                    
                    # Place take profit order
                    self.exchange.create_order(
                        symbol,
                        'limit',
                        'sell',
                        quantity,
                        self.get_precise_quantity(float(take_profit_price), symbol)
                    )
                    
            elif signal == "sell":
                order = self.exchange.create_market_sell_order(
                    symbol,
                    quantity,
                    params={
                        'type': 'market'
                    }
                )
            
            # Update daily trade counter
            self.daily_trades += 1
            
            # Log the trade
            self.logger.info(f"Executed {signal} order: {json.dumps(order, indent=2)}")
            print(f"Successfully executed {signal} order for {quantity} {symbol}")
            
            return order
            
        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Insufficient funds: {e}")
            print(f"Error: Insufficient funds for {signal} trade")
            return None
            
        except ccxt.ExchangeError as e:
            self.logger.error(f"Exchange error: {e}")
            print(f"Exchange error: {e}")
            return None
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            print(f"Unexpected error: {e}")
            return None

    def cancel_all_orders(self, symbol):
        """Cancel all open orders for a symbol."""
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                self.exchange.cancel_order(order['id'], symbol)
            self.logger.info(f"Cancelled all open orders for {symbol}")
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")