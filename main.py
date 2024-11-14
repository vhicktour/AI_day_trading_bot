import sys
import os
import json
import time
import threading
import pandas as pd
import logging
from datetime import datetime
import ccxt
import talib
import numpy as np



# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration and define constants
try:
    with open('config/config.json') as f:
        config = json.load(f)
    logger.info("Successfully loaded config")
    
    # Define constants
    MIN_REQUIRED_BALANCE = float(config.get('initial_investment', 50))
    MIN_TRADE_BALANCE = float(config['trading_rules']['min_order_size'])
    
except Exception as e:
    logger.error(f"Error loading config: {str(e)}")
    sys.exit(1)

# Import custom modules
from utils.data_fetcher import DataFetcher
from strategies.moving_average_strategy import MovingAverageStrategy
from scripts.openai_integration import AIBasedSignal
from utils.trade_executor import TradeExecutor
from models.dqn_model import DQNModel
from train_dqn import DQNAgent
from market_analyzer import MarketAnalyzer
from market_exporter import MarketDataExporter
from pairs_manager import PairsManager
from trading_chat import TradingChat
from models.signal_model import SignalModel

# Initialize global variables
current_pair = config['trade_pair']
running = False
confidence_threshold = 0.7
daily_trades = 0
daily_loss = 0

# Initialize components
try:
    fetcher = DataFetcher(config)
    executor = TradeExecutor(config)  # Initialize executor here
    ai_signal_generator = AIBasedSignal(config['openai_key'], fetcher, current_pair)
    market_analyzer = MarketAnalyzer(fetcher)
    data_exporter = MarketDataExporter(fetcher)
    pairs_manager = PairsManager(fetcher, config)
    trading_chat = TradingChat(ai_signal_generator, fetcher, pairs_manager)
    signal_model = SignalModel(config)
    logger.info("Successfully initialized all components")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    sys.exit(1)


# Utility Functions
def check_balance():
    """Check current balance and verify if it meets minimum requirements."""
    global executor  # Ensure access to 'executor' defined in main script
    try:
        if executor is None:
            raise ValueError("Executor is not initialized")
        balance = executor.exchange.fetch_balance()
        available_balance = float(balance['free'].get('USDT', 0))
        total_balance = float(balance['total'].get('USDT', 0))
        
        status = {
            'available': available_balance,
            'total': total_balance,
            'sufficient': available_balance >= MIN_REQUIRED_BALANCE,
            'can_trade': available_balance >= MIN_TRADE_BALANCE
        }
        
        logger.info(f"Balance check - Available: ${available_balance:.2f} USDT, Total: ${total_balance:.2f} USDT")
        return status
        
    except Exception as e:
        logger.error(f"Error checking balance: {e}")
        return None

def check_current_balance():
    """Display detailed balance information."""
    global executor, daily_trades, daily_loss  # Add executor here to ensure access
    
    try:
        balance = executor.exchange.fetch_balance()
        available_usdt = float(balance['free'].get('USDT', 0))
        total_usdt = float(balance['total'].get('USDT', 0))
        
        print("\n=== Current Balance Information ===")
        print(f"Available USDT: ${available_usdt:.2f}")
        print(f"Total USDT: ${total_usdt:.2f}")
        
        # Show current positions if any
        current_pair_base = current_pair.split('/')[0]
        if current_pair_base in balance['free']:
            asset_balance = float(balance['free'][current_pair_base])
            asset_total = float(balance['total'][current_pair_base])
            try:
                current_price = float(fetcher.get_latest_data(current_pair, granularity='1m', data_limit=1)['close'].iloc[-1])
                print(f"\nCurrent {current_pair_base} Holdings:")
                print(f"Available: {asset_balance:.8f} {current_pair_base}")
                print(f"Total: {asset_total:.8f} {current_pair_base}")
                print(f"Estimated Value: ${(asset_total * current_price):.2f} USDT")
            except Exception as e:
                logger.error(f"Error fetching current price: {e}")
                print(f"\nCurrent {current_pair_base} Holdings:")
                print(f"Available: {asset_balance:.8f} {current_pair_base}")
                print(f"Total: {asset_total:.8f} {current_pair_base}")
                print("Unable to estimate current value - price data unavailable")
        
        # Show daily trading stats
        print("\nTrading Statistics:")
        print(f"Daily Trades: {daily_trades}/{config['risk_management']['max_daily_trades']}")
        max_daily_loss_amount = total_usdt * config['risk_management']['max_daily_loss']
        print(f"Daily Loss: ${daily_loss:.2f} USDT (Max: ${max_daily_loss_amount:.2f} USDT)")
        
        # Show trading limits
        print("\nTrading Limits:")
        print(f"Minimum Order Size: ${config['trading_rules']['min_order_size']:.2f} USDT")
        print(f"Maximum Order Size: ${config['trading_rules']['max_order_size']:.2f} USDT")
        print(f"Maximum Position Size: {config['risk_management']['max_position_size'] * 100}% of balance")
        
        return available_usdt
        
    except Exception as e:
        logger.error(f"Error checking balance: {e}")
        print(f"Error: Unable to fetch balance information - {e}")
        return None



def view_market_data():
    """Display enhanced market data with pairs manager"""
    try:
        print("\n=== Market Overview ===")
        # Update available trading pairs
        pairs_manager.update_available_pairs()
        
        print("\nSort by:")
        print("1. Volume (Default)")
        print("2. 24h Change")
        print("3. Price")
        
        choice = input("Enter sort option (1-3): ").strip()
        
        if choice == "2":
            pairs = pairs_manager.get_sorted_pairs('change')  # Sort pairs by 24h change
            print("\n=== Sorted by 24h Change ===")
        elif choice == "3":
            pairs = pairs_manager.get_sorted_pairs('price')  # Sort pairs by price
            print("\n=== Sorted by Price ===")
        else:
            pairs = pairs_manager.get_sorted_pairs('volume')  # Default sorting by volume
            print("\n=== Sorted by Volume ===")
    
        # Print header for the data table
        print(f"{'No.':<4} {'Symbol':<12} {'Price':<14} {'24h Change':<12} {'24h Volume':<15} {'24h High':<14} {'24h Low':<14}")
        print("-" * 85)
        
        # Display the sorted pairs with their market data
        for i, pair in enumerate(pairs, 1):
            print(
                f"{i:<4} {pair['symbol']:<12} "
                f"${pair['price']:<13,.4f} "
                f"{pair['change_24h']:>+11.2f}% "
                f"${pair['volume']:>14,.2f} "
                f"${pair['high']:<13,.4f} "
                f"${pair['low']:<13,.4f}"
            )
        
        # Show top gainers and losers
        print("\n=== Top Movers ===")
        movers = pairs_manager.get_top_movers(5)  # Get top 5 movers
        
        print("\nTop Gainers:")
        for pair in movers['gainers']:
            print(f"{pair['symbol']}: {pair['change_24h']:+.2f}%")
                
        print("\nTop Losers:")
        for pair in movers['losers']:
            print(f"{pair['symbol']}: {pair['change_24h']:+.2f}%")
                
        # Show pairs with unusual high volume
        print("\n=== Unusual Volume ===")
        high_volume = pairs_manager.get_high_volume_pairs()
        for pair in high_volume[:5]:
            print(f"{pair['symbol']}: ${pair['volume']:,.2f}")
        
        return pairs  # Return the list of pairs for further use
            
    except Exception as e:
        logger.error(f"Error in view_market_data: {e}")
        print(f"Error fetching market data: {e}")
        return []
    
def ai_trade_suggestion_message(signal):
    """Provides custom messages for AI trade suggestions."""
    if signal == "buy":
        return "Sir, I suggest we buy based on the current analysis."
    elif signal == "sell":
        return "Sir, I suggest we sell to maximize potential profit."
    return None  # Return None if signal is 'hold' or unrecognized
    
def ai_chat():
    """Enhanced AI chat with better market context"""
    global current_pair, fetcher
    last_discussed_symbol = current_pair  # Keep track of the last symbol discussed
    
    print("\n=== AI Chat Mode ===")
    print("Ask me anything about the markets. I have access to real-time and historical data.")
    
    while True:
        user_query = input("You: ").strip().lower()
        if user_query == "exit":
            print("Exiting AI chat mode.")
            break
                
        try:
            timeframes = {
                'week': 7 * 24 * 60 * 60 * 1000,  # Time in milliseconds
                'day': 24 * 60 * 60 * 1000,
                'hour': 60 * 60 * 1000,
                'minute': 60 * 1000
            }
                
            # Enhanced pair detection
            pair_mentioned = None
            timeframe_mentioned = None
                
            # Check if any trading pair is mentioned
            available_pairs = pairs_manager.available_pairs
            for pair in available_pairs:
                base_currency = pair.split('/')[0]
                if base_currency.lower() in user_query:
                    pair_mentioned = pair
                    break
                        
            if not pair_mentioned:
                pair_mentioned = last_discussed_symbol  # Use last discussed symbol if none mentioned
                
            # Find mentioned timeframe
            for timeframe in timeframes:
                if timeframe in user_query:
                    timeframe_mentioned = timeframe
                    break
            
            if pair_mentioned:
                pair_info = pairs_manager.get_pair_info(pair_mentioned)
                if pair_info:
                    last_discussed_symbol = pair_mentioned  # Update last discussed symbol
                        
                    print(f"\nCurrent {pair_mentioned} Market Data:")
                    print(f"Price: ${pair_info['price']:,.8f}")
                    print(f"24h Change: {pair_info['change_24h']:+.2f}%")
                    print(f"24h Volume: ${pair_info['volume']:,.2f}")
                    print(f"24h High: ${pair_info['high']:,.8f}")
                    print(f"24h Low: ${pair_info['low']:,.8f}")
                    print(f"Current Spread: ${pair_info['spread']:,.8f}")
                        
                    if timeframe_mentioned:
                        since = int(time.time() * 1000) - timeframes[timeframe_mentioned]
                        ohlcv = fetcher.exchange.fetch_ohlcv(
                            pair_mentioned,
                            '1d' if timeframe_mentioned in ['week'] else '1h',
                            since=since,
                            limit=100
                        )
                        
                            
                        if ohlcv and len(ohlcv) > 0:
                            start_price = ohlcv[0][4]  # Closing price of the first candle
                            price_change = ((pair_info['price'] - start_price) / start_price) * 100
                                
                            print(f"\n{timeframe_mentioned.capitalize()} Performance:")
                            print(f"Starting Price: ${start_price:,.8f}")
                            print(f"Price Change: {price_change:+.2f}%")
                            print(f"Highest Price: ${max(candle[2] for candle in ohlcv):,.8f}")
                            print(f"Lowest Price: ${min(candle[3] for candle in ohlcv):,.8f}")
            
            # Get AI insights with market context
            response = ai_signal_generator.chat(user_query)
            print(f"AI: {response}")
                
        except Exception as e:
            logger.error(f"Error in AI chat: {e}")
            print(f"Error: {str(e)}")
            print("Please try again with a valid trading pair and timeframe.")
    
def change_pair():
    """Enhanced pair selection with detailed validation"""
    global current_pair, ai_signal_generator
        
    try:
        print("\nFetching available trading pairs...")
        pairs = view_market_data()
            
        while True:
            choice = input("\nEnter the number of the pair you want to trade, or type the pair directly (e.g., BTC/USDT), or 'back' to return: ").strip()
                
            if choice.lower() == 'back':
                return
                
            selected_pair = None
                
            if choice.isdigit():
                idx = int(choice) - 1  # Convert to zero-based index
                if 0 <= idx < len(pairs):
                    selected_pair = pairs[idx]['symbol']
            else:
                input_pair = choice.upper()
                if not any(input_pair.endswith(f"/{quote}") for quote in config['pairs_config']['quote_currencies']):
                    input_pair += '/USDT'  # Default to USDT if quote currency not specified
                if pairs_manager.check_pair_validity(input_pair):
                    selected_pair = input_pair
                
            if selected_pair:
                pair_info = pairs_manager.get_pair_info(selected_pair)
                    
                if pair_info:
                    print(f"\n=== {selected_pair} Details ===")
                    print(f"Base Currency: {pair_info['base_currency']}")
                    print(f"Quote Currency: {pair_info['quote_currency']}")
                    print(f"Current Price: ${pair_info['price']:,.8f}")
                    print(f"24h Change: {pair_info['change_24h']:+.2f}%")
                    print(f"24h Volume: ${pair_info['volume']:,.2f}")
                    print(f"Spread: ${pair_info['spread']:,.8f}")
                        
                    if pair_info['min_amount']:
                        print(f"Minimum Order: {pair_info['min_amount']} {pair_info['base_currency']}")
                    if pair_info['max_amount']:
                        print(f"Maximum Order: {pair_info['max_amount']} {pair_info['base_currency']}")
                        
                    confirm = input("\nConfirm selection? (y/n): ").strip().lower()
                    if confirm == 'y':
                        current_pair = selected_pair  # Update current trading pair
                        ai_signal_generator = AIBasedSignal(config['openai_key'], fetcher, current_pair)  # Reinitialize AI signal generator
                        print(f"\nTrading pair changed to {current_pair}")
                        break
                else:
                    print("Error getting pair details. Please try another pair.")
            else:
                print("Invalid pair selection. Please try again.")
                    
    except Exception as e:
        logger.error(f"Error in change_pair: {e}")
        print(f"Error during pair selection: {e}")
    
def trade_bot():
    global running, current_pair, executor
    
    # First check balance before initializing anything
    try:
        balance = executor.exchange.fetch_balance()
        available_balance = float(balance['free'].get('USDT', 0))
        
        if available_balance < config['initial_investment']:
            logger.warning(f"Insufficient starting balance: ${available_balance:.2f} USDT")
            print("\n⚠️ Cannot start trading - Insufficient funds!")
            print(f"Available balance: ${available_balance:.2f} USDT")
            print(f"Minimum required: ${config['initial_investment']:.2f} USDT")
            running = False
            return
            
        logger.info(f"Starting trading with balance: ${available_balance:.2f} USDT")
        print(f"\nInitial balance: ${available_balance:.2f} USDT")
        
    except Exception as e:
        logger.error(f"Error checking balance: {e}")
        print("\n❌ Could not verify balance. Please check your API connection.")
        running = False
        return
    
    # Initialize components only if balance check passes
    strategy = MovingAverageStrategy(
        short_window=config.get('short_window', 5),
        long_window=config.get('long_window', 20)
    )
    executor = TradeExecutor(config)
    signal_model = SignalModel(config)
    
    # Trade tracking initialization
    last_trade_time = None
    min_trade_interval = config.get('timeframe', 60)
    daily_trades = 0
    daily_loss = 0
    initial_balance = None
    max_daily_trades = config['risk_management']['max_daily_trades']
    max_daily_loss = config['risk_management']['max_daily_loss']
    trade_amount = config['trade_amount']
    stop_loss_pct = config['risk_management']['stop_loss_percentage']
    take_profit_pct = config['risk_management']['take_profit_percentage']
    
    # DQN model initialization
    use_dqn = config.get('use_dqn', True)
    if use_dqn:
        state_size = 10
        action_size = 3
        agent = DQNAgent(state_size, action_size)
        agent.load("dqn_trading_model.keras")
        logger.info("Loaded DQN model for trading decisions")

    # Timeframe configuration
    timeframe_seconds = config.get('timeframe', 60)
    interval_map = {
        60: '1m',
        300: '5m',
        900: '15m',
        1800: '30m',
        3600: '1h',
        14400: '4h',
        86400: '1d'
    }
    granularity = interval_map.get(timeframe_seconds, '1m')
    data_limit = config.get('data_limit', 100)
    
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator."""
        prices = pd.Series(prices)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def check_rsi_conditions(data):
        """Check if RSI is in overbought or oversold territory."""
        try:
            rsi_value = calculate_rsi(data['close'].values, period=14)
            if rsi_value >= config['alerts']['rsi_overbought']:
                return 'sell'
            elif rsi_value <= config['alerts']['rsi_oversold']:
                return 'buy'
            return None
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None

    def can_trade():
        """Check if trading is allowed based on all conditions."""
        nonlocal last_trade_time, daily_trades, daily_loss, initial_balance
        current_time = time.time()
        
        if daily_trades >= max_daily_trades:
            logger.info("Daily trade limit reached")
            return False
        
        if initial_balance and daily_loss >= (initial_balance * max_daily_loss):
            logger.info("Daily loss limit reached")
            return False
            
        if last_trade_time and (current_time - last_trade_time) < min_trade_interval:
            return False
            
        if trade_amount < config['trading_rules']['min_order_size']:
            logger.info("Trade amount below minimum order size")
            return False
            
        if trade_amount > config['trading_rules']['max_order_size']:
            logger.info("Trade amount exceeds maximum order size")
            return False
            
        return True
    
    def execute_trade_with_safety(signal, current_price):
        """Execute trade with all safety checks and order management."""
        nonlocal last_trade_time, daily_trades, daily_loss
        
        try:
            # Check balance before trade
            balance = executor.exchange.fetch_balance()
            available_balance = float(balance['free'].get('USDT', 0))
            required_amount = current_price * trade_amount
            
            # Validate sufficient balance
            if available_balance < required_amount:
                logger.warning(f"Insufficient balance for trade: ${available_balance:.2f} USDT available, needed: ${required_amount:.2f} USDT")
                print(f"\n⚠️ Cannot execute {signal} trade - Insufficient funds!")
                print(f"Required: ${required_amount:.2f} USDT")
                print(f"Available: ${available_balance:.2f} USDT")
                return False
            
            # Calculate stop loss and take profit prices
            stop_loss_price = current_price * (1 - stop_loss_pct) if signal == "buy" else current_price * (1 + stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct) if signal == "buy" else current_price * (1 - take_profit_pct)
            
            # Execute main trade
            order = executor.execute_trade(signal)
            
            if order and order.get('status') == 'closed':
                last_trade_time = time.time()
                daily_trades += 1
                
                try:
                    if signal == "buy":
                        # Verify balance again after main trade
                        post_trade_balance = executor.exchange.fetch_balance()
                        available_amount = float(post_trade_balance['free'].get(current_pair.split('/')[0], 0))
                        
                        if available_amount < trade_amount:
                            logger.error("Insufficient balance for stop loss/take profit orders")
                            print("\n⚠️ Warning: Could not place stop loss/take profit orders - insufficient balance")
                            return True  # Main trade succeeded but couldn't place stop loss/take profit
                        
                        # Place stop loss order
                        executor.exchange.create_order(
                            symbol=current_pair,
                            type='stop_loss_limit',
                            side='sell',
                            amount=trade_amount,
                            price=stop_loss_price,
                            params={'stopPrice': stop_loss_price}
                        )
                        
                        # Place take profit order
                        executor.exchange.create_order(
                            symbol=current_pair,
                            type='limit',
                            side='sell',
                            amount=trade_amount,
                            price=take_profit_price
                        )
                        
                        logger.info("Successfully placed stop loss and take profit orders")
                        
                except Exception as e:
                    logger.error(f"Error placing stop loss/take profit orders: {e}")
                    print("\n⚠️ Warning: Error placing stop loss/take profit orders")
                    print("Please monitor position manually!")
                
                # Log trade details
                logger.info(f"Trade executed: {signal.upper()} {current_pair}")
                logger.info(f"Order details: Price=${order.get('price', 0):.4f}, "
                        f"Amount={order.get('amount', 0):.8f}")
                logger.info(f"Stop Loss: ${stop_loss_price:.4f}")
                logger.info(f"Take Profit: ${take_profit_price:.4f}")
                
                # Print trade confirmation
                print(f"\n✅ Trade executed successfully!")
                print(f"Type: {signal.upper()}")
                print(f"Price: ${order.get('price', 0):.4f}")
                print(f"Amount: {order.get('amount', 0):.8f}")
                print(f"Value: ${order.get('cost', 0):.2f}")
                print(f"Stop Loss: ${stop_loss_price:.4f}")
                print(f"Take Profit: ${take_profit_price:.4f}")
                
                # Update balance after trade
                final_balance = executor.exchange.fetch_balance()
                new_balance = float(final_balance['free'].get('USDT', 0))
                print(f"Remaining balance: ${new_balance:.2f} USDT")
                
                return True
                
            return False
            
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds: {e}")
            print(f"\n❌ Insufficient funds to execute {signal} trade!")
            return False
            
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            print(f"\n❌ Exchange error: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            print(f"\n❌ Error executing trade: {e}")
            return False

    def check_signal_model_opportunity(data):
        """Check for high-probability trading opportunities using SignalModel."""
        try:
            # Get signal and confidence from SignalModel
            signal, confidence = signal_model.generate_signal(data)
            
            if signal != 'hold' and confidence >= 0.8:  # Higher threshold for signal model
                # Log detailed analysis
                signal_model.log_trade_opportunity(data, signal, confidence)
                
                # Calculate risk/reward ratio
                current_price = float(data['close'].iloc[-1])
                if signal == 'buy':
                    risk = current_price * stop_loss_pct
                    reward = current_price * take_profit_pct
                else:
                    risk = current_price * take_profit_pct
                    reward = current_price * stop_loss_pct
                
                risk_reward_ratio = reward / risk
                
                # Only return signal if risk/reward is favorable
                if risk_reward_ratio >= 1.5:
                    return signal, confidence
                
            return None, 0
            
        except Exception as e:
            logger.error(f"Error checking signal model opportunity: {e}")
            return None, 0
        
     # Initialize balance tracking
    try:
        balance = executor.exchange.fetch_balance()
        initial_balance = float(balance['total'].get('USDT', 0))
        logger.info(f"Initial balance: ${initial_balance:.2f} USDT")
    except Exception as e:
        logger.error(f"Error fetching initial balance: {e}")
        initial_balance = float(config['initial_investment'])
    
    # Main trading loop
    while running:
        try:
            # Reset daily counters at market open
            if datetime.now().hour == 0 and datetime.now().minute == 0:
                daily_trades = 0
                daily_loss = 0
            
            symbol = current_pair
            data = fetcher.get_latest_data(symbol, granularity=granularity, data_limit=data_limit)
            if data is None or data.empty:
                time.sleep(timeframe_seconds)
                continue

            # First check SignalModel for high-probability trades
            signal_model_signal, signal_model_confidence = check_signal_model_opportunity(data)
            
            if signal_model_signal and can_trade():
                print(f"\nHigh-probability trade detected by SignalModel:")
                print(f"Signal: {signal_model_signal.upper()}")
                print(f"Confidence: {signal_model_confidence * 100:.1f}%")
                
                current_price = float(data['close'].iloc[-1])
                if execute_trade_with_safety(signal_model_signal, current_price):
                    print("High-probability trade executed successfully!")
                    continue  # Skip regular signal processing

            # Get all trading signals
            ai_signal = ai_signal_generator.get_ai_signal(data) if config['use_ai_signals'] else None
            dqn_signal = agent.act(data['close'][-10:].values.reshape(1, -1)) if use_dqn else None
            dqn_signal = {0: "hold", 1: "buy", 2: "sell"}.get(dqn_signal, "hold")
            
            ma_signals = strategy.generate_signals(data)
            ma_final_signal = "buy" if ma_signals['pre_growth'].iloc[-1] == 1 else "sell" if ma_signals['pre_dump'].iloc[-1] == 1 else "hold"
            
            rsi_signal = check_rsi_conditions(data)

            # Combine all signals
            signals = [s for s in [ai_signal, dqn_signal, ma_final_signal, rsi_signal] if s is not None]
            
            # Determine final signal
            if not signals:
                final_signal = "hold"
            elif ai_signal == dqn_signal and ai_signal is not None:
                final_signal = ai_signal
            elif signals.count("buy") >= 2:
                final_signal = "buy"
            elif signals.count("sell") >= 2:
                final_signal = "sell"
            else:
                final_signal = "hold"

            # Display AI suggestion
            suggestion_message = ai_trade_suggestion_message(final_signal)
            if suggestion_message:
                print(suggestion_message)

            # Get AI confidence and current price
            ai_confidence = ai_signal_generator.get_confidence(data)
            print(f"AI Confidence: {ai_confidence * 100:.1f}%")
            current_price = float(data['close'].iloc[-1])
            
            # Execute trade if conditions are met
            if final_signal in ["buy", "sell"] and ai_confidence >= confidence_threshold:
                if can_trade():
                    logger.info(f"High confidence signal: {final_signal} ({ai_confidence:.2f})")
                    print(f"Executing {final_signal.upper()} trade with {ai_confidence * 100:.1f}% confidence")
                    print(f"Current price: ${current_price:.4f}")
                    
                    if execute_trade_with_safety(final_signal, current_price):
                        print("Trade executed successfully!")
                    else:
                        print("Trade execution failed. See logs for details.")
                else:
                    print("Trade skipped: Trading limits or restrictions in effect")
            else:
                if final_signal != "hold":
                    print(f"Trade suggestion only: {final_signal} (Confidence too low or signal unclear)")

            time.sleep(timeframe_seconds)
                
        except Exception as e:
            logger.error(f"Error in trade_bot: {e}")
            print(f"Error in trading loop: {e}")
            time.sleep(timeframe_seconds)   
    
def start_bot():
    global running, executor, daily_trades, daily_loss, config
    try:
        # Initialize executor if not already done
        if 'executor' not in globals() or executor is None:
            executor = TradeExecutor(config)
            logger.info("Trade executor initialized")

        # Check balance before starting
        balance_status = check_balance()
        if balance_status is None:
            print("\n❌ Error checking balance. Please verify your API connection.")
            return False

        available_balance = balance_status['available']
        print(f"\n💰 Current Balance: ${available_balance:.2f} USDT")

        # Prompt for trade amount
        while True:
            try:
                print("\n=== Set Trade Amount ===")
                print(f"Available Balance: ${available_balance:.2f} USDT")
                print("Minimum trade: $10.00 USDT")
                print(f"Maximum trade: ${min(1000, available_balance):.2f} USDT")
                
                amount = float(input("\nEnter trade amount in USDT (min $10): $"))
                
                if amount < 10:
                    print("❌ Amount too small. Minimum is $10 USDT")
                    continue
                    
                if amount > min(1000, available_balance):
                    print(f"❌ Amount too large. Maximum is ${min(1000, available_balance):.2f} USDT")
                    continue
                
                # Update config with new trade amount
                ticker = executor.exchange.fetch_ticker(config['trade_pair'])
                current_price = float(ticker['last'])
                
                # Calculate quantity in base currency (e.g., BTC)
                quantity = amount / current_price
                
                # Update config
                config['initial_investment'] = amount
                config['trade_amount'] = quantity
                
                print(f"\n✅ Trade settings:")
                print(f"Amount per trade: ${amount:.2f} USDT")
                print(f"Quantity per trade: {quantity:.8f} {config['trade_pair'].split('/')[0]}")
                
                confirm = input("\nConfirm these settings? (y/n): ").lower()
                if confirm == 'y':
                    break
                
            except ValueError:
                print("❌ Please enter a valid number")

        if not running:
            # Reset daily counters
            daily_trades = 0
            daily_loss = 0
            
            # Start the bot
            running = True
            bot_thread = threading.Thread(target=trade_bot)
            bot_thread.daemon = True
            bot_thread.start()
            
            # Log and display success message
            logger.info(f"Trading bot started with ${amount:.2f} USDT per trade")
            print(f"\n✅ Trading bot started successfully!")
            print(f"Trading amount: ${amount:.2f} USDT per trade")
            print(f"Available balance: ${available_balance:.2f} USDT")
            print(f"Daily trade limit: {config['risk_management']['max_daily_trades']} trades")
            print(f"Stop loss: {config['risk_management']['stop_loss_percentage'] * 100}%")
            print(f"Take profit: {config['risk_management']['take_profit_percentage'] * 100}%")
            return True
        else:
            print("\n⚠️ Trading bot is already running.")
            return True
            
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        print(f"\n❌ Error: Unable to start trading bot. Please check your API keys and connection.")
        return False
    
def stop_bot():
    global running
    if running:
        running = False
        logger.info("Trading bot stopping")
        print("Stopping the trading bot...")
    else:
        print("Trading bot is not currently running.")
    
def export_data():
    """Enhanced data export functionality"""
    print("\n=== Export Market Data ===")
    print("1. Export current market overview")
    print("2. Export historical data for a symbol")
    print("3. Export all pairs data")
    print("4. Back to main menu")
        
    choice = input("Enter your choice: ").strip()
        
    try:
        if choice == "1":
            data_exporter.export_current_market_data()
        elif choice == "2":
            symbol = input("Enter symbol (e.g., BTC/USDT): ").strip().upper()
            if pairs_manager.check_pair_validity(symbol):
                timeframe = input("Enter timeframe (1m/5m/15m/1h/4h/1d): ").strip()
                limit = int(input("Enter number of candles (max 1000): ").strip())
                data_exporter.export_historical_data(symbol, timeframe, limit)
            else:
                print("Invalid trading pair.")
        elif choice == "3":
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'market_data/all_pairs_{timestamp}.csv'
            pairs_manager.export_pairs_data(filename)
            print(f"All pairs data exported to {filename}")
    except Exception as e:
        logger.error(f"Error in export_data: {e}")
        print(f"Error exporting data: {e}")
    
def view_market_analysis():
    """Enhanced market analysis with alerts and insights"""
    print("\n=== Market Analysis ===")
        
    try:
        # Update available pairs
        pairs_manager.update_available_pairs()
            
        # Get market summary
        summary = market_analyzer.get_market_summary()
        alerts = market_analyzer.get_latest_alerts()
            
        # Get additional analysis from pairs_manager
        movers = pairs_manager.get_top_movers(5)
        high_volume = pairs_manager.get_high_volume_pairs()
            
        print("\n=== Price Movement Analysis ===")
        print("\nTop Gainers:")
        for coin in movers['gainers']:
            print(f"{coin['symbol']}: {coin['change_24h']:+.2f}%")
                
        print("\nTop Losers:")
        for coin in movers['losers']:
            print(f"{coin['symbol']}: {coin['change_24h']:+.2f}%")
                
        print("\n=== Volume Analysis ===")
        print("\nHigh Volume Pairs:")
        for pair in high_volume[:5]:
            print(f"{pair['symbol']}: ${pair['volume']:,.2f}")
                
        print("\n=== Recent Market Alerts ===")
        for alert in alerts:
            print(f"{alert['timestamp']} - {alert['symbol']}: {alert['alert']}")
            
        print("\n=== Trading Opportunities ===")
        sorted_gainers = sorted(movers['gainers'], key=lambda x: x.get('volume') or 0, reverse=True)
        for coin in sorted_gainers[:3]:
            coin_volume = coin.get('volume', 0)
            min_volume = config['pairs_config'].get('min_volume', 10000)
            if coin_volume > min_volume:
                print(f"Strong Momentum: {coin['symbol']} (+{coin['change_24h']:.2f}% / ${coin_volume:,.2f})")
                    
    except Exception as e:
        logger.error(f"Error in view_market_analysis: {e}")
        print(f"Error performing market analysis: {e}")


def check_current_balance():
    """Display detailed balance information."""
    try:
        balance = executor.exchange.fetch_balance()
        available_usdt = float(balance['free'].get('USDT', 0))
        total_usdt = float(balance['total'].get('USDT', 0))
        
        print("\n=== Current Balance Information ===")
        print(f"Available USDT: ${available_usdt:.2f}")
        print(f"Total USDT: ${total_usdt:.2f}")
        
        # Show current positions if any
        current_pair_base = current_pair.split('/')[0]
        if current_pair_base in balance['free']:
            asset_balance = float(balance['free'][current_pair_base])
            asset_total = float(balance['total'][current_pair_base])
            current_price = float(fetcher.get_latest_data(current_pair, granularity='1m', data_limit=1)['close'].iloc[-1])
            
            print(f"\nCurrent {current_pair_base} Holdings:")
            print(f"Available: {asset_balance:.8f} {current_pair_base}")
            print(f"Total: {asset_total:.8f} {current_pair_base}")
            print(f"Estimated Value: ${(asset_total * current_price):.2f} USDT")
        
        # Show daily trading stats
        print("\nTrading Limits:")
        print(f"Daily Trades: {daily_trades}/{config['risk_management']['max_daily_trades']}")
        print(f"Daily Loss: ${daily_loss:.2f} USDT (Max: ${total_usdt * config['risk_management']['max_daily_loss']:.2f} USDT)")
        
        return available_usdt
        
    except Exception as e:
        logger.error(f"Error checking balance: {e}")
        print(f"Error: Unable to fetch balance information - {e}")
        return None  

import logging

# Set up logging to file and console
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ai_trade_pilot():
    """AI Trade Pilot for profit-focused trading with comprehensive analysis and feedback."""
    global running, current_pair, executor
    pilot_running = True
    last_trade_time = None
    min_trade_interval = 300  # 5 minutes between trades
    scan_count = 0
    analyzed_pairs = set()
    total_opportunities_found = 0
    best_score_today = 0

    def analyze_pair(symbol, data, pair_data):
        """Detailed analysis of a trading pair."""
        try:
            logger.info(f"Starting analysis for {symbol}")
            # Initialize analysis structure
            analysis = {
                'symbol': symbol,
                'price': pair_data['price'],
                'volume': pair_data['volume'],
                'change_24h': pair_data['change_24h'],
                'signals': {},
                'indicators': {},
                'risk_metrics': {},
                'score': 0
            }
            
            # Get all signals
            signal_model_signal, signal_confidence = signal_model.generate_signal(data)
            ai_signal = ai_signal_generator.get_ai_signal(data)
            ai_confidence = ai_signal_generator.get_confidence(data)
            
            # Store signals in analysis
            analysis['signals'] = {
                'signal_model': {'signal': signal_model_signal, 'confidence': signal_confidence},
                'ai': {'signal': ai_signal, 'confidence': ai_confidence}
            }
            logger.info(f"{symbol} - Signals: Signal Model={signal_model_signal} (Confidence={signal_confidence}), AI Signal={ai_signal} (Confidence={ai_confidence})")
            
            # Calculate technical indicators
            closes = data['close'].values
            volumes = data['volume'].values
            
            # RSI Analysis
            rsi = talib.RSI(closes)[-1]
            
            # MACD
            macd, signal, hist = talib.MACD(closes)
            macd_signal = 'buy' if macd[-1] > signal[-1] else 'sell'
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(closes)
            bb_position = (closes[-1] - lower[-1]) / (upper[-1] - lower[-1])
            
            # Volume Analysis
            volume_sma = np.mean(volumes)
            volume_increase = volumes[-1] > volume_sma * 1.5
            
            # Trend Analysis
            sma_20 = talib.SMA(closes, timeperiod=20)[-1]
            sma_50 = talib.SMA(closes, timeperiod=50)[-1]
            trend = 'uptrend' if sma_20 > sma_50 else 'downtrend'
            
            # Additional trend strength indicators
            recent_trend = (closes[-1] - closes[-5]) / closes[-5] * 100  # 5-period trend

            analysis['indicators'] = {
                'rsi': rsi,
                'macd_signal': macd_signal,
                'bb_position': bb_position,
                'volume_strength': volumes[-1] / volume_sma,
                'trend': trend,
                'trend_strength': abs(recent_trend),
                'above_sma20': closes[-1] > sma_20,
                'above_sma50': closes[-1] > sma_50
            }
            
            # Risk Metrics
            volatility = np.std(np.diff(closes) / closes[:-1]) * 100
            avg_volume = np.mean(volumes)
            price_momentum = (closes[-1] - closes[-10]) / closes[-10] * 100
            
            # Calculate support and resistance
            recent_high = np.max(closes[-20:])
            recent_low = np.min(closes[-20:])
            
            analysis['risk_metrics'] = {
                'volatility': volatility,
                'liquidity': pair_data['volume'] / avg_volume,
                'momentum': price_momentum,
                'distance_to_high': (recent_high - closes[-1]) / closes[-1] * 100,
                'distance_to_low': (closes[-1] - recent_low) / closes[-1] * 100
            }
            
            # Calculate overall score (0-100)
            signal_score = (signal_confidence + ai_confidence) * 50  # 0-100
            
            technical_score = (
                (50 - abs(rsi - 50)) / 50 * 25 +  # RSI closer to 50 is better
                (bb_position if trend == 'uptrend' else (1 - bb_position)) * 25 +
                (volume_increase * 25) +
                (25 if macd_signal == signal_model_signal else 0)
            )
            
            trend_score = (
                (analysis['indicators']['trend_strength'] * 0.5) +  # Trend strength
                (20 if trend == 'uptrend' else 0) +  # Uptrend bonus
                (15 if analysis['indicators']['above_sma20'] else 0) +  # Above SMA20
                (15 if analysis['indicators']['above_sma50'] else 0)  # Above SMA50
            )
            
            # Calculate the final weighted score
            analysis['score'] = (signal_score * 0.4 + technical_score * 0.4 + trend_score * 0.2)  # Weighted average
            logger.info(f"Completed analysis for {symbol} with score {analysis['score']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def print_analysis(analysis):
        """Print detailed analysis in a clear format."""
        print(f"\n{'='*50}")
        print(f"📊 Analysis for {analysis['symbol']}")
        print(f"{'='*50}")
        
        print(f"\n💰 Price: ${analysis['price']:.8f}")
        print(f"📈 24h Change: {analysis['change_24h']:+.2f}%")
        print(f"📊 Volume: ${analysis['volume']:,.2f}")
        
        print("\n🎯 Trading Signals:")
        print(f"Signal Model: {analysis['signals']['signal_model']['signal'].upper()} "
              f"({analysis['signals']['signal_model']['confidence']*100:.1f}%)")
        print(f"AI Signal: {analysis['signals']['ai']['signal'].upper()} "
              f"({analysis['signals']['ai']['confidence']*100:.1f}%)")
        
        print("\n📈 Technical Indicators:")
        print(f"RSI: {analysis['indicators']['rsi']:.1f}")
        print(f"MACD Signal: {analysis['indicators']['macd_signal'].upper()}")
        print(f"BB Position: {analysis['indicators']['bb_position']:.2f}")
        print(f"Volume Strength: {analysis['indicators']['volume_strength']:.1f}x average")
        print(f"Trend: {analysis['indicators']['trend'].upper()}")
        
        print("\n⚠️ Risk Analysis:")
        print(f"Volatility: {analysis['risk_metrics']['volatility']:.2f}%")
        print(f"Liquidity: {analysis['risk_metrics']['liquidity']:.1f}x average")
        print(f"Momentum: {analysis['risk_metrics']['momentum']:+.2f}%")
        print(f"Distance to High: {analysis['risk_metrics']['distance_to_high']:.2f}%")
        print(f"Distance to Low: {analysis['risk_metrics']['distance_to_low']:.2f}%")
        
        print(f"\n📈 Overall Score: {analysis['score']:.1f}/100")
        
        # Trading recommendation
        if analysis['score'] >= 75:
            print("\n🟢 Strong Buy Opportunity")
        elif analysis['score'] >= 65:
            print("\n🟡 Potential Opportunity - Use Caution")
        else:
            print("\n🔴 Not Recommended")

    def check_trade_safety(analysis):
        """Verify if trade meets safety criteria."""
        try:
            # Check balance
            balance_status = check_balance()
            if not balance_status or not balance_status['can_trade']:
                print("\n⚠️ Insufficient balance for trading!")
                return False
                
            # Check minimum score
            if analysis['score'] < 65:
                print("\n⚠️ Score too low for safe trading")
                return False
                
            # Check signal agreement
            if (analysis['signals']['signal_model']['signal'] != 
                analysis['signals']['ai']['signal']):
                print("\n⚠️ Conflicting signals detected")
                return False
                
            # Check trading interval
            if (last_trade_time and 
                time.time() - last_trade_time < min_trade_interval):
                print("\n⚠️ Minimum trade interval not met")
                return False
                
            # Check risk metrics
            if analysis['risk_metrics']['volatility'] > 5:  # 5% volatility max
                print("\n⚠️ Volatility too high")
                return False
                
            if analysis['risk_metrics']['liquidity'] < 1.2:  # 20% above avg volume
                print("\n⚠️ Insufficient liquidity")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            return False

    print("\n=== AI Trade Pilot Active ===")
    logger.info("AI Trade Pilot started")

    while pilot_running:
        try:
            scan_count += 1
            print(f"\n🔄 Scan #{scan_count} Started")
            logger.info(f"Starting Scan #{scan_count}")
            
            pairs = pairs_manager.get_sorted_pairs('volume')
            
            for pair_data in pairs[:10]:
                try:
                    symbol = pair_data['symbol']
                    print(f"\n🔍 Analyzing {symbol}...")
                    logger.info(f"Fetching data for {symbol}")
                    data = fetcher.get_latest_data(symbol, granularity='1m', data_limit=50)
                    
                    if data is None:
                        print(f"⚠️ Skipping {symbol} - No data available")
                        logger.warning(f"No data available for {symbol}, skipping.")
                        continue
                    
                    analysis = analyze_pair(symbol, data, pair_data)
                    if analysis:
                        print_analysis(analysis)
                        if analysis['score'] >= 65 and check_trade_safety(analysis):
                            signal = analysis['signals']['signal_model']['signal']
                            price = analysis['price']
                            logger.info(f"Executing trade for {symbol}: Signal={signal}, Price=${price}")
                            execute_trade_with_safety(signal, price)
                        else:
                            print("\n⚠️ Trade skipped due to safety checks or low score.")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            logger.info(f"Scan #{scan_count} completed.")
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\n⏹️ Stopping AI Trade Pilot...")
            logger.info("AI Trade Pilot stopped by user.")
            pilot_running = False
            break
            
        except Exception as e:
            logger.error(f"Error in AI Trade Pilot loop: {e}")
            print(f"\n❌ Error: {e}")
            time.sleep(60)

def test_direct_trade():
    """Direct trade test with minimal amount."""
    try:
        # Get current balance
        balance = executor.exchange.fetch_balance()
        initial_usdt = float(balance['free'].get('USDT', 0))
        print(f"\n💰 Initial USDT: ${initial_usdt:.2f}")

        # Get current market price
        symbol = config['trade_pair']
        ticker = executor.exchange.fetch_ticker(symbol)
        price = float(ticker['last'])
        print(f"Current {symbol} price: ${price:.8f}")

        # Calculate minimum trade amount (use a very small amount for testing)
        trade_amount = 0.001  # Minimum trade amount
        cost_usdt = trade_amount * price
        print(f"Attempting to buy {trade_amount} {symbol.split('/')[0]} (${cost_usdt:.2f} USDT)")

        if cost_usdt > initial_usdt:
            print("❌ Insufficient funds for test trade")
            return

        # Place market buy order directly
        try:
            print("\n🚀 Placing market buy order...")
            order = executor.exchange.create_market_buy_order(
                symbol,
                trade_amount,
                {
                    'test': False,  # Set to True for test mode
                    'type': 'market'
                }
            )
            
            print("\nOrder response:")
            print(json.dumps(order, indent=2))

            # Check if order was successful
            if order and order['status'] == 'closed':
                print("\n✅ Order successful!")
                print(f"Amount: {order['filled']} {symbol.split('/')[0]}")
                print(f"Price: ${order['price']:.8f}")
                print(f"Cost: ${order['cost']:.2f} USDT")

                # Verify balance change
                new_balance = executor.exchange.fetch_balance()
                final_usdt = float(new_balance['free'].get('USDT', 0))
                print(f"\nFinal USDT: ${final_usdt:.2f}")
                print(f"USDT Change: ${final_usdt - initial_usdt:.2f}")
            else:
                print("\n❌ Order not completed")

        except ccxt.InsufficientFunds:
            print("\n❌ Insufficient funds to execute trade")
        except Exception as e:
            print(f"\n❌ Error executing trade: {str(e)}")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")




## Menu
def menu():
    """Main menu for the trading bot."""
    try:
        # Start background analysis
        market_analyzer.start_analysis()
        logger.info("Background market analysis started")
        print("\n✅ Market analysis started successfully")
    except Exception as e:
        logger.error(f"Failed to start market analysis: {e}")
        print("\n⚠️ Warning: Background market analysis failed to start")

    while True:
        print("\n" + "="*40)
        print("=== Trading Bot Menu by Victor Udeh ===")
        print("="*40)
        print("1. 🚀 Start Trading Bot")
        print("2. ⏹️ Stop Trading Bot")
        print("3. 🔄 Change Trading Pair")
        print("4. 📊 View Market Data")
        print("5. 💬 AI Chat (Ask for insights)")
        print("6. 📥 Export Data")
        print("7. 📈 View Market Analysis")
        print("8. 💰 Check Balance")
        print("9. 🤖 Start AI Trade Pilot")
        print("10. ❌ Exit")
        print("11. Test Direct Trade")
        print("-"*40)
        
        choice = input("Enter your choice: ").strip()
        
        try:
            if choice == "1":
                # Start bot with balance check
                balance_status = check_balance()
                if balance_status and balance_status['sufficient']:
                    start_bot()
                else:
                    print("\n⚠️ Cannot start bot - insufficient funds!")
                    print(f"Available: ${balance_status['available']:.2f} USDT")
                    print(f"Required: ${MIN_REQUIRED_BALANCE:.2f} USDT")
                    
            elif choice == "2":
                if running:
                    stop_bot()
                    print("\n✅ Trading bot stopped successfully")
                else:
                    print("\n⚠️ Trading bot is not running")
                    
            elif choice == "3":
                change_pair()
                
            elif choice == "4":
                print("\n📊 Loading market data...")
                view_market_data()
                
            elif choice == "5":
                print("\n💬 Starting AI Chat...")
                ai_chat()
                
            elif choice == "6":
                print("\n📥 Export Data Options")
                export_data()
                
            elif choice == "7":
                print("\n📈 Loading market analysis...")
                view_market_analysis()
                
            elif choice == "8":
                print("\n💰 Checking balance...")
                check_current_balance()
                
            elif choice == "9":
                print("\n🤖 Starting AI Trade Pilot...")
                balance_status = check_balance()
                
                if not balance_status or not balance_status['can_trade']:
                    print("\n⚠️ Cannot start AI Trade Pilot - insufficient funds!")
                    print(f"Available: ${balance_status['available']:.2f} USDT")
                    continue
                    
                print("\n✅ Balance check passed")
                print("Starting market analysis...")
                
                try:
                    pilot_thread = threading.Thread(target=ai_trade_pilot)
                    pilot_thread.daemon = True
                    pilot_thread.start()
                    
                    print("\n🤖 AI Trade Pilot is now running")
                    print("Press Enter to return to main menu")
                    input()
                    
                except Exception as e:
                    logger.error(f"Error starting AI Trade Pilot: {e}")
                    print(f"\n❌ Error starting AI Trade Pilot: {e}")
                
            elif choice == "10":
                if running:
                    print("\n⏹️ Stopping trading bot...")
                    stop_bot()
                
                print("\n🛑 Stopping market analysis...")
                market_analyzer.stop_analysis()
                logger.info("Trading bot shutting down")
                
                print("\n✅ Trading bot shutdown complete")
                print("Thank you for using the trading bot!")
                break

            elif choice == "11":
                print("\n=== Testing Direct Trade ===")
                test_direct_trade()  

            else:
                print("\n❌ Invalid choice. Please enter a number between 1 and 11.")
                
        except Exception as e:
            logger.error(f"Error in menu selection: {e}")
            print(f"\n❌ Error processing selection: {e}")
            print("Please try again.")

if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("Welcome to the AI Trading Bot by Victor Udeh")
        print("="*50)
        
        logger.info("Starting trading bot application...")
        menu()
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
        
    finally:
        print("\nThank you for using the trading bot!")
        logger.info("Trading bot application terminated")