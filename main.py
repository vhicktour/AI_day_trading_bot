import sys
import os
import json
import time
import threading
import pandas as pd
import logging
from datetime import datetime
import ccxt

# Set up logging configuration first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration from JSON file
try:
    with open('config/config.json') as f:
        config = json.load(f)
    logger.info("Successfully loaded config")
    
    # Define constants after config is loaded
    MIN_REQUIRED_BALANCE = float(config.get('initial_investment', 50))
    MIN_TRADE_BALANCE = float(config['trading_rules']['min_order_size'])
    
except Exception as e:
    logger.error(f"Error loading config: {str(e)}")
    sys.exit(1)

# Import custom modules after config is loaded
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

def check_balance():
    """Check current balance and verify if it meets minimum requirements."""
    try:
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

# Initialize global variables and components
current_pair = config['trade_pair']
running = False
confidence_threshold = 0.7

try:
    # Initialize all components needed for the trading bot
    fetcher = DataFetcher(config)
    executor = TradeExecutor(config)
    ai_signal_generator = AIBasedSignal(config['openai_key'], fetcher, current_pair)
    market_analyzer = MarketAnalyzer(fetcher)
    data_exporter = MarketDataExporter(fetcher)
    pairs_manager = PairsManager(fetcher, config)
    trading_chat = TradingChat(ai_signal_generator, fetcher, pairs_manager)
    logger.info("Successfully initialized all components")
    
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    sys.exit(1) # Exit the program if configuration fails to load


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
    global running, current_pair
    
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
    global running
    try:
        # Check balance before starting
        balance_status = check_balance()
        if balance_status is None:
            print("\n❌ Error checking balance. Please verify your API connection.")
            return False
        
        if not balance_status['sufficient']:
            print("\n⚠️ Insufficient funds to start trading!")
            print(f"Available balance: ${balance_status['available']:.2f} USDT")
            print(f"Minimum required: ${MIN_REQUIRED_BALANCE:.2f} USDT")
            print("\nPlease deposit funds before starting the trading bot.")
            return False
        
        if not running:
            running = True
            bot_thread = threading.Thread(target=trade_bot)
            bot_thread.daemon = True
            bot_thread.start()
            logger.info(f"Trading bot started with ${balance_status['available']:.2f} USDT")
            print(f"\nTrading bot started with {balance_status['available']:.2f} USDT")
        else:
            print("Trading bot is already running.")
            
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
    
def menu():
    # Start background analysis when program starts
    try:
        market_analyzer.start_analysis()
        logger.info("Background market analysis started")
    except Exception as e:
        logger.error(f"Failed to start market analysis: {e}")
        print("Warning: Background market analysis failed to start")
    
    while True:
        print("\n=== Trading Bot Menu by Victor Udeh ===")
        print("1. Start Trading Bot")
        print("2. Stop Trading Bot")
        print("3. Change Trading Pair")
        print("4. View Market Data")
        print("5. AI Chat (Ask for insights)")
        print("6. Export Data")
        print("7. View Market Analysis")
        print("8. Exit")
            
        choice = input("Enter your choice: ").strip()
            
        try:
            if choice == "1":
                start_bot()
            elif choice == "2":
                stop_bot()
            elif choice == "3":
                change_pair()
            elif choice == "4":
                view_market_data()
            elif choice == "5":
                ai_chat()
            elif choice == "6":
                export_data()
            elif choice == "7":
                view_market_analysis()
            elif choice == "8":
                if running:
                    stop_bot()
                market_analyzer.stop_analysis()
                logger.info("Trading bot shutting down")
                print("Exiting the program.")
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 8.")
        except Exception as e:
            logger.error(f"Error in menu selection: {e}")
            print(f"Error processing selection: {e}")
            print("Please try again.")
    
if __name__ == "__main__":
    try:
        logger.info("Starting trading bot application...")
        menu()
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        sys.exit(1)
    finally:
        logger.info("Trading bot application terminated")
