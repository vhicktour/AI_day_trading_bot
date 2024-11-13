# AI Day Trading Bot

An advanced cryptocurrency trading bot that combines artificial intelligence, deep learning, and traditional trading strategies for automated cryptocurrency trading. Built with Python, it leverages OpenAI's GPT models for market analysis and multiple trading strategies for optimal decision making.

## Features

### 🤖 AI-Powered Trading
- Integration with OpenAI's GPT models for market analysis
- Real-time market sentiment analysis
- Dynamic confidence-based trade execution
- Natural language interaction for market insights
- Enhanced AI chat functionality for trading insights
- Comprehensive market data analysis
- Automated decision-making system

### 📊 Advanced Market Analysis
- Real-time market data monitoring
- Multi-timeframe analysis
- Volume profile analysis
- Price action patterns recognition
- Top movers and market momentum tracking
- Dynamic support and resistance detection
- Advanced technical indicators integration
- Profit potential calculation
- Real-time price pattern detection
- Market structure analysis
- Volume-weighted analysis
- Trend strength indicators

### 📈 Multiple Trading Strategies
- Moving Average Crossover
- Deep Q-Learning Network (DQN)
- AI signal generation
- Strategy aggregation for improved accuracy
- New SignalModel with profit-focused parameters
- Pattern recognition (engulfing, morning/evening star)
- Volatility-based trading signals
- Momentum-based strategies
- Breakout detection
- Mean reversion tactics
- Multi-timeframe confirmation

### 🔄 Dynamic Pair Management
- Support for all major cryptocurrency pairs
- Real-time market scanning
- Volume-based filtering
- Automated pair selection based on market conditions
- Enhanced market opportunity detection
- Dynamic pair rotation
- Liquidity analysis
- Correlation monitoring
- Risk-adjusted pair selection

### 📉 Risk Management
- Configurable position sizing
- Automated stop-loss and take-profit orders
- Maximum daily loss limits
- Trading volume restrictions
- Risk/reward ratio analysis
- Dynamic position sizing based on confidence
- Enhanced safety checks for trade execution
- Drawdown protection
- Exposure limits
- Volatility-adjusted position sizing

### 📊 Data Export & Analysis
- CSV export functionality
- Historical data analysis
- Performance metrics tracking
- Market data archiving
- Detailed trade opportunity logging
- Technical analysis reporting
- Custom report generation
- Performance visualization
- Trade history analysis

## New Features in Version 1.1

### 🎯 Profit-Focused Trading
- Minimum profit threshold configuration (0.5%)
- Volume-based breakout detection (1.5x average volume)
- Trend strength analysis using ADX (25 threshold)
- Volatility-based opportunity detection (2% threshold)
- Dynamic support/resistance levels
- Advanced pattern recognition
- Risk/reward optimization

### 📈 Enhanced Technical Analysis
- Bollinger Bands integration
- MACD signal generation (12/26/9)
- RSI with configurable levels (75/25)
- Volume profile analysis
- Support and resistance detection
- Price pattern recognition
- Enhanced momentum indicators
- Advanced trend analysis

### 🤖 Improved AI Integration
- Enhanced signal generation
- Better confidence calculation
- Detailed trade opportunity analysis
- Real-time market insights
- Pattern-based predictions
- Multi-factor analysis
- Automated decision verification

### ⚡ Performance Improvements
- Faster signal processing
- Better error handling
- More reliable trade execution
- Enhanced safety checks
- Improved response times
- Optimized calculations
- Reduced false signals

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vhicktour/AI_day_trading_bot.git
cd AI_day_trading_bot
```

2. Create and activate a virtual environment:
```bash
# For Windows:
python -m venv venv
venv\Scripts\activate

# For MacOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install required dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow ccxt openai talib-binary
```

5. Configure your API keys:
- Copy `config/config.example.json` to `config/config.json`
- Add your Binance API and OpenAI API keys

## Configuration

The bot is configured through `config/config.json`. Here's a complete configuration example:

```json
{
    "api_key": "your_binance_api_key_here",
    "api_secret": "your_binance_api_secret_here",
    "openai_key": "your_openai_api_key_here",
    "initial_investment": 50,
    "trade_pair": "BTC/USDT",
    "trade_amount": 0.001,
    "use_ai_signals": true,
    "use_dqn": true,
    "short_window": 5,
    "long_window": 20,
    "timeframe": 60,
    "data_limit": 100,
    "pairs_config": {
      "quote_currencies": ["USDT", "USD", "BTC", "ETH"],
      "min_volume": 10000,
      "excluded_pairs": [],
      "preferred_pairs": [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "DOGE/USDT",
        "XRP/USDT",
        "BNB/USDT",
        "ADA/USDT",
        "AVAX/USDT",
        "MATIC/USDT",
        "DOT/USDT"
      ]
    },
    "trading_rules": {
      "min_order_size": 10,
      "max_order_size": 1000,
      "min_price": 0.00000001,
      "max_leverage": 1,
      "default_slippage": 0.001
    },
    "risk_management": {
      "max_position_size": 0.1,
      "stop_loss_percentage": 0.02,
      "take_profit_percentage": 0.05,
      "max_daily_trades": 10,
      "max_daily_loss": 0.05
    },
    "alerts": {
      "price_change_threshold": 0.05,
      "volume_spike_threshold": 3,
      "rsi_overbought": 70,
      "rsi_oversold": 30
    }
  }
```

## Usage

1. Start the bot:
```bash
python main.py
```

2. Menu Options:
```
=== Trading Bot Menu by Victor Udeh ===
1. Start Trading Bot
2. Stop Trading Bot
3. Change Trading Pair
4. View Market Data
5. AI Chat (Ask for insights)
6. Export Data
7. View Market Analysis
8. Exit
```

### Using AI Chat
The AI chat feature provides real-time market insights and trading suggestions:
- Ask about specific pairs
- Get technical analysis
- Receive trading suggestions
- Analyze market conditions
- View risk assessments

### Trading Signals
The bot combines multiple signals for trading decisions:
- AI-based signals
- Technical indicators
- DQN predictions
- Price patterns
- Volume analysis

## Trading Strategies

### Moving Average Strategy
- Uses short and long-term moving averages
- Identifies potential entry and exit points
- Configurable timeframes
- Trend confirmation
- Signal validation

### DQN (Deep Q-Learning Network)
- Learns from market patterns
- Adapts to changing market conditions
- Uses reinforcement learning
- State-action mapping
- Reward optimization
- Experience replay

### SignalModel Strategy
- Profit-focused parameters
- Technical indicator combination
- Pattern recognition
- Risk/reward analysis
- Volume confirmation
- Trend strength assessment

### AI Signal Generation
- Market sentiment analysis
- Confidence-based signals
- Natural language insights
- Pattern detection
- Multi-factor analysis

## Project Structure

```
AI_day_trading_bot/
├── config/
│   └── config.json
├── models/
│   └── dqn_model.py
├── scripts/
│   └── openai_integration.py
├── strategies/
│   └── moving_average_strategy.py
├── utils/
│   ├── data_fetcher.py
│   └── trade_executor.py
├── market_data/
├── main.py
├── market_analyzer.py
├── market_exporter.py
├── pairs_manager.py
├── train_dqn.py
├── signal_model.py
├── trading_chat.py
└── requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Version History

### Version 1.1 (Current)
- Added SignalModel for improved trade analysis
- Enhanced trade execution with safety checks
- Implemented comprehensive technical indicators
- Added profit-focused trading strategies
- Improved risk management system
- Enhanced AI chat functionality
- Added support/resistance detection
- Improved pattern recognition

### Version 1.0
- Initial release
- Basic trading functionality
- AI integration
- Market analysis features
- DQN implementation
- Basic risk management

## License

This project is licensed under the MIT License

## Disclaimer

This trading bot is for educational purposes only. Cryptocurrency trading involves substantial risk and may result in the loss of your invested capital. Make sure you understand the risks involved and never trade with money you cannot afford to lose.

## Contact

Victor Udeh - [@vhicktour](https://twitter.com/realvictorudeh)

Project Link: [https://github.com/vhicktour/AI_day_trading_bot](https://github.com/vhicktour/AI_day_trading_bot)

## Acknowledgments

- OpenAI for GPT integration
- Binance for cryptocurrency trading API
- TensorFlow team for deep learning capabilities
- TA-Lib for technical analysis functions
- Contributors and community members

---
Created by [Victor Udeh](https://github.com/vhicktour)
