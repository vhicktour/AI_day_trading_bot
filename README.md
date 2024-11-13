# AI Day Trading Bot

An advanced cryptocurrency trading bot that combines artificial intelligence, deep learning, and traditional trading strategies for automated cryptocurrency trading. Built with Python, it leverages OpenAI's GPT models for market analysis and multiple trading strategies for optimal decision making.

![Trading Bot Banner](https://raw.githubusercontent.com/vhicktour/AI_day_trading_bot/main/docs/images/banner.png)

## Features

### 🤖 AI-Powered Trading
- Integration with OpenAI's GPT models for market analysis
- Real-time market sentiment analysis
- Dynamic confidence-based trade execution
- Natural language interaction for market insights

### 📊 Advanced Market Analysis
- Real-time market data monitoring
- Multi-timeframe analysis
- Volume profile analysis
- Price action patterns recognition
- Top movers and market momentum tracking

### 📈 Multiple Trading Strategies
- Moving Average Crossover
- Deep Q-Learning Network (DQN)
- AI signal generation
- Strategy aggregation for improved accuracy

### 🔄 Dynamic Pair Management
- Support for all major cryptocurrency pairs
- Real-time market scanning
- Volume-based filtering
- Automated pair selection based on market conditions

### 📉 Risk Management
- Configurable position sizing
- Stop-loss and take-profit management
- Maximum daily loss limits
- Trading volume restrictions

### 📊 Data Export & Analysis
- CSV export functionality
- Historical data analysis
- Performance metrics tracking
- Market data archiving

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vhicktour/AI_day_trading_bot.git
cd AI_day_trading_bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Configure your API keys:
- Copy `config/config.example.json` to `config/config.json`
- Add your Binance API and OpenAI API keys

## Configuration

The bot is configured through `config/config.json`:

```json
{
    "api_key": "your_binance_api_key",
    "api_secret": "your_binance_api_secret",
    "initial_investment": 50,
    "trade_pair": "BTC/USDT",
    "trade_amount": 0.001,
    "openai_key": "your_openai_api_key",
    "use_ai_signals": true,
    "use_dqn": true,
    "pairs_config": {
        "quote_currencies": ["USDT", "USD", "BTC", "ETH"],
        "min_volume": 10000
    }
}
```

## Usage

1. Start the bot:
```bash
python main.py
```

2. Menu Options:
- Start/Stop Trading Bot
- Change Trading Pair
- View Market Data
- AI Chat for Market Insights
- Export Market Data
- View Market Analysis

## Trading Strategies

### Moving Average Strategy
- Uses short and long-term moving averages
- Identifies potential entry and exit points
- Configurable timeframes

### DQN (Deep Q-Learning Network)
- Learns from market patterns
- Adapts to changing market conditions
- Uses reinforcement learning for decision making

### AI Signal Generation
- Analyzes market sentiment
- Provides confidence-based signals
- Natural language market insights

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
└── pairs_manager.py
|__train_dqn.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Disclaimer

This trading bot is for educational purposes only. Cryptocurrency trading involves substantial risk and may result in the loss of your invested capital. Make sure you understand the risks involved and never trade with money you cannot afford to lose.

## Contact

Victor Udeh - [@vhicktour](https://twitter.com/vhicktour)

Project Link: [https://github.com/vhicktour/AI_day_trading_bot](https://github.com/vhicktour/AI_day_trading_bot)

## Acknowledgments

- OpenAI for GPT integration
- Binance for cryptocurrency trading API
- TensorFlow team for deep learning capabilities

---
Created by [Victor Udeh](https://github.com/vhicktour)
