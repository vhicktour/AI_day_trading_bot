import pandas as pd

class MovingAverageStrategy:
    def __init__(self, short_window=5, long_window=20):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        data = data.copy()
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()

        # Signal calculation based on moving average crossover
        data['signal'] = 0
        data.loc[self.short_window:, 'signal'] = (
            data['short_ma'][self.short_window:] > data['long_ma'][self.short_window:]
        ).astype(int)

        # Early indicators for pre-growth and pre-dump
        data['pre_growth'] = (
            (data['short_ma'].shift(1) < data['long_ma'].shift(1)) &
            (data['short_ma'] >= data['long_ma'])
        ).astype(int)

        data['pre_dump'] = (
            (data['short_ma'].shift(1) > data['long_ma'].shift(1)) &
            (data['short_ma'] <= data['long_ma'])
        ).astype(int)

        # Calculate final position based on signals
        data['position'] = data['signal'].diff()

        return data[['close', 'short_ma', 'long_ma', 'signal', 'pre_growth', 'pre_dump', 'position']]
