import numpy as np

class TradingEnvironment:
    def __init__(self, data, initial_balance=1000):
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.holdings = 0
        self.state_size = 5  # Number of data points used as state

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        return self._get_state()

    def _get_state(self):
        # Returns the last 'state_size' closing prices as the state
        state = self.data['close'].values[self.current_step:self.current_step + self.state_size]
        return np.reshape(state, [1, self.state_size])

    def step(self, action):
        # Ensure we do not go out of bounds
        if self.current_step + self.state_size >= len(self.data):
            done = True
            return self._get_state(), 0, done

        current_price = self.data['close'].values[self.current_step + self.state_size]
        reward = 0
        done = False

        if action == 1:  # Buy
            if self.balance > 0:
                self.holdings += self.balance / current_price
                self.balance = 0
        elif action == 2 and self.holdings > 0:  # Sell
            self.balance += self.holdings * current_price
            self.holdings = 0
            reward = self.balance - self.initial_balance  # Reward based on profit

        # Increment step
        self.current_step += 1
        done = self.current_step >= len(self.data) - self.state_size - 1
        next_state = self._get_state()

        return next_state, reward, done
