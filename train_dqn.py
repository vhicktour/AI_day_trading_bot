import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

import numpy as np
import pandas as pd
from models.dqn_model import DQNModel
from utils.environment import TradingEnvironment  # Explicitly importing from utils
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = DQNModel(state_size, action_size)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.train(state, target_f)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load(name)

    def save(self, name):
        self.model.save(name)

# Initialize environment and agent
def train_dqn_agent():
    # Load Data
    data = pd.read_csv('data/historical_data.csv')
    env = TradingEnvironment(data)
    agent = DQNAgent(env.state_size, 3)  # Actions: [Hold, Buy, Sell]

    # Training loop
    for e in range(1000):  # Train for 1000 episodes
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {e}/{1000}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
                break

        if len(agent.memory) > 32:
            agent.replay(32)

        # Save model after each episode
        agent.save("dqn_trading_model.keras")  # Save the model in .keras format


if __name__ == "__main__":
    train_dqn_agent()
