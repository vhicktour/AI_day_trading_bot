import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

class DQNModel:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        # Neural Network for Deep Q-Learning
        model = models.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_size, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(self.action_size, activation="linear"))  # Q-value output for each action
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        return model

    def predict(self, state):
        return self.model.predict(state)

    def train(self, state, target):
        self.model.train_on_batch(state, target)

    def save(self, path="dqn_trading_model.keras"):
        self.model.save(path)  # Saves in .keras format by default


    def load(self, path="dqn_model.h5"):
        self.model = models.load_model(path)
