import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Download Apple stock hourly data
data = yf.download('AAPL', interval='1h', period='1y')
data.to_csv('AAPL_hourly.csv')

# Load Apple stock data
df = pd.read_csv('AAPL_hourly.csv')
# Drop the 'data' column
df = df.drop(columns=['Price'])

# Delete the first two rows
df = df.iloc[2:].reset_index(drop=True)
df = df.apply(pd.to_numeric, errors='coerce')  # Ensure all data is numeric
df.head()

# Custom environment for stock trading
class StockTradingEnv(gym.Env):
    # ...existing code...
    def __init__(self, df, render_mode=None):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(df.shape[1],), dtype=np.float32)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = 10000
        self.initial_net_worth = 10000
        self.action_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = 10000
        self.action_history = []
        return self._next_observation().astype(np.float32), {}

    def _next_observation(self):
        return self.df.iloc[self.current_step].values

    def step(self, action):
        # Action: 0 = Hold, 1 = Buy, 2 = Sell
        current_price = self.df.iloc[self.current_step]['Close']
        self.action_history.append((self.current_step, action, current_price))
        if action == 1:  # Buy
            self.shares_held += self.balance // current_price
            self.balance %= current_price
        elif action == 2 and self.shares_held > 0:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - self.initial_net_worth
        done = self.current_step >= len(self.df) - 1
        obs = self._next_observation().astype(np.float32)
        terminated = done
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        if self.render_mode is not None:
            # ...existing code...
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Shares held: {self.shares_held}')
            print(f'Net worth: {self.net_worth}')

# Create and check environment
env = StockTradingEnv(df, render_mode='human')
check_env(env)

# Vectorize environment
vec_env = DummyVecEnv([lambda: env])

# # Train model
# model = PPO('MlpPolicy', vec_env, verbose=1)
# model.learn(total_timesteps=10000)

# # Save model
# model.save("ppo_stock_trading")

# Load model
model = PPO.load("ppo_stock_trading")

# Test model
obs, _ = env.reset()
net_worths = []
for i in range(len(df)):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    net_worths.append(env.net_worth)
    env.render()

# Visualization routine
plt.figure(figsize=(10, 5))
plt.plot(net_worths, label='Net Worth')
plt.xlabel('Time Step')
plt.ylabel('Net Worth')
plt.title('Trading Simulation')
plt.legend()
plt.show()

# Plot action prices
actions = pd.DataFrame(env.action_history, columns=['Step', 'Action', 'Price'])
plt.figure(figsize=(10, 5))
plt.plot(df['Close'], label='Close Price')
plt.scatter(actions[actions['Action'] == 1]['Step'], actions[actions['Action'] == 1]['Price'], color='green', label='Buy', marker='^', alpha=1)
plt.scatter(actions[actions['Action'] == 2]['Step'], actions[actions['Action'] == 2]['Price'], color='red', label='Sell', marker='v', alpha=1)
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.title('Trading Actions')
plt.legend()
plt.show()
