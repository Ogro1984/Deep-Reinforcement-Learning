import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import A2C  # Import A2C instead of SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Download Apple stock hourly data
data = yf.download('AAPL', interval='1d', period='10y')
data.to_csv('AAPL_hourly.csv')

# Load Apple stock data
df = pd.read_csv('AAPL_hourly.csv')
# Print the first few rows of the DataFrame
print(df.head())

# Rename 'Price' column to 'Datetime'
df.rename(columns={'Price': 'Datetime'}, inplace=True)
print(df.head())
# Delete the first two rows
df = df.iloc[2:].reset_index(drop=True)

# Coerce columns to float numbers
df[['Close', 'High', 'Low', 'Open', 'Volume']] = df[['Close', 'High', 'Low', 'Open', 'Volume']].astype(float)

print(df.head())

# Convert index to datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

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

# Function to split data into training and testing periods
def split_data(df, train_period='3mo', test_period='1mo'):
    train_data = df.loc[df.index < df.index[-1] - pd.DateOffset(months=1)]
    test_data = df.loc[df.index >= df.index[-1] - pd.DateOffset(months=1)]
    return train_data, test_data

# Iterate through the data and train/test the model
start_date = pd.to_datetime(df.index[0])
end_date = pd.to_datetime(df.index[-1])
current_date = end_date

while current_date >= start_date + pd.DateOffset(months=4):
    train_end_date = current_date - pd.DateOffset(months=1)
    train_start_date = train_end_date - pd.DateOffset(months=3)
    test_start_date = train_end_date
    test_end_date = current_date

    train_data = df.loc[train_start_date:train_end_date]
    test_data = df.loc[test_start_date:test_end_date]

    # Create and check environment
    train_env = StockTradingEnv(train_data, render_mode='human')
    check_env(train_env)

    # Vectorize environment
    vec_env = DummyVecEnv([lambda: train_env])

    # Train model
    model = A2C('MlpPolicy', vec_env, verbose=1, device='cpu')
    model.learn(total_timesteps=10000)

    # Save model
    model.save(f"a2c_stock_trading_{train_start_date.strftime('%Y%m%d')}_{train_end_date.strftime('%Y%m%d')}")

    # Test model
    test_env = StockTradingEnv(test_data, render_mode='human')
    obs, _ = test_env.reset()
    net_worths = []
    for i in range(len(test_data)):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = test_env.step(action)
        net_worths.append(test_env.net_worth)
        test_env.render()

    # Visualization routine
    plt.figure(figsize=(10, 5))
    plt.plot(net_worths, label='Net Worth')
    plt.xlabel('Time Step')
    plt.ylabel('Net Worth')
    plt.title(f'Trading Simulation {test_start_date.strftime("%Y-%m-%d")} to {test_end_date.strftime("%Y-%m-%d")}')
    plt.legend()
    plt.show()

    # Plot action prices
    actions = pd.DataFrame(test_env.action_history, columns=['Step', 'Action', 'Price'])
    plt.figure(figsize=(10, 5))
    plt.plot(test_data['Close'].reset_index(drop=True), label='Close Price')
    plt.scatter(actions[actions['Action'] == 1]['Step'], actions[actions['Action'] == 1]['Price'], color='green', label='Buy', marker='^', alpha=1)
    plt.scatter(actions[actions['Action'] == 2]['Step'], actions[actions['Action'] == 2]['Price'], color='red', label='Sell', marker='v', alpha=1)
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title('Trading Actions')
    plt.legend()
    plt.show()

    # Move to the next period
    current_date = train_start_date
