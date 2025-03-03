import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import A2C  # Import A2C instead of SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from sklearn.preprocessing import MinMaxScaler

# # Download Apple stock hourly data
# data = yf.download('AAPL', interval='1d', period='10y')
# data.to_csv('AAPL_hourly.csv')

# Load Apple stock data


# Cargar archivos CSV en DataFrames de pandas, especificando la primera columna como el índice
df_diario = pd.read_csv('fx_data\\GBPUSD_D1', sep=';', index_col=0)
df_15min = pd.read_csv('fx_data\\GBPUSD_15min', sep=';', index_col=0)

# Eliminar columnas con 'Unnamed' en su nombre
df_diario = df_diario.loc[:, ~df_diario.columns.str.contains('^Unnamed')]
df_15min = df_15min.loc[:, ~df_15min.columns.str.contains('^Unnamed')]

# Eliminar las columnas especificadas
columnas_a_eliminar = [ 'Open', 'High', 'Low', 'Real_volume', 'Spread', 'timeframe', 'symbol']
df_diario = df_diario.drop(columns=columnas_a_eliminar)
df_15min = df_15min.drop(columns=columnas_a_eliminar)



# Print the first few rows of the DataFrame
print(df_diario.head())

# # Rename 'Price' column to 'Datetime'
# df_diario.rename(columns={'Price': 'Datetime'}, inplace=True)
# print(df_diario.head())
# Delete the first two rows
df_diario = df_diario.iloc[2:].reset_index(drop=True)
print(df_diario.head())

# Convertir la columna 'Time' a datetime
df_diario['Time'] = pd.to_datetime(df_diario['Time'])
df_15min['Time'] = pd.to_datetime(df_15min['Time'])


# Función para calcular el porcentaje de diferencia del precio actual sobre la media histórica
def calcular_diferencia_media(df, rolling_window):
    df['diferencia_media'] = (df['Close'] - df['Close'].rolling(window=rolling_window).mean()) / df['Close'].rolling(window=rolling_window).mean() * 100
    return df


df_15min = calcular_diferencia_media(df_15min, 45000)
df_diario = calcular_diferencia_media(df_diario, 750)


# Reemplazar NaN con 0 en todo el DataFrame
df_diario.fillna(0, inplace=True)
df_15min.fillna(0, inplace=True)

# Crear nuevos datasets a partir del primer dato que sea diferente de 0 en el campo diferencia_media
df_diario_nuevo = df_diario[df_diario['diferencia_media'] != 0].reset_index(drop=True)
df_15min_nuevo = df_15min[df_15min['diferencia_media'] != 0].reset_index(drop=True)


# Imprimir los 5 primeros registros de los nuevos datasets resultantes
print("\nDatos finales - Diario (diferencia de precio a la media de 750)")
print(df_diario_nuevo.head())

print("\nDatos finales - 15 Minutos (diferencia de precio a la media de 45000)")
print(df_15min_nuevo.head())

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize the specified columns
columns_to_normalize = ['Close', 'Volume', 'diferencia_media']
df_diario_normalized = df_diario_nuevo.copy()
df_15min_normalized = df_15min_nuevo.copy()

df_diario_normalized[columns_to_normalize] = scaler.fit_transform(df_diario_nuevo[columns_to_normalize])
df_15min_normalized[columns_to_normalize] = scaler.fit_transform(df_15min_nuevo[columns_to_normalize])

# Print the first few rows of the normalized DataFrames
print("\nDatos normalizados - Diario")
print(df_diario_normalized.head())

print("\nDatos normalizados - 15 Minutos")
print(df_15min_normalized.head())


# # Coerce columns to float numbers
# df_diario[['Close', 'High', 'Low', 'Open', 'Volume']] = df_diario[['Close', 'High', 'Low', 'Open', 'Volume']].astype(float)

# print(df_diario.head())

# # Convert index to datetime
# df_diario['Datetime'] = pd.to_datetime(df_diario['Datetime'])
df_diario.set_index('Time', inplace=True)

# Custom environment for stock trading
class StockTradingEnv(gym.Env):
    # ...existing code...
    def __init__(self, df_diario, render_mode=None):
        super(StockTradingEnv, self).__init__()
        self.df_diario = df_diario
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(df_diario.shape[1],), dtype=np.float32)
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
        return self.df_diario.iloc[self.current_step].values

    def step(self, action):
        # Action: 0 = Hold, 1 = Buy, 2 = Sell
        current_price = self.df_diario.iloc[self.current_step]['Close']
        self.action_history.append((self.current_step, action, current_price))
        if action == 1:  # Buy
            self.shares_held += self.balance // current_price
            self.balance %= current_price
        elif action == 2 and self.shares_held > 0:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        if self.current_step >= len(self.df_diario):
            self.current_step = len(self.df_diario) - 1

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - self.initial_net_worth
        done = self.current_step >= len(self.df_diario) - 1
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
def split_data(df_diario, train_period='3mo', test_period='1mo'):
    train_data = df_diario.loc[df_diario.index < df_diario.index[-1] - pd.DateOffset(months=1)]
    test_data = df_diario.loc[df_diario.index >= df_diario.index[-1] - pd.DateOffset(months=1)]
    return train_data, test_data

# Split data into training and testing datasets (50/50 split)
split_index = len(df_diario) // 2
train_data = df_diario.iloc[:split_index]
test_data = df_diario.iloc[split_index:]

# Print the first few rows of the training and testing datasets
print("\nTraining Data")
print(train_data.head())

print("\nTesting Data")
print(test_data.head())

# Create and check environment
train_env = StockTradingEnv(train_data, render_mode='human')
check_env(train_env)

# Vectorize environment
vec_env = DummyVecEnv([lambda: train_env])

# # Train model
model = A2C('MlpPolicy', vec_env, verbose=1, device='cpu')
model.learn(total_timesteps=10000)

# # Save model
# model.save("a2c_stock_trading")

nombre_modelo = "modelos2\\a2c_trading_acciones_15min_10000_20250224_175845_lr0.0003_gamma0.95_nsteps5_entcoef0.01_vfcoef0.5_maxgradnorm0.5_rmspropeps1e-05"

# Cargar el modelo guardado
modelo_cargado = A2C.load(nombre_modelo,device='cpu')


# Test model
test_env = StockTradingEnv(test_data, render_mode=None)
obs, _ = test_env.reset()
net_worths = []
for i in range(len(test_data)):
    action, _states = modelo_cargado.predict(obs)
    obs, rewards, terminated, truncated, info = test_env.step(action)
    net_worths.append(test_env.net_worth)
    test_env.render()

# Visualization routine
plt.figure(figsize=(10, 5))
plt.plot(net_worths, label='Net Worth')
plt.xlabel('Time Step')
plt.ylabel('Net Worth')
plt.title('Trading Simulation')
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
