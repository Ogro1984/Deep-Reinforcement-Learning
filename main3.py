import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Definir el entorno StockTradingEnv
class EntornoTradingAcciones(gym.Env):
    def __init__(self, df, modo_render=None):
        super(EntornoTradingAcciones, self).__init__()
        self.df = df
        self.action_space = gym.spaces.Discrete(3)  # Comprar, Mantener, Vender
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = 10000
        self.render_mode = modo_render
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
        self.current_step += 1
        reward = self.net_worth - 10000
        done = self.current_step >= len(self.df) - 1
        obs = self._next_observation().astype(np.float32)
        terminated = done
        truncated = False
        info = {}
        self.action_history.append([self.current_step, action, self.df.iloc[self.current_step]['Close']])
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        if self.render_mode is not None:
            print(f'Paso: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Acciones poseídas: {self.shares_held}')
            print(f'Valor neto: {self.net_worth}')

# Función para calcular la diferencia media
def calcular_diferencia_media(df, window):
    df['diferencia_media'] = df['Close'].rolling(window=window).mean() - df['Close']
    return df

# Función para normalizar DataFrames
def normalizar_df(df):
    scaler = MinMaxScaler()
    df_normalizado = df.copy()
    columnas_a_normalizar = ['Close', 'Volume', 'diferencia_media']
    df_normalizado[columnas_a_normalizar] = scaler.fit_transform(df[columnas_a_normalizar])
    return df_normalizado

# Función para crear una nueva instancia del entorno
def make_env(df):
    return lambda: EntornoTradingAcciones(df)

# Asegurarse de que todos los DataFrames tengan las mismas columnas
def alinear_columnas(df, columnas_referencia):
    columnas_faltantes = set(columnas_referencia) - set(df.columns)
    for columna in columnas_faltantes:
        df[columna] = 0
    return df[columnas_referencia]

# Cargar archivos CSV en DataFrames de pandas, especificando la primera columna como el índice
df_diario = pd.read_csv('fx_data\\GBPUSD_D1', sep=';', index_col=0)
df_15min = pd.read_csv('fx_data\\GBPUSD_15min', sep=';', index_col=0)

# Eliminar columnas con 'Unnamed' en su nombre
df_diario = df_diario.loc[:, ~df_diario.columns.str.contains('^Unnamed')]
df_15min = df_15min.loc[:, ~df_15min.columns.str.contains('^Unnamed')]

# Eliminar las columnas especificadas
columnas_a_eliminar = ['Open', 'High', 'Low', 'Real_volume', 'Spread', 'timeframe', 'symbol']
df_diario = df_diario.drop(columns=columnas_a_eliminar)
df_15min = df_15min.drop(columns=columnas_a_eliminar)

# Convertir la columna 'Time' a datetime
df_diario['Time'] = pd.to_datetime(df_diario['Time'])
df_15min['Time'] = pd.to_datetime(df_15min['Time'])

# Obtener las columnas de referencia del DataFrame original
columnas_referencia = df_15min.columns

# Crear el directorio para guardar los resultados
os.makedirs('resultados', exist_ok=True)

# Lista para almacenar los resultados
resultados = []

# Obtener la lista de modelos en la carpeta 'modelos'
modelos = [f for f in os.listdir('modelos2') ]

# Evaluar cada modelo
for nombre_modelo in modelos:
    # Extraer información del nombre del modelo
    match = re.search(r'_(15min|diario)_(\d+)_', nombre_modelo)
    if match:
        tipo = match.group(1)
        window = int(match.group(2))

        # Seleccionar el dataset correspondiente
        if tipo == '15min':
            df = df_15min.copy()
        else:
            df = df_diario.copy()

        # Calcular la diferencia media
        df = calcular_diferencia_media(df, window)
        print(df.head(10))
        # Reemplazar NaN con 0 en todo el DataFrame
        df.fillna(0, inplace=True)
        print(df.head(10))
        # Dividir el DataFrame en 60% para entrenamiento y 40% para pruebas
        split_index = int(len(df) * 0.6)
        df_test = df.iloc[split_index:].copy()
        print(df_test.head(10))
        # Crear variantes normalizadas y no normalizadas
        df_normalizado = normalizar_df(df_test)
        df_no_normalizado = df_test.copy()
        print(df_normalizado.head(10))  
        # Alinear las columnas de los DataFrames con las columnas de referencia
        #df_normalizado = alinear_columnas(df_normalizado, columnas_referencia)
        #df_no_normalizado = alinear_columnas(df_no_normalizado, columnas_referencia)
        print(df_normalizado.head(10))
        print('kawabanga')
        print(df_no_normalizado.head(10))
        # Establecer la columna 'Time' como índice
        df_normalizado.set_index('Time', inplace=True)
        df_no_normalizado.set_index('Time', inplace=True)

        print(df_normalizado.head(10))
        print(df_no_normalizado.head(10))

        # Evaluar el modelo con ambos datasets
        for df_variant, variant_name in [(df_normalizado, 'normalizado'), (df_no_normalizado, 'no_normalizado')]:
            # Cargar el modelo guardado
            modelo_cargado = A2C.load(os.path.join('modelos', nombre_modelo),device='cpu')

            # Crear el entorno de prueba
            test_env = EntornoTradingAcciones(df_variant, modo_render=None)
            obs, _ = test_env.reset()
            net_worths = []
            for i in range(len(df_variant)):
                action, _states = modelo_cargado.predict(obs)
                obs, rewards, terminated, truncated, info = test_env.step(action)
                net_worths.append(test_env.net_worth)
                if terminated or truncated:
                    break

            # Calcular el rendimiento
            rendimiento = net_worths[-1] - net_worths[0]
            resultados.append([nombre_modelo, tipo, variant_name, rendimiento])

# Guardar los resultados en un archivo CSV
resultados_df = pd.DataFrame(resultados, columns=['Modelo', 'Tipo', 'Variante', 'Rendimiento'])
resultados_df.to_csv('resultados/resultados_modelos.csv', index=False)

# Función para renderizar gráficos
def renderizar_grafico(resultados_df, tipo, variante, filename):
    resultados_filtrados = resultados_df[(resultados_df['Tipo'] == tipo) & (resultados_df['Variante'] == variante)]
    resultados_filtrados = resultados_filtrados.sort_values(by='Rendimiento', ascending=False)
    top_10 = resultados_filtrados.head(10)

    plt.figure(figsize=(12, 6))
    plt.plot(resultados_filtrados['Modelo'], resultados_filtrados['Rendimiento'], label='Rendimiento')
    plt.scatter(top_10['Modelo'], top_10['Rendimiento'], color='red', label='Top 10', zorder=5)
    plt.xlabel('Modelo')
    plt.ylabel('Rendimiento')
    plt.title(f'Rendimiento de Modelos - {tipo.capitalize()} - {variante.capitalize()}')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Renderizar gráficos para cada grupo
renderizar_grafico(resultados_df, '15min', 'normalizado', 'resultados/rendimiento_modelos_15min_normalizado.png')
renderizar_grafico(resultados_df, '15min', 'no_normalizado', 'resultados/rendimiento_modelos_15min_no_normalizado.png')
renderizar_grafico(resultados_df, 'diario', 'normalizado', 'resultados/rendimiento_modelos_diario_normalizado.png')
renderizar_grafico(resultados_df, 'diario', 'no_normalizado', 'resultados/rendimiento_modelos_diario_no_normalizado.png')