import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym  # Use gymnasium instead of gym
from gymnasium.spaces import Discrete, Box  # Use gymnasium spaces
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from gymnasium import spaces
import re
import csv
# Entorno personalizado para el trading de acciones
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, shares_per_step=10, commission=0.001, render_mode=None):
        super().__init__()
        self.df = df  # DataFrame con los datos del mercado
        self.initial_balance = initial_balance  # Balance inicial
        self.balance = initial_balance  # Balance actual
        self.net_worth = initial_balance  # Patrimonio neto actual
        self.shares_held = 0  # Cantidad de acciones en posesión
        self.shares_per_step = shares_per_step  # Cantidad de acciones a comprar/vender en cada paso
        self.commission = commission  # Comisión por transacción
        self.current_step = 0  # Paso actual en el entorno
        self.reward_range = (-float('inf'), float('inf'))  # Rango de recompensas
        self.action_space = Discrete(3)  # Espacio de acciones: 0: hold, 1: buy, 2: sell
        self.observation_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)  # Espacio de observaciones
        self.render_mode = render_mode  # Modo de renderización
        self.action_history = []  # Historial de acciones
        self.observation_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)

    # Función para generar la siguiente observación
    def _next_observation(self):
        frame = np.array([
            self.df.iloc[self.current_step]['Close'] / 1000,  # Precio de cierre escalado
            self.df.iloc[self.current_step]['Volume'] / 1000000,  # Volumen escalado
            self.balance / self.initial_balance,  # Balance relativo al balance inicial
            self.shares_held / 100,  # Acciones en posesión escaladas
            self.net_worth / self.initial_balance,  # Patrimonio neto relativo al balance inicial
        ], dtype=np.float32)
        return frame

    # Función para realizar una acción
    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['Close']  # Precio actual
        trade_quantity = self.shares_per_step  # Cantidad a transar
        cost = trade_quantity * current_price * (1 + self.commission)  # Costo de la transacción

        if action == 1:  # Comprar
            if self.balance >= cost:  # Verificar si hay suficiente balance
                self.balance -= cost  # Reducir el balance
                self.shares_held += trade_quantity  # Aumentar las acciones en posesión
        elif action == 2:  # Vender
            if self.shares_held >= trade_quantity:  # Verificar si hay suficientes acciones
                self.balance += trade_quantity * current_price * (1 - self.commission)  # Aumentar el balance
                self.shares_held -= trade_quantity  # Reducir las acciones en posesión

        self.net_worth = self.balance + self.shares_held * current_price  # Calcular el patrimonio neto

    # Función para realizar un paso en el entorno
    def step(self, action):
        terminated = self.current_step >= len(self.df) - 1  # Verificar si el episodio ha terminado
        truncated = False  # No se utiliza en este entorno

        if not terminated:
            self.current_step += 1  # Avanzar al siguiente paso
            self._take_action(action)  # Realizar la acción
            obs = self._next_observation()  # Obtener la siguiente observación
            reward = (self.net_worth - self.initial_balance) / self.initial_balance  # Calcular la recompensa
        else:
            obs = self._next_observation()  # Obtener la observación final
            reward = 0  # Recompensa cero al final del episodio

        info = {'step': self.current_step, 'balance': self.balance, 'shares_held': self.shares_held, 'net_worth': self.net_worth}  # Información adicional
        self.action_history.append([self.current_step, action, self.df.iloc[self.current_step]['Close']])  # Registrar la acción
        return obs, reward, terminated, truncated, info  # Devolver los resultados

    # Función para renderizar el entorno (opcional)
    def render(self, mode='human'):
        if self.render_mode is not None:
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Shares held: {self.shares_held}')
            print(f'Net worth: {self.net_worth}')

    # Función para resetear el entorno
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance  # Resetear el balance
        self.net_worth = self.initial_balance  # Resetear el patrimonio neto
        self.shares_held = 0  # Resetear las acciones en posesión
        self.current_step = 0  # Resetear el paso actual
        obs = self._next_observation()  # Obtener la observación inicial
        info = {}  # Información adicional
        self.action_history = []  # Resetear el historial de acciones
        return obs, info  # Devolver la observación y la información

# Entorno personalizado para el trading de acciones compatible con SAC
class StockTradingEnvSAC(gym.Env):
    def __init__(self, df, initial_balance=10000, shares_per_step=10, commission=0.001, render_mode=None):
        super().__init__()
        self.df = df  # DataFrame con los datos del mercado
        self.initial_balance = initial_balance  # Balance inicial
        self.balance = initial_balance  # Balance actual
        self.net_worth = initial_balance  # Patrimonio neto actual
        self.shares_held = 0  # Cantidad de acciones en posesión
        self.shares_per_step = shares_per_step  # Cantidad de acciones a comprar/vender en cada paso
        self.commission = commission  # Comisión por transacción
        self.current_step = 0  # Paso actual en el entorno
        self.reward_range = (-float('inf'), float('inf'))  # Rango de recompensas
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Espacio de acciones continuo
        self.observation_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)  # Espacio de observaciones
        self.render_mode = render_mode  # Modo de renderización
        self.action_history = []  # Historial de acciones
        self.observation_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)

    # Función para generar la siguiente observación
    def _next_observation(self):
        frame = np.array([
            self.df.iloc[self.current_step]['Close'] / 1000,  # Precio de cierre escalado
            self.df.iloc[self.current_step]['Volume'] / 1000000,  # Volumen escalado
            self.balance / self.initial_balance,  # Balance relativo al balance inicial
            self.shares_held / 100,  # Acciones en posesión escaladas
            self.net_worth / self.initial_balance,  # Patrimonio neto relativo al balance inicial
        ], dtype=np.float32)
        return frame

    # Función para realizar una acción
    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['Close']  # Precio actual
        trade_quantity = self.shares_per_step * action[0]  # Cantidad a transar, ajustada por la acción

        cost = abs(trade_quantity) * current_price * (1 + self.commission)  # Costo de la transacción

        if action[0] > 0:  # Comprar
            if self.balance >= cost:  # Verificar si hay suficiente balance
                self.balance -= cost  # Reducir el balance
                self.shares_held += trade_quantity  # Aumentar las acciones en posesión
        elif action[0] < 0:  # Vender
            if self.shares_held >= abs(trade_quantity):  # Verificar si hay suficientes acciones
                self.balance += abs(trade_quantity) * current_price * (1 - self.commission)  # Aumentar el balance
                self.shares_held -= abs(trade_quantity)  # Reducir las acciones en posesión

        self.net_worth = self.balance + self.shares_held * current_price  # Calcular el patrimonio neto

    # Función para realizar un paso en el entorno
    def step(self, action):
        terminated = self.current_step >= len(self.df) - 1  # Verificar si el episodio ha terminado
        truncated = False  # No se utiliza en este entorno

        if not terminated:
            self.current_step += 1  # Avanzar al siguiente paso
            self._take_action(action)  # Realizar la acción
            obs = self._next_observation()  # Obtener la siguiente observación
            reward = (self.net_worth - self.initial_balance) / self.initial_balance  # Calcular la recompensa
        else:
            obs = self._next_observation()  # Obtener la observación final
            reward = 0  # Recompensa cero al final del episodio

        info = {'step': self.current_step, 'balance': self.balance, 'shares_held': self.shares_held, 'net_worth': self.net_worth}  # Información adicional
        self.action_history.append([self.current_step, action, self.df.iloc[self.current_step]['Close']])  # Registrar la acción
        return obs, reward, terminated, truncated, info  # Devolver los resultados

    # Función para renderizar el entorno (opcional)
    def render(self, mode='human'):
        if self.render_mode is not None:
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Shares held: {self.shares_held}')
            print(f'Net worth: {self.net_worth}')

    # Función para resetear el entorno
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance  # Resetear el balance
        self.net_worth = self.initial_balance  # Resetear el patrimonio neto
        self.shares_held = 0  # Resetear las acciones en posesión
        self.current_step = 0  # Resetear el paso actual
        obs = self._next_observation()  # Obtener la observación inicial
        info = {}  # Información adicional
        self.action_history = []  # Resetear el historial de acciones
        return obs, info  # Devolver la observación y la información



# Función para entrenar el modelo A2C
def train_model(train_data_path):
    """
    Entrena un modelo A2C con los datos de entrenamiento proporcionados.

    Args:
        train_data_path (str): Ruta al archivo CSV de datos de entrenamiento.
    """
    # Cargar los datos de entrenamiento
    train_df = pd.read_csv(train_data_path)

    # Crear y verificar el entorno de entrenamiento
    train_env = StockTradingEnv(train_df, render_mode=None)
    check_env(train_env)

    # Vectorizar el entorno de entrenamiento
    vec_env = DummyVecEnv([lambda: train_env])

    # Definir rangos de hiperparámetros
    learning_rates = [0.0001, 0.0007, 0.001]  # Tasas de aprendizaje
    
    gammas = [0.95, 0.99]  # Factores de descuento
   
    n_steps_list = [5, 10]  # Número de pasos antes de actualizar el modelo
    
    ent_coefs = [0.01]  # Coeficientes de la pérdida de entropía
    vf_coefs = [0.5]  # Coeficientes de la pérdida de la función de valor
    max_grad_norms = [0.5]  # Valores máximos para la normalización del gradiente
    rms_prop_epss = [1e-5]  # Epsilon para RMSProp

    # Extract information from the training data path
    train_data_filename = os.path.basename(train_data_path)
    is_filtered = "no_filtrado" not in train_data_path
    is_normalized = "no_normalizado" not in train_data_path

    # CSV file path
    csv_file = os.path.join(base_path, "training_results.csv")
    file_exists = os.path.isfile(csv_file)


    # Iterar a través de las combinaciones de hiperparámetros
    for learning_rate in learning_rates:
        for gamma in gammas:
            for n_steps in n_steps_list:
                for ent_coef in ent_coefs:
                    for vf_coef in vf_coefs:
                        for max_grad_norm in max_grad_norms:
                            for rms_prop_eps in rms_prop_epss:
                                # Definir el nombre del modelo
                                model_name = f"a2c_lr{learning_rate}_gamma{gamma}_nsteps{n_steps}_ent{ent_coef}_vf{vf_coef}_gradnorm{max_grad_norm}_rms{rms_prop_eps}"

                                # Crear la carpeta para los entrenamientos
                                training_folder = os.path.join(os.path.dirname(train_data_path), "entrenamientos", "A2C")
                                os.makedirs(training_folder, exist_ok=True)

                                # Crear la carpeta para el modelo entrenado
                                model_folder = os.path.join(training_folder, model_name)
                                os.makedirs(model_folder, exist_ok=True)

                                model_path = os.path.join(model_folder, "model")

                                print(f"Entrenando modelo: {model_name}")

                                # Entrenar el modelo
                                model = A2C('MlpPolicy', vec_env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps,
                                            ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, rms_prop_eps=rms_prop_eps,
                                            verbose=0, device='cpu')
                                  # Entrenar el modelo
                                # model = A2C('MlpPolicy',vec_env,verbose=0, device='cpu')
                                model.learn(total_timesteps=10000)

                                # Guardar el modelo
                                model.save(model_path)
                                print(f"Modelo guardado en: {model_path}")

                                # Guardar las estadísticas de entrenamiento
                                stats_path = os.path.join(model_folder, "stats.txt")
                                with open(stats_path, "w") as f:
                                    f.write(f"Tasa de Aprendizaje: {learning_rate}\n")
                                    f.write(f"Gamma: {gamma}\n")
                                    f.write(f"N Pasos: {n_steps}\n")
                                    f.write(f"Ent Coef: {ent_coef}\n")
                                    f.write(f"VF Coef: {vf_coef}\n")
                                    f.write(f"Max Grad Norm: {max_grad_norm}\n")
                                    f.write(f"RMS Prop Eps: {rms_prop_eps}\n")
                                print(f"Estadísticas de entrenamiento guardadas en: {stats_path}")
                                # Extract algorithm name from the model path
                                algorithm_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

                                # Use a more robust regex to extract parameters
                                params = re.findall(r"([a-z]+)([0-9\.e-]+)", model_name)
                                params_dict = {p[0]: p[1] for p in params}

                                # Write to CSV
                                with open(csv_file, mode='a', newline='') as f:
                                    writer = csv.writer(f)
                                    if not file_exists:
                                        writer.writerow([
                                            "Algorithm", "Learning Rate", "Gamma", "N Steps", "Ent Coef", "VF Coef", "Max Grad Norm", "RMS Prop Eps",
                                            "Train Data", "Filtered", "Normalized"
                                        ])
                                        file_exists = True  # Ensure header is only written once

                                    writer.writerow([
                                        algorithm_name, params_dict.get("lr", ""), params_dict.get("gamma", ""), params_dict.get("nsteps", ""),
                                        params_dict.get("ent", ""), params_dict.get("vf", ""), params_dict.get("gradnorm", ""), params_dict.get("rms", ""),
                                        train_data_filename, is_filtered, is_normalized
                                    ])

    print(f"Estadísticas de entrenamiento guardadas en: {csv_file}")

def test_model(model_path, test_data_path):
    """
    Prueba un modelo A2C con los datos de prueba proporcionados y genera un gráfico de las acciones tomadas.

    Args:
        model_path (str): Ruta al modelo entrenado.
        test_data_path (str): Ruta al archivo CSV de datos de prueba.
    """
    # Cargar los datos de prueba
    test_df = pd.read_csv(test_data_path)

    # Crear el entorno de prueba
    test_env = StockTradingEnv(test_df, render_mode=None)
    obs, _ = test_env.reset()
    net_worths = []
    actions = []

    # Cargar el modelo
    
    model = A2C.load(model_path, device="cpu")

    # Probar el modelo
    for i in range(len(test_df)):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = test_env.step(action)
        net_worths.append(test_env.net_worth)
        actions.append(action)

    # Guardar las estadísticas de prueba
    stats_path = os.path.dirname(model_path) + "/test_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Patrimonio Neto Final: {test_env.net_worth}\n")
    print(f"Estadísticas de prueba guardadas en: {stats_path}")

# Calcular el número de operaciones de compra y venta
    buy_operations = actions.count(1)
    sell_operations = actions.count(2)

    # Calcular el retorno general
    initial_net_worth = test_env.initial_balance
    final_net_worth = test_env.net_worth
    overall_return = (final_net_worth - initial_net_worth) / initial_net_worth

    # Obtener el nombre del modelo y los parámetros
    model_name = os.path.basename(os.path.dirname(model_path))
    params = re.findall(r"([a-z]+[-]?)([0-9\.e-]+)", model_name)
    params_dict = {p[0]: p[1] for p in params}
# Obtener información del conjunto de datos de prueba
    test_data_filename = os.path.basename(test_data_path)
    is_filtered = "no_filtrado" not in test_data_path
    is_normalized = "no_normalizado" not in test_data_path

   # Guardar las estadísticas de prueba en un archivo CSV
    csv_file = os.path.join(base_path, "testing_results.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Algorithm", "Learning Rate", "Gamma", "N Steps", "Ent Coef", "VF Coef", "Max Grad Norm", "RMS Prop Eps",
                "Test Data", "Filtered", "Normalized",
                "Overall Return", "Final Net Worth", "Buy Operations", "Sell Operations"
            ])

        # Extract algorithm name from the model path
        algorithm_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

        # Use a more robust regex to extract parameters
        params = re.findall(r"([a-z]+)([0-9\.e-]+)", model_name)
        params_dict = {p[0]: p[1] for p in params}

        writer.writerow([
            algorithm_name, params_dict.get("lr", ""), params_dict.get("gamma", ""), params_dict.get("nsteps", ""),
            params_dict.get("ent", ""), params_dict.get("vf", ""), params_dict.get("gradnorm", ""), params_dict.get("rms", ""),
            test_data_filename, is_filtered, is_normalized,
            overall_return, final_net_worth, buy_operations, sell_operations
        ])
    print(f"Estadísticas de prueba guardadas en: {csv_file}")
   





    # Rutina de visualización
    plt.figure(figsize=(10, 5))
    plt.plot(net_worths, label='Patrimonio Neto')
    plt.xlabel('Paso de Tiempo')
    plt.ylabel('Patrimonio Neto')
    plt.title(f'Patrimonio Neto a lo Largo del Tiempo - {os.path.basename(os.path.dirname(model_path))} (Prueba)')

    # Marcar las acciones en el gráfico
    close_prices = test_df['Close'].values
    buy_indices = np.where(np.array(actions) == 1)[0]
    sell_indices = np.where(np.array(actions) == 2)[0]

    plt.scatter(buy_indices, [net_worths[i] for i in buy_indices], marker='^', color='green', label='Compra')
    plt.scatter(sell_indices, [net_worths[i] for i in sell_indices], marker='v', color='red', label='Venta')

    plt.legend()
    plt.savefig(os.path.dirname(model_path) + "/net_worth_test.png")
    plt.close()

def process_training_data(base_path):
    """
    Procesa los datos de entrenamiento en la estructura de carpetas especificada.

    Args:
        base_path (str): Ruta base donde se encuentran las carpetas de datos de entrenamiento.
    """
    # Iterar a través de las carpetas de divisas
    for currency_pair in os.listdir(base_path):
        currency_path = os.path.join(base_path, currency_pair)

        # Verificar si es una carpeta
        if not os.path.isdir(currency_path):
            continue

        # Iterar a través de las carpetas de timeframe
        for timeframe in os.listdir(currency_path):
            timeframe_path = os.path.join(currency_path, timeframe)

            # Verificar si es una carpeta
            if not os.path.isdir(timeframe_path):
                continue

            # Ir a la carpeta de datos de entrenamiento
            data_path = os.path.join(timeframe_path, "train")

            # Verificar si la carpeta existe
            if not os.path.exists(data_path):
                continue

            # Iterar a través de las carpetas de filtrado
            for filter_type in os.listdir(data_path):
                filter_path = os.path.join(data_path, filter_type)

                # Verificar si es una carpeta
                if not os.path.isdir(filter_path):
                    continue

                # Si la palabra "no_filtrado" está en la ruta, la iteración debe comenzar en las subcarpetas
                if "no_filtrado" in filter_path:
                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(filter_path):
                        window_path = os.path.join(filter_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        # Entrenar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                train_data_path = os.path.join(window_path, filename)
                                train_model(train_data_path)
                                train_ppo_model(train_data_path, base_path)  # Train PPO as well

                # Si "no_filtrado" no está en la ruta, ir a la subcarpeta "wavelet"
                else:
                    wavelet_path = os.path.join(filter_path, "wavelet")

                    # Verificar si la carpeta "wavelet" existe
                    if not os.path.exists(wavelet_path):
                        continue

                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(wavelet_path):
                        window_path = os.path.join(wavelet_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        # Entrenar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                train_data_path = os.path.join(window_path, filename)
                                train_model(train_data_path)
                                train_ppo_model(train_data_path, base_path)  # Train PPO as well

def process_testing_data(base_path):
    """
    Procesa los datos de prueba en la estructura de carpetas especificada.

    Args:
        base_path (str): Ruta base donde se encuentran las carpetas de datos de prueba.
    """
    # Iterar a través de las carpetas de divisas
    for currency_pair in os.listdir(base_path):
        currency_path = os.path.join(base_path, currency_pair)

        # Verificar si es una carpeta
        if not os.path.isdir(currency_path):
            continue

        # Iterar a través de las carpetas de timeframe
        for timeframe in os.listdir(currency_path):
            timeframe_path = os.path.join(currency_path, timeframe)

            # Verificar si es una carpeta
            if not os.path.isdir(timeframe_path):
                continue

            # Ir a la carpeta de datos de prueba
            data_path = os.path.join(timeframe_path, "test")

            # Verificar si la carpeta existe
            if not os.path.exists(data_path):
                continue

            # Iterar a través de las carpetas de filtrado
            for filter_type in os.listdir(data_path):
                filter_path = os.path.join(data_path, filter_type)

                # Verificar si es una carpeta
                if not os.path.isdir(filter_path):
                    continue

                # Si la palabra "no_filtrado" está en la ruta, la iteración debe comenzar en las subcarpetas
                if "no_filtrado" in filter_path:
                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(filter_path):
                        window_path = os.path.join(filter_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        # Probar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                test_data_path = os.path.join(window_path, filename)
                                
                                # Construir la ruta al modelo entrenado
                                train_base_path = test_data_path.replace("test", "train")
                                train_base_path = os.path.dirname(train_base_path)  # Remove the test file name

                                training_folder = os.path.join(train_base_path, "entrenamientos")
                                                              
                                # Iterate through the algorithm folders (e.g., A2C)
                                for algorithm_folder_name in os.listdir(training_folder):
                                    algorithm_folder = os.path.join(training_folder, algorithm_folder_name)

                                    # Check if it's a directory
                                    if not os.path.isdir(algorithm_folder):
                                        continue
                                    if algorithm_folder_name == "A2C":
                                        # Iterate through the trained model subfolders
                                        for model_name in os.listdir(algorithm_folder):
                                            model_folder = os.path.join(algorithm_folder, model_name)

                                            # Check if it's a directory
                                            if not os.path.isdir(model_folder):
                                                continue

                                            model_path = os.path.join(model_folder, "model")

                                            # Check if the model exists
                                            if not os.path.exists(model_path + ".zip"):
                                                print(f"Model not found: {model_path}")
                                                continue

                                            test_model(model_path, test_data_path)

                # Si "no_filtrado" no está en la ruta, ir a la subcarpeta "wavelet"
                else:
                    wavelet_path = os.path.join(filter_path, "wavelet")

                    # Verificar si la carpeta "wavelet" existe
                    if not os.path.exists(wavelet_path):
                        continue

                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(wavelet_path):
                        window_path = os.path.join(wavelet_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        
                         # Probar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                test_data_path = os.path.join(window_path, filename)
                                # Construir la ruta al modelo entrenado
                                train_base_path = test_data_path.replace("test", "train")
                                train_base_path = os.path.dirname(train_base_path)  # Remove the test file name

                                training_folder = os.path.join(train_base_path, "entrenamientos")

                                # Check if the training folder exists
                                if not os.path.exists(training_folder):
                                    print(f"Training folder not found: {training_folder}")
                                    continue

                                # Iterate through the algorithm folders (e.g., A2C)
                                for algorithm_folder_name in os.listdir(training_folder):
                                    algorithm_folder = os.path.join(training_folder, algorithm_folder_name)

                                    # Check if it's a directory
                                    if not os.path.isdir(algorithm_folder):
                                        continue
                                    if algorithm_folder_name == "A2C":        
                                        # Iterate through the trained model subfolders
                                        for model_name in os.listdir(algorithm_folder):
                                            model_folder = os.path.join(algorithm_folder, model_name)

                                            # Check if it's a directory
                                            if not os.path.isdir(model_folder):
                                                continue

                                            model_path = os.path.join(model_folder, "model")

                                            # Check if the model exists
                                            if not os.path.exists(model_path + ".zip"):
                                                print(f"Model not found: {model_path}")
                                                continue

                                            test_model(model_path, test_data_path)


def train_ppo_model(train_data_path, base_path):
    """
    Entrena un modelo PPO con los datos de entrenamiento proporcionados.

    Args:
        train_data_path (str): Ruta al archivo CSV de datos de entrenamiento.
        base_path (str): Ruta base del proyecto para guardar el CSV de resultados.
    """
    # Cargar los datos de entrenamiento
    train_df = pd.read_csv(train_data_path)

    # Crear y verificar el entorno de entrenamiento
    train_env = StockTradingEnv(train_df, render_mode=None)
    check_env(train_env)

    # Vectorizar el entorno de entrenamiento
    vec_env = DummyVecEnv([lambda: train_env])

    # Definir rangos de hiperparámetros
    learning_rates = [0.0001, 0.0007, 0.001]  # Tasas de aprendizaje
    gammas = [0.95, 0.99]  # Factores de descuento
    n_steps_list = [64]  # Número de pasos antes de actualizar el modelo
    ent_coefs = [0.01]  # Coeficientes de la pérdida de entropía
    vf_coefs = [0.5]  # Coeficientes de la pérdida de la función de valor
    max_grad_norms = [0.5]  # Valores máximos para la normalización del gradiente
    gae_lambdas = [0.95]  # GAE lambda parameter
    batch_sizes = [64]  # Batch size

    # Extract information from the training data path
    train_data_filename = os.path.basename(train_data_path)
    is_filtered = "no_filtrado" not in train_data_path
    is_normalized = "no_normalizado" not in train_data_path

    # CSV file path
    csv_file = os.path.join(base_path, "training_results.csv")
    file_exists = os.path.isfile(csv_file)

    # Iterar a través de las combinaciones de hiperparámetros
    for learning_rate in learning_rates:
        for gamma in gammas:
            for n_steps in n_steps_list:
                for ent_coef in ent_coefs:
                    for vf_coef in vf_coefs:
                        for max_grad_norm in max_grad_norms:
                            for gae_lambda in gae_lambdas:
                                for batch_size in batch_sizes:
                                    # Definir el nombre del modelo
                                    model_name = f"ppo_lr{learning_rate}_gamma{gamma}_nsteps{n_steps}_ent{ent_coef}_vf{vf_coef}_gradnorm{max_grad_norm}_gae{gae_lambda}_batch{batch_size}"

                                    # Crear la carpeta para los entrenamientos
                                    training_folder = os.path.join(os.path.dirname(train_data_path), "entrenamientos", "PPO")
                                    os.makedirs(training_folder, exist_ok=True)

                                    # Crear la carpeta para el modelo entrenado
                                    model_folder = os.path.join(training_folder, model_name)
                                    os.makedirs(model_folder, exist_ok=True)

                                    model_path = os.path.join(model_folder, "model")

                                    print(f"Entrenando modelo: {model_name}")

                                    # Entrenar el modelo
                                    model = PPO('MlpPolicy', vec_env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps,
                                                ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, gae_lambda=gae_lambda,
                                                batch_size=batch_size, verbose=0, device='cpu')
                                    # # Entrenar el modelo
                                    # model = PPO('MlpPolicy',vec_env,verbose=0, device='cpu')
                                    model.learn(total_timesteps=10000)

                                    # Guardar el modelo
                                    model.save(model_path)
                                    print(f"Modelo guardado en: {model_path}")

                                    # Guardar las estadísticas de entrenamiento
                                    stats_path = os.path.join(model_folder, "stats.txt")
                                    with open(stats_path, "w") as f:
                                        f.write(f"Tasa de Aprendizaje: {learning_rate}\n")
                                        f.write(f"Gamma: {gamma}\n")
                                        f.write(f"N Pasos: {n_steps}\n")
                                        f.write(f"Ent Coef: {ent_coef}\n")
                                        f.write(f"VF Coef: {vf_coef}\n")
                                        f.write(f"Max Grad Norm: {max_grad_norm}\n")
                                        f.write(f"GAE Lambda: {gae_lambda}\n")
                                        f.write(f"Batch Size: {batch_size}\n")
                                    print(f"Estadísticas de entrenamiento guardadas en: {stats_path}")

                                    # Extract algorithm name from the model path
                                    algorithm_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

                                    # Use a more robust regex to extract parameters
                                    params = re.findall(r"([a-z]+)([0-9\.e-]+)", model_name)
                                    params_dict = {p[0]: p[1] for p in params}

                                    # Write to CSV
                                    with open(csv_file, mode='a', newline='') as f:
                                        writer = csv.writer(f)
                                        if not file_exists:
                                            writer.writerow([
                                                "Algorithm", "Learning Rate", "Gamma", "N Steps", "Ent Coef", "VF Coef", "Max Grad Norm", "GAE Lambda", "Batch Size",
                                                "Train Data", "Filtered", "Normalized"
                                            ])
                                            file_exists = True  # Ensure header is only written once

                                        writer.writerow([
                                            algorithm_name, params_dict.get("lr", ""), params_dict.get("gamma", ""), params_dict.get("nsteps", ""),
                                            params_dict.get("ent", ""), params_dict.get("vf", ""), params_dict.get("gradnorm", ""), params_dict.get("gae", ""), params_dict.get("batch", ""),
                                            train_data_filename, is_filtered, is_normalized
                                        ])

    print(f"Estadísticas de entrenamiento guardadas en: {csv_file}")

def test_ppo_model(model_path, test_data_path):
    """
    Prueba un modelo PPO con los datos de prueba proporcionados y genera un gráfico de las acciones tomadas.

    Args:
        model_path (str): Ruta al modelo entrenado.
        test_data_path (str): Ruta al archivo CSV de datos de prueba.
    """
    # Cargar los datos de prueba
    test_df = pd.read_csv(test_data_path)

    # Crear el entorno de prueba
    test_env = StockTradingEnv(test_df, render_mode=None)
    obs, _ = test_env.reset()
    net_worths = []
    actions = []

    # Cargar el modelo

    model = PPO.load(model_path, device="cpu")

    # Probar el modelo
    for i in range(len(test_df)):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = test_env.step(action)
        net_worths.append(test_env.net_worth)
        actions.append(action)

    # Guardar las estadísticas de prueba
    stats_path = os.path.dirname(model_path) + "/test_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Patrimonio Neto Final: {test_env.net_worth}\n")
    print(f"Estadísticas de prueba guardadas en: {stats_path}")

# Calcular el número de operaciones de compra y venta
    buy_operations = actions.count(1)
    sell_operations = actions.count(2)

    # Calcular el retorno general
    initial_net_worth = test_env.initial_balance
    final_net_worth = test_env.net_worth
    overall_return = (final_net_worth - initial_net_worth) / initial_net_worth

    # Obtener el nombre del modelo y los parámetros
    model_name = os.path.basename(os.path.dirname(model_path))
    params = re.findall(r"([a-z]+[-]?)([0-9\.e-]+)", model_name)
    params_dict = {p[0]: p[1] for p in params}
# Obtener información del conjunto de datos de prueba
    test_data_filename = os.path.basename(test_data_path)
    is_filtered = "no_filtrado" not in test_data_path
    is_normalized = "no_normalizado" not in test_data_path

   # Guardar las estadísticas de prueba en un archivo CSV
    csv_file = os.path.join(base_path, "testing_results.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Algorithm", "Learning Rate", "Gamma", "N Steps", "Ent Coef", "VF Coef", "Max Grad Norm", "GAE Lambda", "Batch Size",
                "Test Data", "Filtered", "Normalized",
                "Overall Return", "Final Net Worth", "Buy Operations", "Sell Operations"
            ])

        # Extract algorithm name from the model path
        algorithm_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

        # Use a more robust regex to extract parameters
        params = re.findall(r"([a-z]+)([0-9\.e-]+)", model_name)
        params_dict = {p[0]: p[1] for p in params}

        writer.writerow([
            algorithm_name, params_dict.get("lr", ""), params_dict.get("gamma", ""), params_dict.get("nsteps", ""),
            params_dict.get("ent", ""), params_dict.get("vf", ""), params_dict.get("gradnorm", ""),params_dict.get("gae", ""), params_dict.get("batch", ""),
            test_data_filename, is_filtered, is_normalized,
            overall_return, final_net_worth, buy_operations, sell_operations
        ])
    print(f"Estadísticas de prueba guardadas en: {csv_file}")
   





    # Rutina de visualización
    plt.figure(figsize=(10, 5))
    plt.plot(net_worths, label='Patrimonio Neto')
    plt.xlabel('Paso de Tiempo')
    plt.ylabel('Patrimonio Neto')
    plt.title(f'Patrimonio Neto a lo Largo del Tiempo - {os.path.basename(os.path.dirname(model_path))} (Prueba)')

    # Marcar las acciones en el gráfico
    close_prices = test_df['Close'].values
    buy_indices = np.where(np.array(actions) == 1)[0]
    sell_indices = np.where(np.array(actions) == 2)[0]

    plt.scatter(buy_indices, [net_worths[i] for i in buy_indices], marker='^', color='green', label='Compra')
    plt.scatter(sell_indices, [net_worths[i] for i in sell_indices], marker='v', color='red', label='Venta')

    plt.legend()
    plt.savefig(os.path.dirname(model_path) + "/net_worth_test.png")
    plt.close()

def process_ppo_testing_data(base_path):
    """
    Procesa los datos de prueba para PPO en la estructura de carpetas especificada.

    Args:
        base_path (str): Ruta base donde se encuentran las carpetas de datos de prueba.
    """
    # Iterar a través de las carpetas de divisas
    for currency_pair in os.listdir(base_path):
        currency_path = os.path.join(base_path, currency_pair)

        # Verificar si es una carpeta
        if not os.path.isdir(currency_path):
            continue

        # Iterar a través de las carpetas de timeframe
        for timeframe in os.listdir(currency_path):
            timeframe_path = os.path.join(currency_path, timeframe)

            # Verificar si es una carpeta
            if not os.path.isdir(timeframe_path):
                continue

            # Ir a la carpeta de datos de prueba
            data_path = os.path.join(timeframe_path, "test")

            # Verificar si la carpeta existe
            if not os.path.exists(data_path):
                continue

            # Iterar a través de las carpetas de filtrado
            for filter_type in os.listdir(data_path):
                filter_path = os.path.join(data_path, filter_type)

                # Verificar si es una carpeta
                if not os.path.isdir(filter_path):
                    continue

                # Si la palabra "no_filtrado" está en la ruta, la iteración debe comenzar en las subcarpetas
                if "no_filtrado" in filter_path:
                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(filter_path):
                        window_path = os.path.join(filter_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        # Probar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                test_data_path = os.path.join(window_path, filename)

                                # Construir la ruta al modelo entrenado
                                train_base_path = test_data_path.replace("test", "train")
                                train_base_path = os.path.dirname(train_base_path)  # Remove the test file name

                                training_folder = os.path.join(train_base_path, "entrenamientos", "PPO")

                                # Check if the training folder exists
                                if not os.path.exists(training_folder):
                                    print(f"Training folder not found: {training_folder}")
                                    continue
                                        
                                # Iterate through the trained model subfolders
                                for model_name in os.listdir(training_folder):
                                    model_folder = os.path.join(training_folder, model_name)

                                    # Check if it's a directory
                                    if not os.path.isdir(model_folder):
                                        continue

                                    model_path = os.path.join(model_folder, "model")

                                    # Check if the model exists
                                    if not os.path.exists(model_path + ".zip"):
                                        print(f"Model not found: {model_path}")
                                        continue

                                    test_ppo_model(model_path, test_data_path)

                # Si "no_filtrado" no está en la ruta, ir a la subcarpeta "wavelet"
                else:
                    wavelet_path = os.path.join(filter_path, "wavelet")

                    # Verificar si la carpeta "wavelet" existe
                    if not os.path.exists(wavelet_path):
                        continue

                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(wavelet_path):
                        window_path = os.path.join(wavelet_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        # Probar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                test_data_path = os.path.join(window_path, filename)
                                # Construir la ruta al modelo entrenado
                                train_base_path = test_data_path.replace("test", "train")
                                train_base_path = os.path.dirname(train_base_path)  # Remove the test file name

                                training_folder = os.path.join(train_base_path, "entrenamientos", "PPO")

                                # Check if the training folder exists
                                if not os.path.exists(training_folder):
                                    print(f"Training folder not found: {training_folder}")
                                    continue

                                # Iterate through the trained model subfolders
                                for model_name in os.listdir(training_folder):
                                    model_folder = os.path.join(training_folder, model_name)

                                    # Check if it's a directory
                                    if not os.path.isdir(model_folder):
                                        continue

                                    model_path = os.path.join(model_folder, "model")

                                    # Check if the model exists
                                    if not os.path.exists(model_path + ".zip"):
                                        print(f"Model not found: {model_path}")
                                        continue

                                    test_ppo_model(model_path, test_data_path)

def train_sac_model(train_data_path, base_path):
    """
    Entrena un modelo SAC con los datos de entrenamiento proporcionados.

    Args:
        train_data_path (str): Ruta al archivo CSV de datos de entrenamiento.
        base_path (str): Ruta base del proyecto para guardar el CSV de resultados.
    """
    # Cargar los datos de entrenamiento
    train_df = pd.read_csv(train_data_path)

    # Crear y verificar el entorno de entrenamiento
    train_env = StockTradingEnvSAC(train_df, render_mode=None)
    check_env(train_env)

    # Vectorizar el entorno de entrenamiento
    vec_env = DummyVecEnv([lambda: train_env])

    # # Definir rangos de hiperparámetros
    # learning_rates = [0.0001, 0.0007, 0.001]
    # gammas = [0.95, 0.99]
    # buffer_sizes = [10000, 20000]  # Tamaño del buffer de replay
    # batch_sizes = [64, 128]
    # ent_coefs = ['auto', 0.01, 0.001]  # Coeficiente de entropía
    # tau = [0.005, 0.01]  # Coeficiente de actualización suave para las redes objetivo


    # Definir rangos de hiperparámetros
    learning_rates = [0.0003]  # Default learning rate
    gammas = [0.99]  # Default gamma
    buffer_sizes = [10000]  # Default buffer size
    batch_sizes = [256]  # Default batch size
    ent_coefs = ['auto']  # Default entropy coefficient
    tau = [0.005,0.01]  # Default tau


    # Extract information from the training data path
    train_data_filename = os.path.basename(train_data_path)
    is_filtered = "no_filtrado" not in train_data_path
    is_normalized = "no_normalizado" not in train_data_path

    # CSV file path
    csv_file = os.path.join(base_path, "training_results.csv")
    file_exists = os.path.isfile(csv_file)

    # Iterar a través de las combinaciones de hiperparámetros
    for learning_rate in learning_rates:
        for gamma in gammas:
            for buffer_size in buffer_sizes:
                for batch_size in batch_sizes:
                    for ent_coef in [ent_coefs[0]]:  # Only use 'auto' for ent_coef
                        for tau_value in tau:
                            # Definir el nombre del modelo
                            model_name = f"sac_lr{learning_rate}_gamma{gamma}_buf{buffer_size}_batch{batch_size}_ent{ent_coef}_tau{tau_value}"

                            # Crear la carpeta para los entrenamientos
                            training_folder = os.path.join(os.path.dirname(train_data_path), "entrenamientos", "SAC")
                            os.makedirs(training_folder, exist_ok=True)

                            # Crear la carpeta para el modelo entrenado
                            model_folder = os.path.join(training_folder, model_name)
                            os.makedirs(model_folder, exist_ok=True)

                            model_path = os.path.join(model_folder, "model")

                            print(f"Entrenando modelo: {model_name}")

                            # Entrenar el modelo
                            model = SAC('MlpPolicy', vec_env, learning_rate=learning_rate, gamma=gamma,
                                        #replay_buffer_kwargs=dict(n_sampled_goal=5),
                                        buffer_size=buffer_size, batch_size=batch_size, ent_coef=ent_coef,
                                        tau=tau_value, verbose=0, device='cpu')
                            model.learn(total_timesteps=10000)

                            # Guardar el modelo
                            model.save(model_path)
                            print(f"Modelo guardado en: {model_path}")

                            # Guardar las estadísticas de entrenamiento
                            stats_path = os.path.join(model_folder, "stats.txt")
                            with open(stats_path, "w") as f:
                                f.write(f"Tasa de Aprendizaje: {learning_rate}\n")
                                f.write(f"Gamma: {gamma}\n")
                                f.write(f"Buffer Size: {buffer_size}\n")
                                f.write(f"Batch Size: {batch_size}\n")
                                f.write(f"Ent Coef: {ent_coef}\n")
                                f.write(f"Tau: {tau_value}\n")
                            print(f"Estadísticas de entrenamiento guardadas en: {stats_path}")

                            # Extract algorithm name from the model path
                            algorithm_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

                            # Use a more robust regex to extract parameters
                            params = re.findall(r"([a-z]+)([0-9\.e-]+)", model_name)
                            params_dict = {p[0]: p[1] for p in params}

                            # Write to CSV
                            with open(csv_file, mode='a', newline='') as f:
                                writer = csv.writer(f)
                                if not file_exists:
                                    writer.writerow([
                                        "Algorithm", "Learning Rate", "Gamma", "Buffer Size", "Batch Size", "Ent Coef", "Tau",
                                        "Train Data", "Filtered", "Normalized"
                                    ])
                                    file_exists = True  # Ensure header is only written once

                                writer.writerow([
                                    algorithm_name, params_dict.get("lr", ""), params_dict.get("gamma", ""),
                                    params_dict.get("buf", ""), params_dict.get("batch", ""), params_dict.get("ent", ""),
                                    params_dict.get("tau", ""), train_data_filename, is_filtered, is_normalized
                                ])

    print(f"Estadísticas de entrenamiento guardadas en: {csv_file}")

def test_sac_model(model_path, test_data_path):
    """
    Prueba un modelo SAC con los datos de prueba proporcionados y genera un gráfico de las acciones tomadas.

    Args:
        model_path (str): Ruta al modelo entrenado.
        test_data_path (str): Ruta al archivo CSV de datos de prueba.
    """
    # Cargar los datos de prueba
    test_df = pd.read_csv(test_data_path)

    # Crear el entorno de prueba
    test_env = StockTradingEnvSAC(test_df, render_mode=None)
    obs, _ = test_env.reset()
    net_worths = []
    actions = []

    # Cargar el modelo
    model = SAC.load(model_path, device="cpu")

    # Probar el modelo
    for i in range(len(test_df)):
        action, _states = model.predict(obs, deterministic=True)  # SAC uses continuous actions
        obs, rewards, terminated, truncated, info = test_env.step(action)
        net_worths.append(test_env.net_worth)
        actions.append(action)

    # Guardar las estadísticas de prueba
    stats_path = os.path.dirname(model_path) + "/test_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Patrimonio Neto Final: {test_env.net_worth}\n")
    print(f"Estadísticas de prueba guardadas en: {stats_path}")

    # Calcular el número de operaciones de compra y venta
    # (Adapt this part if SAC outputs continuous actions)
    buy_operations = sum(1 for a in actions if a > 0.5)  # Example threshold
    sell_operations = sum(1 for a in actions if a < -0.5) # Example threshold

    # Calcular el retorno general
    initial_net_worth = test_env.initial_balance
    final_net_worth = test_env.net_worth
    overall_return = (final_net_worth - initial_net_worth) / initial_net_worth

    # Obtener el nombre del modelo y los parámetros
    model_name = os.path.basename(os.path.dirname(model_path))
    params = re.findall(r"([a-z]+)([0-9\.e-]+)", model_name)
    params_dict = {p[0]: p[1] for p in params}

    # Obtener información del conjunto de datos de prueba
    test_data_filename = os.path.basename(test_data_path)
    is_filtered = "no_filtrado" not in test_data_path
    is_normalized = "no_normalizado" not in test_data_path

    # Guardar las estadísticas de prueba en un archivo CSV
    csv_file = os.path.join(base_path, "testing_results.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Algorithm", "Learning Rate", "Gamma", "Buffer Size", "Batch Size", "Ent Coef", "Tau",
                "Test Data", "Filtered", "Normalized",
                "Overall Return", "Final Net Worth", "Buy Operations", "Sell Operations"
            ])

        # Extract algorithm name from the model path
        algorithm_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

        # Use a more robust regex to extract parameters
        params = re.findall(r"([a-z]+)([0-9\.e-]+)", model_name)
        params_dict = {p[0]: p[1] for p in params}

        writer.writerow([
            algorithm_name, params_dict.get("lr", ""), params_dict.get("gamma", ""), params_dict.get("buf", ""),
            params_dict.get("batch", ""), params_dict.get("ent", ""), params_dict.get("tau", ""),
            test_data_filename, is_filtered, is_normalized,
            overall_return, final_net_worth, buy_operations, sell_operations
        ])
    print(f"Estadísticas de prueba guardadas en: {csv_file}")

    # Rutina de visualización
    plt.figure(figsize=(10, 5))
    plt.plot(net_worths, label='Patrimonio Neto')
    plt.xlabel('Paso de Tiempo')
    plt.ylabel('Patrimonio Neto')
    plt.title(f'Patrimonio Neto a lo Largo del Tiempo - {os.path.basename(os.path.dirname(model_path))} (Prueba)')

    # Marcar las acciones en el gráfico
    close_prices = test_df['Close'].values
    # Adapt this part to visualize continuous actions
    buy_indices = [i for i, a in enumerate(actions) if a > 0.5]
    sell_indices = [i for i, a in enumerate(actions) if a < -0.5]

    plt.scatter(buy_indices, [net_worths[i] for i in buy_indices], marker='^', color='green', label='Compra')
    plt.scatter(sell_indices, [net_worths[i] for i in sell_indices], marker='v', color='red', label='Venta')

    plt.legend()
    plt.savefig(os.path.dirname(model_path) + "/net_worth_test.png")
    plt.close()

def process_sac_testing_data(base_path):

    """
    Procesa los datos de prueba para SAC en la estructura de carpetas especificada.

    Args:
        base_path (str): Ruta base donde se encuentran las carpetas de datos de prueba.
    """
    # Iterar a través de las carpetas de divisas
    for currency_pair in os.listdir(base_path):
        currency_path = os.path.join(base_path, currency_pair)

        # Verificar si es una carpeta
        if not os.path.isdir(currency_path):
            continue

        # Iterar a través de las carpetas de timeframe
        for timeframe in os.listdir(currency_path):
            timeframe_path = os.path.join(currency_path, timeframe)

            # Verificar si es una carpeta
            if not os.path.isdir(timeframe_path):
                continue

            # Ir a la carpeta de datos de prueba
            data_path = os.path.join(timeframe_path, "test")

            # Verificar si la carpeta existe
            if not os.path.exists(data_path):
                continue

            # Iterar a través de las carpetas de filtrado
            for filter_type in os.listdir(data_path):
                filter_path = os.path.join(data_path, filter_type)

                # Verificar si es una carpeta
                if not os.path.isdir(filter_path):
                    continue

                # Si la palabra "no_filtrado" está en la ruta, la iteración debe comenzar en las subcarpetas
                if "no_filtrado" in filter_path:
                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(filter_path):
                        window_path = os.path.join(filter_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        # Probar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                test_data_path = os.path.join(window_path, filename)

                                # Construir la ruta al modelo entrenado
                                train_base_path = test_data_path.replace("test", "train")
                                train_base_path = os.path.dirname(train_base_path)  # Remove the test file name

                                training_folder = os.path.join(train_base_path, "entrenamientos", "SAC")

                                # Check if the training folder exists
                                if not os.path.exists(training_folder):
                                    print(f"Training folder not found: {training_folder}")
                                    continue

                                # Iterate through the trained model subfolders
                                for model_name in os.listdir(training_folder):
                                    model_folder = os.path.join(training_folder, model_name)

                                    # Check if it's a directory
                                    if not os.path.isdir(model_folder):
                                        continue

                                    model_path = os.path.join(model_folder, "model")

                                    # Check if the model exists
                                    if not os.path.exists(model_path + ".zip"):
                                        print(f"Model not found: {model_path}")
                                        continue

                                    test_sac_model(model_path, test_data_path)

                # Si "no_filtrado" no está en la ruta, ir a la subcarpeta "wavelet"
                else:
                    wavelet_path = os.path.join(filter_path, "wavelet")

                    # Verificar si la carpeta "wavelet" existe
                    if not os.path.exists(wavelet_path):
                        continue

                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(wavelet_path):
                        window_path = os.path.join(wavelet_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        # Probar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                test_data_path = os.path.join(window_path, filename)
                                # Construir la ruta al modelo entrenado
                                train_base_path = test_data_path.replace("test", "train")
                                train_base_path = os.path.dirname(train_base_path)  # Remove the test file name

                                training_folder = os.path.join(train_base_path, "entrenamientos", "SAC")

                                # Check if the training folder exists
                                if not os.path.exists(training_folder):
                                    print(f"Training folder not found: {training_folder}")
                                    continue

                                # Iterate through the trained model subfolders
                                for model_name in os.listdir(training_folder):
                                    model_folder = os.path.join(training_folder, model_name)

                                    # Check if it's a directory
                                    if not os.path.isdir(model_folder):
                                        continue

                                    model_path = os.path.join(model_folder, "model")

                                    # Check if the model exists
                                    if not os.path.exists(model_path + ".zip"):
                                        print(f"Model not found: {model_path}")
                                        continue

                                    test_sac_model(model_path, test_data_path)

def process_data_sac_training_data(base_path):
    """
    Procesa los datos de entrenamiento para SAC en la estructura de carpetas especificada.

    Args:
        base_path (str): Ruta base donde se encuentran las carpetas de datos de entrenamiento.
    """
    # Iterar a través de las carpetas de divisas
    for currency_pair in os.listdir(base_path):
        currency_path = os.path.join(base_path, currency_pair)

        # Verificar si es una carpeta
        if not os.path.isdir(currency_path):
            continue

        # Iterar a través de las carpetas de timeframe
        for timeframe in os.listdir(currency_path):
            timeframe_path = os.path.join(currency_path, timeframe)

            # Verificar si es una carpeta
            if not os.path.isdir(timeframe_path):
                continue

            # Ir a la carpeta de datos de entrenamiento
            data_path = os.path.join(timeframe_path, "train")

            # Verificar si la carpeta existe
            if not os.path.exists(data_path):
                continue

            # Iterar a través de las carpetas de filtrado
            for filter_type in os.listdir(data_path):
                filter_path = os.path.join(data_path, filter_type)

                # Verificar si es una carpeta
                if not os.path.isdir(filter_path):
                    continue

                # Si la palabra "no_filtrado" está en la ruta, la iteración debe comenzar en las subcarpetas
                if "no_filtrado" in filter_path:
                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(filter_path):
                        window_path = os.path.join(filter_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        # Entrenar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                train_data_path = os.path.join(window_path, filename)
                                train_sac_model(train_data_path, base_path)

                # Si "no_filtrado" no está en la ruta, ir a la subcarpeta "wavelet"
                else:
                    wavelet_path = os.path.join(filter_path, "wavelet")

                    # Verificar si la carpeta "wavelet" existe
                    if not os.path.exists(wavelet_path):
                        continue

                    # Iterar a través de las carpetas de ventana
                    for window_size in os.listdir(wavelet_path):
                        window_path = os.path.join(wavelet_path, window_size)

                        # Verificar si es una carpeta
                        if not os.path.isdir(window_path):
                            continue

                        # Entrenar el modelo
                        for filename in os.listdir(window_path):
                            if filename.endswith(".csv"):
                                train_data_path = os.path.join(window_path, filename)
                                train_sac_model(train_data_path, base_path)

def process_data(base_path):
    """
    Procesa los datos de entrenamiento y prueba en la estructura de carpetas especificada.

    Args:
        base_path (str): Ruta base donde se encuentran las carpetas de datos.
    """
    process_data_sac_training_data(base_path)
    process_sac_testing_data(base_path)
    process_training_data(base_path)
    
    process_testing_data(base_path)
    #process_data_ppo_training_data(base_path)
    process_ppo_testing_data(base_path)

    
# Ejemplo de uso
# Reemplazar 'ruta/a/tus/datos_de_entrenamiento.csv' y 'ruta/a/tus/datos_de_prueba.csv'
# con las rutas reales a tus datos de entrenamiento y prueba
base_path = 'processed_data'
process_data(base_path)