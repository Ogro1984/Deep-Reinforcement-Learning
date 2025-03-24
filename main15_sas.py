import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym  # Use gymnasium instead of gym
from gymnasium.spaces import Discrete, Box  # Use gymnasium spaces
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from gymnasium import spaces
import re
import csv

# Custom callback to print training statistics
class PrintTrainingStatisticsCallback(BaseCallback):
    def __init__(self, model_path, verbose=0):
        super(PrintTrainingStatisticsCallback, self).__init__(verbose)
        self.csv_file = model_path + "_training_stats.csv"
        self.file_exists = os.path.isfile(self.csv_file)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:  # Print every 100 steps
            stats = {
                "Step": self.n_calls,
                "Actor Loss": self.model.logger.name_to_value['train/actor_loss'],
                "Critic Loss": self.model.logger.name_to_value['train/critic_loss'],
                "Entropy Coef": self.model.logger.name_to_value['train/ent_coef'],
                "Entropy Coef Loss": self.model.logger.name_to_value['train/ent_coef_loss']
            }
            # print(f"Step: {stats['Step']}")
            # print(f"  Loss: {stats['Loss']}")
            # print(f"  Value Loss: {stats['Value Loss']}")
            # print(f"  Policy Loss: {stats['Policy Loss']}")
            # print(f"  Entropy: {stats['Entropy']}")

            # Save statistics to CSV
            with open(self.csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not self.file_exists:
                    writer.writerow(stats.keys())
                    self.file_exists = True
                writer.writerow(stats.values())
        return True



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
        self.observation_space = Box(low=0, high=1, shape=(9,), dtype=np.float32)  # Espacio de observaciones
        self.render_mode = render_mode  # Modo de renderización
        self.action_history = []  # Historial de acciones
        #self.observation_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)

    # Función para generar la siguiente observación
    def _next_observation(self):
        frame = np.array([
             self.df.iloc[self.current_step]['Close'],  # Precio de cierre escalado
            self.df.iloc[self.current_step]['Volume'],# Volumen escalado
            self.df.iloc[self.current_step]['Dif_a_la_Media_200'],  # Precio de cierre escalado
            self.df.iloc[self.current_step]['Cambio_Porcentual'],
            self.df.iloc[self.current_step]['Close_Filtrado_Wavelet'],
            self.df.iloc[self.current_step]['Volumen_Relativo_200'],
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
            # Calcular la recompensa basada en el cambio de patrimonio neto
            reward = (self.net_worth - self.initial_balance) / self.initial_balance

            # Penalización por inactividad (mantener)
            if action[0] == 0:
                reward -= 0.01

            # Recompensa adicional por operaciones exitosas
            if action[0] > 0 and self.net_worth > self.initial_balance:
                reward += 0.1
            elif action[0] < 0 and self.net_worth > self.initial_balance:
                reward += 0.1

            # Penalización por riesgo (volatilidad del patrimonio neto)
            if self.net_worth < self.initial_balance:
                reward -= 0.1
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

def train_sac_model(train_data_path, base_path):
    """
    Entrena un modelo SAC con los datos de entrenamiento proporcionados.

    Args:
        train_data_path (str): Ruta al archivo CSV de datos de entrenamiento.
        base_path (str): Ruta base del proyecto para guardar el CSV de resultados.
    """
    
    for nombre_archivo in os.listdir(base_path):
        ruta_archivo = os.path.join(base_path, nombre_archivo)
        if (nombre_archivo.startswith("EUR") or nombre_archivo.startswith("GBP")):           
        # Leer el archivo CSV en un DataFrame, usando ';' como separador
            try:
                train_df = pd.read_csv(ruta_archivo, sep=';')
            except Exception as e:
                print(f"Error al leer el archivo {nombre_archivo}: {e}")
                continue
            # Cargar los datos de entrenamiento
    
            # Cargar los datos de entrenamiento
            #train_df = pd.read_csv(train_data_path)

            # Crear y verificar el entorno de entrenamiento
            train_env = StockTradingEnvSAC(train_df, render_mode=True)
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
            #tau = [0.01]

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
                                                tau=tau_value, verbose=1, device='cuda')
                                    callback = PrintTrainingStatisticsCallback(model_path)
                                    model.learn(total_timesteps=6000, callback=callback)
                                    

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

            #print(f"Estadísticas de entrenamiento guardadas en: {csv_file}")

def test_sac_model(model_path, test_data_path):
    """
    Prueba un modelo SAC con los datos de prueba proporcionados y genera un gráfico de las acciones tomadas.

    Args:
        model_path (str): Ruta al modelo entrenado.
        test_data_path (str): Ruta al archivo CSV de datos de prueba.
    """
    # Cargar los datos de prueba
    test_df = pd.read_csv(test_data_path, sep=';')

    # Crear el entorno de prueba
    test_env = StockTradingEnvSAC(test_df, render_mode=None)
    obs, _ = test_env.reset(seed=42)
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
    plt.figure(figsize=(12, 8))
    # Subplot 1: Valores de Close y operaciones
    plt.subplot(2, 1, 1)
    plt.plot(test_df['Close'].values, label='Close')
    plt.xlabel('Paso de Tiempo')
    plt.ylabel('Precio de Cierre')
    plt.title(f'Precio de Cierre y Operaciones - {os.path.basename(os.path.dirname(model_path))} (Prueba)')

    # Marcar las acciones en el gráfico
    buy_indices = np.where(np.array(actions) == 1)[0]
    sell_indices = np.where(np.array(actions) == 2)[0]

    plt.scatter(buy_indices, test_df['Close'].values[buy_indices], marker='^', color='green', label='Compra')
    plt.scatter(sell_indices, test_df['Close'].values[sell_indices], marker='v', color='red', label='Venta')

    plt.legend()

    # Subplot 2: Valor de la cartera
    plt.subplot(2, 1, 2)
    plt.plot(net_worths, label='Patrimonio Neto')
    plt.xlabel('Paso de Tiempo')
    plt.ylabel('Patrimonio Neto')
    plt.title(f'Patrimonio Neto a lo Largo del Tiempo - {os.path.basename(os.path.dirname(model_path))} (Prueba)')

    plt.legend()

    # Guardar el gráfico
    plt.tight_layout()
    plt.savefig(os.path.dirname(model_path) + "/net_worth_test.png")
    plt.close()


    
base_path = 'fxtrainm\GBPUSD'
#model_path = 'modelos entrenados para test\PPO\ppo_lr0.0001_gamma0.99_nsteps2048_ent0.01_vf0.5_gradnorm0.5_gae0.95_batch128\model.zip'
#model_path = 'modelos entrenados para test\PPO\ppo_lr0.0001_gamma0.99_nsteps2048_ent0.01_vf0.5_gradnorm0.5_gae0.95_batch128_2domejor\model.zip'
#model_path = 'modelos entrenados para test\PPO\ppo_lr0.0003_gamma0.99_nsteps2048_ent0.01_vf0.5_gradnorm0.5_gae0.95_batch128_bueno\model.zip'
#model_path = 'modelos entrenados para test\PPO\ppo_lr0.0007_gamma0.95_nsteps2048_ent0.01_vf0.5_gradnorm0.5_gae0.95_batch128\model.zip'
#model_path = 'modelos entrenados para test\PPO_eurjpy_old_2\ppo_lr0.0001_gamma0.99_nsteps64_ent0.01_vf0.5_gradnorm0.5_gae0.95_batch128\model.zip'
#model_path = 'modelos entrenados para test\PPO_eurjpy_old_2\ppo_lr0.0003_gamma0.99_nsteps64_ent0.01_vf0.5_gradnorm0.5_gae0.95_batch128\model.zip'
model_path = 'modelos entrenados\modelos entrenados para test sac eurjpy d\sac_lr0.0003_gamma0.99_buf10000_batch256_entauto_tau0.01\model.zip'
model_path = 'modelos entrenados\modelos entrenados para test sac eurusd d\sac_lr0.0003_gamma0.99_buf10000_batch256_entauto_tau0.005\model.zip'

test_data_path = 'fxtestd\EURUSD_D1'
#train_sac_model(base_path, base_path)  # Train PPO

test_sac_model(model_path, test_data_path)  # Test PPO
