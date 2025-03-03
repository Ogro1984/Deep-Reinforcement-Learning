import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym  # Use gymnasium instead of gym
from gymnasium.spaces import Discrete, Box  # Use gymnasium spaces
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

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

# Función para entrenar y evaluar el modelo A2C
def train_and_evaluate_model(train_data_path, test_data_path):
    """
    Entrena y evalúa un modelo A2C con los datos de entrenamiento y prueba proporcionados.

    Args:
        train_data_path (str): Ruta al archivo CSV de datos de entrenamiento.
        test_data_path (str): Ruta al archivo CSV de datos de prueba.
    """
    # Cargar los datos de entrenamiento y prueba
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

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
                                model_path = os.path.dirname(train_data_path) + '/' + model_name

                                print(f"Entrenando modelo: {model_name}")

                                # Entrenar el modelo
                                model = A2C('MlpPolicy', vec_env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps,
                                            ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, rms_prop_eps=rms_prop_eps,
                                            verbose=0, device='cpu')
                                model.learn(total_timesteps=10000)

                                # Probar el modelo
                                test_env = StockTradingEnv(test_df, render_mode=None)
                                obs, _ = test_env.reset()
                                net_worths = []
                                for i in range(len(test_df)):
                                    action, _states = model.predict(obs)
                                    obs, rewards, terminated, truncated, info = test_env.step(action)
                                    net_worths.append(test_env.net_worth)

                                # Guardar las estadísticas de entrenamiento
                                stats_path = os.path.dirname(train_data_path) + '/' + f"{model_name}_stats.txt"
                                with open(stats_path, "w") as f:
                                    f.write(f"Tasa de Aprendizaje: {learning_rate}\n")
                                    f.write(f"Gamma: {gamma}\n")
                                    f.write(f"N Pasos: {n_steps}\n")
                                    f.write(f"Ent Coef: {ent_coef}\n")
                                    f.write(f"VF Coef: {vf_coef}\n")
                                    f.write(f"Max Grad Norm: {max_grad_norm}\n")
                                    f.write(f"RMS Prop Eps: {rms_prop_eps}\n")
                                    f.write(f"Patrimonio Neto Final: {test_env.net_worth}\n")
                                print(f"Estadísticas de entrenamiento guardadas en: {stats_path}")

                                # Rutina de visualización
                                plt.figure(figsize=(10, 5))
                                plt.plot(net_worths, label='Patrimonio Neto')
                                plt.xlabel('Paso de Tiempo')
                                plt.ylabel('Patrimonio Neto')
                                plt.title(f'Patrimonio Neto a lo Largo del Tiempo - {model_name}')
                                plt.legend()
                                plt.savefig(os.path.dirname(train_data_path) + '/' + f"{model_name}_net_worth.png")
                                plt.close()

# Ejemplo de uso
# Reemplazar 'ruta/a/tus/datos_de_entrenamiento.csv' y 'ruta/a/tus/datos_de_prueba.csv'
# con las rutas reales a tus datos de entrenamiento y prueba
train_data_path = 'processed_data\\GBPUSD_D1\D1\\train\\no_filtrado_no_normalizado\\10\\10.csv'
test_data_path = 'processed_data\\GBPUSD_D1\D1\\test\\no_filtrado_no_normalizado\\10\\10.csv'
train_and_evaluate_model(train_data_path, test_data_path)