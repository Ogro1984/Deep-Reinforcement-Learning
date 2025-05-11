if __name__ == '__main__':
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
    from stable_baselines3.common.callbacks import BaseCallback
    #from optuna.integration import PyTorchLightningPruningCallback
    import os
    os.environ["OMP_NUM_THREADS"] = "4"
    from stable_baselines3.common.vec_env import SubprocVecEnv


    class OptunaCallback(BaseCallback):
        """
        Custom callback for integrating Optuna with Stable-Baselines3.
        This callback reports intermediate results to Optuna and handles pruning.
        """
        def __init__(self, trial, train_df,initial_balance,eval_env, sample_size=1000, verbose=0):
            super(OptunaCallback, self).__init__(verbose)
            self.trial = trial
            self.eval_env = eval_env
            self.sample_size = sample_size
            self.train_df = train_df
            self.initial_balance = initial_balance
        def _on_step(self) -> bool:
            # Evaluate the model every 1000 steps
            if self.n_calls % 1000 == 0:
                metrics = evaluate_model_min(self.model, self.eval_env,self.train_df, self.sample_size)
                # combined_metric = (
                #     # metrics['Profit Factor'] * 0.1 +
                #     # metrics['Win Rate'] * 0.1 +
                #     metrics['Profit Factor']*0.2 -
                #     metrics['Max Drawdown'] 
                # )
                net_worth_ratio = metrics['Final Balance'] / self.initial_balance
                # Report the combined metric to Optuna
                self.trial.report(net_worth_ratio, self.n_calls)

                # Prune the trial if needed
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return True

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
                    "Loss": self.model.logger.name_to_value['train/loss'],
                    "Value Loss": self.model.logger.name_to_value['train/value_loss'],
                    "Policy Loss": self.model.logger.name_to_value['train/policy_loss'],
                    "Entropy": self.model.logger.name_to_value['train/entropy_loss']
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

    # Entorno personalizado para el trading de acciones
    class StockTradingEnv15min(gym.Env):
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
            #self.observation_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)  # Espacio de observaciones
            self.render_mode = render_mode  # Modo de renderización
            self.action_history = []  # Historial de acciones
            self.observation_space = Box(low=0, high=1, shape=(9,), dtype=np.float32)

        # Función para generar la siguiente observación
        def _next_observation(self):
            frame = np.array([
                self.df.iloc[self.current_step]['Close'],  # Precio de cierre escalado
                self.df.iloc[self.current_step]['Volume'],# Volumen escalado
                self.df.iloc[self.current_step]['diff_to_mean'],  # Precio de cierre escalado
                self.df.iloc[self.current_step]['mean_volume'],  # Precio de cierre escalado
                self.df.iloc[self.current_step]['Close_Filtrado_Wavelet'],
                self.df.iloc[self.current_step]['ATR'],
                self.balance / self.initial_balance,  # Balance relativo al balance inicial
                self.shares_held / 100,  # Acciones en posesión escaladas
                self.net_worth / self.initial_balance,  # Patrimonio neto relativo al balance inicial
            ], dtype=np.float32)
            return frame

        def _take_action(self, action):
            current_price = self.df.iloc[self.current_step]['Close']  # Precio actual
            trade_quantity = self.shares_per_step  # Cantidad a transar
            cost = trade_quantity * current_price * (1 + self.commission)  # Costo de la transacción

            trade = {
                'entry_price': None,
                'exit_price': None,
                'profit': 0,
                'adverse_excursion': 0,
                'favorable_excursion': 0
            }

            if action == 1:  # Comprar
                if self.balance >= cost:  # Verificar si hay suficiente balance
                    self.balance -= cost  # Reducir el balance
                    self.shares_held += trade_quantity  # Aumentar las acciones en posesión
                    trade['entry_price'] = current_price

            elif action == 2:  # Vender
                if self.shares_held >= trade_quantity:  # Verificar si hay suficientes acciones
                    self.balance += trade_quantity * current_price * (1 - self.commission)  # Aumentar el balance
                    self.shares_held -= trade_quantity  # Reducir las acciones en posesión
                    trade['exit_price'] = current_price
                    trade['profit'] = (current_price - trade['entry_price']) * trade_quantity if trade['entry_price'] else 0

            # Actualizar el patrimonio neto
            self.net_worth = self.balance + self.shares_held * current_price

            # Registrar la operación
            trade['adverse_excursion'] = min(0, current_price - (trade['entry_price'] or current_price))
            trade['favorable_excursion'] = max(0, current_price - (trade['entry_price'] or current_price))
            self.action_history.append(trade)

        # Función para realizar un paso en el entorno
        def step(self, action):
            terminated = self.current_step >= len(self.df) - 1  # Verificar si el episodio ha terminado
            # print (self.current_step)
            # print (len(self.df))
            truncated = False  # No se utiliza en este entorno

            if not terminated:
                self.current_step += 1  # Avanzar al siguiente paso
                self._take_action(action)  # Realizar la acción
                obs = self._next_observation()  # Obtener la siguiente observación

                # recompensa basada en el cambio de patrimonio neto
                reward = (self.net_worth - self.initial_balance) / self.initial_balance
                #print(f"Reward1: {reward}")
                
                # # Penalización por inactividad (mantener)
                if action == 0:
                    reward -= 0.001

                # Recompensa adicional por operaciones exitosas
                if action == 1 and self.net_worth > self.initial_balance:
                    reward += 0.1
                elif action == 2 and self.net_worth > self.initial_balance:
                    reward += 0.1

                # # Penalización por riesgo (volatilidad del patrimonio neto)
                # if self.net_worth < self.initial_balance:
                #     reward -= 0.1
            else:
                obs = self._next_observation()  # Obtener la observación final
                reward = 0  # Recompensa cero al final del episodio

            #print(f"Reward2: {reward}")
            # Registrar la acción como un diccionario
            trade = {
                'step': self.current_step,
                'action': action,
                'price': self.df.iloc[self.current_step]['Close'],
                'entry_price': None,
                'exit_price': None,
                'profit': 0,
                'adverse_excursion': 0,
                'favorable_excursion': 0
            }
            self.action_history.append(trade)



            info = {
                'step': self.current_step,
                'balance': self.balance,
                'shares_held': self.shares_held,
                'net_worth': self.net_worth,
                'trades': self.action_history  # Pasar el historial de acciones como parte de la información
                }
            return obs, reward, terminated, truncated, info  # Devolver los resultados


        # Función para renderizar el entorno (opcional)
        def render(self, mode='human'):
            if self.render_mode is not None:
                print(f'Step: {self.current_step}')
                print(f'Balance: {self.balance}')
                print(f'Shares held: {self.shares_held}')
                print(f'Net worth: {self.net_worth}')

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.balance = self.initial_balance  # Resetear el balance
            self.net_worth = self.initial_balance  # Resetear el patrimonio neto
            self.shares_held = 0  # Resetear las acciones en posesión
            self.current_step = 0  # Resetear el paso actual
            self.action_history = []  # Resetear el historial de acciones
            obs = self._next_observation()  # Obtener la observación inicial
            info = {}  # Información adicional
            return obs, info  # Devolver la observación y la información

    def objective(trial,train_df):
        """
        Entrena un modelo PPO con los datos de entrenamiento proporcionados.

        Args:
            train_data_path (str): Ruta al archivo CSV de datos de entrenamiento.
            base_path (str): Ruta base del proyecto para guardar el CSV de resultados.
        """
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float('gamma', 0.8, 0.99)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
        vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)
        gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
        n_steps = trial.suggest_int('n_steps', 5, 2048, log=True)
        
        train_env = StockTradingEnv15min(train_df)
        vec_env = DummyVecEnv([lambda: train_env])

        # Number of parallel environments
        num_envs = 6  # Adjust based on the number of CPU cores available

        # Create parallel environments
        def make_env():
            return StockTradingEnv15min(train_df)

        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])



        model = A2C('MlpPolicy', vec_env, learning_rate=learning_rate, gamma=gamma,
                    ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    gae_lambda=gae_lambda, n_steps =n_steps, verbose=1,device="cpu")

        #model.learn(total_timesteps=1000)

        # # Entrenar el modelo con un callback de pruning
        # try:
        #     model.learn(total_timesteps=2000, callback=OptunaCallback(trial,initial_balance=train_env.initial_balance, eval_env=vec_env,train_df=train_df, sample_size=1000))
        # except optuna.exceptions.TrialPruned:
        #     raise optuna.exceptions.TrialPruned()
        
        #model.learn(total_timesteps=5000)
        # Train the model with Optuna callback
        # Split train_df into training and evaluation subsets
        from sklearn.model_selection import train_test_split
        train_df1, eval_data = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=False)

        try:
            print("punga 1")
        
            model.learn(total_timesteps=5000, callback=OptunaCallback(
                trial, eval_env=vec_env, train_df=train_df1, initial_balance=10000, sample_size=500))
            print("punga 2")
        except optuna.exceptions.TrialPruned:
            raise optuna.exceptions.TrialPruned()
      

        metrics = evaluate_model_min(model, vec_env, eval_data,sample_size=2000)
        print ("punga 4")
        
        
        
        
        # # Métrica combinada
        # combined_metric = (
        #     metrics['Sharpe Ratio'] +
        #     metrics['Sortino Ratio'] +
        #     metrics['Calmar Ratio'] +
        #     metrics['Profit Factor'] +
        #     metrics['Win Rate'] +
        #     metrics['Expectancy'] -
        #     metrics['Max Drawdown'] -
        #     metrics['Maximum Adverse Excursion (MAE)'] -
        #     metrics['Maximum Favorable Excursion (MFE)'] -
        #     metrics['Ulcer Index']
        # )

        # Métrica combinada para optimización
        # combined_metric = (
        #     #metrics['Profit Factor'] +  # Ponderar más el Profit Factor
        #     #metrics['Win Rate'] * 0.1 +      # Ponderar la tasa de éxito
        #     metrics['Profit Factor']*0.2- # Ponderar el beneficio final
        #     metrics['Max Drawdown']     # Penalizar el drawdown
        # )

        combined_metric = metrics['Final Balance'] / train_env.initial_balance

        # Manejar casos donde la métrica combinada sea infinita o no válida
        if not np.isfinite(combined_metric):
            combined_metric = -float('inf')

        return combined_metric
    

    def evaluate_model(model, vec_env, train_df):
        """
        Evalúa un modelo A2C utilizando métricas financieras como máxima ganancia, mínimo drawdown,
        Sharpe Ratio, Sortino Ratio, Calmar Ratio, Profit Factor, Win Rate, Expectancy, MAE, MFE, y Ulcer Index.

        Args:
            model (A2C): Modelo entrenado.
            vec_env (DummyVecEnv): Entorno vectorizado.
            train_df (pd.DataFrame): DataFrame de entrenamiento.

        Returns:
            dict: Métricas financieras calculadas.
        """
        net_worths = []
        trades = []  # Lista para registrar las operaciones
        obs = vec_env.reset()  # Usar vec_env para resetear

        for _ in range(len(train_df)):
            action, _ = model.predict(obs)
            obs, reward, done, info = vec_env.step(action)  # Usar vec_env para realizar pasos
            net_worths.append(vec_env.envs[0].net_worth)  # Registrar el patrimonio neto

            # Registrar las operaciones realizadas
            if 'trades' in info[0]:
                trades.extend(info[0]['trades'])

            if done:
                break

        # Calcular métricas financieras
        metrics = calculate_financial_metrics(net_worths, trades)
        # Agregar el balance final a las métricas
        metrics['Final Balance'] = vec_env.envs[0].balance
        return metrics

    def evaluate_model_min(model, vec_env, train_df, sample_size=3000):
        """
        Optimized evaluation function for faster performance.
        """
        net_worths = []
        trades = []
        total_steps = min(len(train_df), sample_size)
        obs = vec_env.reset()
        current_net_worth = [10000] * vec_env.num_envs  # Replace with actual initial balance

        for step in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)

            # Update net worths based on the info dictionary
            for i, info in enumerate(infos):
                current_net_worth[i] = info.get('net_worth', current_net_worth[i])

            net_worths.append(np.mean(current_net_worth))

            # Collect trades only if necessary
            for info in infos:
                if 'trades' in info:
                    trades.extend(info['trades'])

            if all(dones):
                break

        # Calculate metrics only once at the end
        metrics = calculate_financial_metrics(net_worths, trades)
        metrics['Final Balance'] = np.mean(current_net_worth)
        return metrics


    def evaluate_model_min_2(model, vec_env, train_df,sample_size=3000):
        """
        Evalúa un modelo A2C utilizando métricas financieras como máxima ganancia, mínimo drawdown,
        Sharpe Ratio, Sortino Ratio, Calmar Ratio, Profit Factor, Win Rate, Expectancy, MAE, MFE, y Ulcer Index.

        Args:
            model (A2C): Modelo entrenado.
            vec_env (DummyVecEnv): Entorno vectorizado.
            train_df (pd.DataFrame): DataFrame de entrenamiento.

        Returns:
            dict: Métricas financieras calculadas.
        """
        net_worths = []
        trades = []  # Lista para registrar las operaciones
    
        total_steps = min(len(train_df), sample_size)
        obs = vec_env.reset()  # Usar vec_env para resetear
        current_net_worth = [env.initial_balance for env in vec_env.remotes]  # Inicializar el patrimonio neto
        for step in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)  # Usar predicción determinista para consistencia
            obs, reward, dones, infos = vec_env.step(action)  # Usar vec_env para realizar pasos
                # Actualizar el patrimonio neto basado en las recompensas
            for i, info in enumerate(infos):
                current_net_worth[i] = info.get('net_worth', current_net_worth[i])

            net_worths.append(np.mean(current_net_worth))  # Registrar el promedio del patrimonio neto

            # Registrar las operaciones realizadas
            for info in infos:
                if 'trades' in info:
                    trades.extend(info['trades'])

            if all(dones):  # Si todos los entornos han terminado
                break
        
        # Calcular métricas financieras
        metrics = calculate_financial_metrics(net_worths, trades)
        # Agregar el balance final a las métricas
        metrics['Final Balance'] = np.mean(current_net_worth)  # Usar el promedio del patrimonio neto final
        return metrics

    def evaluate_model_min_3(model, vec_env, train_df, sample_size=3000):
        """
        Evalúa un modelo A2C utilizando métricas financieras como máxima ganancia, mínimo drawdown,
        Sharpe Ratio, Sortino Ratio, Calmar Ratio, Profit Factor, Win Rate, Expectancy, MAE, MFE, y Ulcer Index.

        Args:
            model (A2C): Modelo entrenado.
            vec_env (SubprocVecEnv): Entorno vectorizado.
            train_df (pd.DataFrame): DataFrame de entrenamiento.

        Returns:
            dict: Métricas financieras calculadas.
        """
        net_worths = []
        trades = []  # Lista para registrar las operaciones
        
        total_steps = min(len(train_df), sample_size)
        obs = vec_env.reset()  # Usar vec_env para resetear

        # Initialize net worths using the initial balance from the first environment
        initial_balance = 10000  # Replace with the actual initial balance used in your environment
        current_net_worth = [initial_balance] * vec_env.num_envs
        lock=0
        for step in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)  # Usar predicción determinista para consistencia
            obs, rewards, dones, infos = vec_env.step(action)  # Usar vec_env para realizar pasos

            # Update net worths based on the info dictionary
            for i, info in enumerate(infos):
                current_net_worth[i] = info.get('net_worth', current_net_worth[i])

            net_worths.append(np.mean(current_net_worth))  # Registrar el promedio del patrimonio neto

            # Registrar las operaciones realizadas
            for info in infos:
                if 'trades' in info:
                    trades.extend(info['trades'])
                # if info['action'] == 1:
                #  print("Acción de compra")
                # elif info['action'] == 2:
                #     print("nadaa")
                # if lock==0:
                #         print(info)
                #         lock=1
            # Check if all environments are done
            if all(dones):  # Si todos los entornos han terminado
                break
           
            
            
            
        # Calcular métricas financieras
        metrics = calculate_financial_metrics(net_worths, trades)
        # Agregar el balance final a las métricas
        metrics['Final Balance'] = np.mean(current_net_worth)  # Usar el promedio del patrimonio neto final
        return metrics


    def calculate_financial_metrics(net_worths, trades):
        """
        Calcula métricas financieras como Drawdown, Sharpe Ratio, Sortino Ratio, Calmar Ratio,
        Profit Factor, Win Rate, Expectancy, MAE, MFE, y Ulcer Index.

        Args:
            net_worths (list): Lista de valores de patrimonio neto.
            trades (list): Lista de operaciones realizadas, cada una como un diccionario con claves:
                        'entry_price', 'exit_price', 'profit', 'adverse_excursion', 'favorable_excursion'.

        Returns:
            dict: Diccionario con las métricas calculadas.
        """
        epsilon = 1e-8  # Small value to prevent division by zero
        #returns = np.diff(net_worths) / (net_worths[:-1] + epsilon)
        returns = np.diff(net_worths) / net_worths[:-1]
    

        # Drawdown
        peak = np.maximum.accumulate(net_worths)
        drawdown = (peak - net_worths) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Sharpe Ratio
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = np.mean(returns) / np.std(downside_returns) if np.std(downside_returns) != 0 else 0

        # Calmar Ratio
        calmar_ratio = (np.mean(returns) * 252) / max_drawdown if max_drawdown != 0 else 0

        # Profit Factor
        total_profit = sum([trade['profit'] for trade in trades if trade['profit'] > 0])
        total_loss = abs(sum([trade['profit'] for trade in trades if trade['profit'] < 0]))
        profit_factor = total_profit / total_loss if total_loss != 0 else 0

        # Win Rate
        winning_trades = len([trade for trade in trades if trade['profit'] > 0])
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Expectancy
        expectancy = np.mean([trade['profit'] for trade in trades]) if total_trades > 0 else 0

        # Maximum Adverse Excursion (MAE)
        mae = max([trade['adverse_excursion'] for trade in trades]) if trades else 0

        # Maximum Favorable Excursion (MFE)
        mfe = max([trade['favorable_excursion'] for trade in trades]) if trades else 0

        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown ** 2)) if len(drawdown) > 0 else 0

        return {
            'returns': returns,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Profit Factor': profit_factor,
            'Win Rate': win_rate,
            'Expectancy': expectancy,
            'Maximum Adverse Excursion (MAE)': mae,
            'Maximum Favorable Excursion (MFE)': mfe,
            'Ulcer Index': ulcer_index
        }

    # Optimizar los hiperparámetros
    def optimize_hyperparameters(train_df, n_trials=2,study_name="a2c_optimization", storage_path="sqlite:///optuna_study.db"):
        # Crear o cargar el estudio
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',
            load_if_exists=True
        )
        
        
        study.optimize(lambda trial: objective(trial, train_df), n_trials=n_trials)
        print("Mejores hiperparámetros:", study.best_params)
        # Guardar el resumen del estudio en un archivo CSV
        study.trials_dataframe().to_csv("optuna_trials.csv", index=False)
        
        return study.best_params

    def train_a2c_with_best_params(train_df, best_params,model_path="a2c_model",total_timesteps=10000, seed=42):
        """
        Entrena un modelo A2C con los mejores hiperparámetros.

        Args:
            train_df (pd.DataFrame): DataFrame de entrenamiento.
            best_params (dict): Mejores hiperparámetros obtenidos con Optuna.

        Returns:
            A2C: Modelo entrenado.
        """
        num_envs = 6  # Número de entornos paralelos
        def make_env(seed_offset=0):
            def _init():
                env = StockTradingEnv15min(train_df, initial_balance=10000)
                env.reset(seed=seed + seed_offset)  # Resetear el entorno con la semilla
                return env
            return _init

        vec_env = SubprocVecEnv([make_env(seed_offset=i) for i in range(num_envs)])

        print("punga 5")
        # Crear el modelo A2C con los mejores hiperparámetros
        model = A2C('MlpPolicy', vec_env, **best_params, verbose=1, device="cpu")
        print("punga 6")
        # Entrenar el modelo
        model.learn(total_timesteps=total_timesteps, callback=PrintTrainingStatisticsCallback(model_path))

        # Guardar el modelo entrenado
        model.save(model_path)
        print(f"Modelo guardado en: {model_path}.zip")
        return model    

    def test_a2c_model(model, test_df, seed=42, nombre="test_plot.png"):
        """
        Prueba un modelo A2C con los datos de prueba proporcionados y genera un gráfico de las acciones tomadas.

        Args:
            model (A2C): Modelo entrenado o ruta al modelo guardado.
            test_df (pd.DataFrame): DataFrame de datos de prueba.
            seed (int): Semilla para reproducibilidad.
            nombre (str): Nombre del archivo para guardar el gráfico.
        """
        # Si el modelo es una ruta, cargarlo
        if isinstance(model, str):
            model = A2C.load(model, device="cpu")

        # Crear el entorno de prueba
        test_env = StockTradingEnv15min(test_df, render_mode=True)
        obs, info = test_env.reset(seed=seed)

        net_worths = []
        actions = []

        # Probar el modelo
        for i in range(len(test_df)):
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = test_env.step(action)
            net_worths.append(test_env.net_worth)
            actions.append(action)

            if terminated or truncated:
                break

        # Guardar estadísticas de prueba
        stats_path = os.path.join(os.getcwd(), "test_stats.txt")
        with open(stats_path, "w") as f:
            f.write(f"Patrimonio Neto Final: {test_env.net_worth}\n")
        print(f"Estadísticas de prueba guardadas en: {stats_path}")

        # Rutina de visualización
        plt.figure(figsize=(12, 8))
        # Subplot 1: Valores de Close y operaciones
        plt.subplot(2, 1, 1)
        plt.plot(test_df['Close'].values, label='Close')
        plt.xlabel('Paso de Tiempo')
        plt.ylabel('Precio de Cierre')
        plt.title('Precio de Cierre y Operaciones (Prueba)')

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
        plt.title('Patrimonio Neto a lo Largo del Tiempo (Prueba)')

        plt.legend()

        # Guardar el gráfico
        plt.tight_layout()
        plt.savefig(nombre)
        plt.close()



        
    import optuna
    from stable_baselines3 import A2C
    from stable_baselines3.common.vec_env import DummyVecEnv

    from sklearn.utils import shuffle
    import pandas as pd

    def prepare_drl_training_data_sklearn(df, random_state=42):
        """
        Prepara un conjunto de datos para entrenar modelos de Deep Reinforcement Learning utilizando Scikit-learn para shuffling.

        Args:
            df (pd.DataFrame): DataFrame que contiene los datos.
            random_state (int): Semilla para la aleatorización (por defecto 42).

        Returns:
            pd.DataFrame: DataFrame mezclado.
        """
        # Mezclar los datos utilizando sklearn.utils.shuffle
        df_shuffled = shuffle(df, random_state=random_state)
        return df_shuffled
    # Optimizar los hiperparámetros

    def evaluate_with_multiple_seeds(train_df, test_df, best_params, model_path, num_seeds=5):
        """
        Entrena y evalúa el modelo con múltiples semillas aleatorias para observar la variabilidad del rendimiento.

        Args:
            train_df (pd.DataFrame): Datos de entrenamiento.
            test_df (pd.DataFrame): Datos de prueba.
            best_params (dict): Mejores hiperparámetros.
            model_path (str): Ruta para guardar el modelo.
            num_seeds (int): Número de semillas aleatorias para evaluar.

        Returns:
            dict: Resultados promedio y desviación estándar de las métricas.
        """
        results = []
        for seed in range(num_seeds):
            print(f"Evaluando con semilla: {seed}")
            model = train_a2c_with_best_params(train_df, best_params, model_path=f"{model_path}_seed{seed}.zip", seed=seed)
            metrics = test_and_validate_model(model, test_df, results_csv=f"test_results_seed{seed}.csv")
            results.append(metrics['Final Balance'])

        avg_balance = np.mean(results)
        std_balance = np.std(results)
        print(f"Balance Promedio: {avg_balance}, Desviación Estándar: {std_balance}")
        return {"Average Balance": avg_balance, "Std Balance": std_balance}
    # Cargar los datos
    file_path = r'C:\Users\cyber\Documents\Deep Learning\Deep Reinforcement Learning\A2C\dataset normalized\EURJPY_15min_NORMALIZED_normalized'
    df = pd.read_csv(file_path, sep=';')

    # Dividir los datos en entrenamiento y prueba sin mezclar
    train_size = int(len(df) * 0.8)  # Usar el 80% para entrenamiento
    train_df = df[:train_size]       # Datos de entrenamiento (ordenados temporalmente)
    test_df = df[train_size:]        


    # Preparar los datos con shuffling
    train_df = prepare_drl_training_data_sklearn(train_df)
    # Hiperparámetros óptimos
    best_params = {
                    'learning_rate': 8.603582509594346e-05,
                    'gamma': 0.9831301691926517,
                    'ent_coef': 2.7537065089483735e-07,
                    'vf_coef': 0.7997236244404522,
                    'max_grad_norm': 0.4466393228219806,
                    'gae_lambda': 0.9119112512928644,
                    'n_steps': 22
                }

    # Entrenar el modelo con los datos de entrenamiento
    #est_params = optimize_hyperparameters(train_df, n_trials=20)
    #model = train_a2c_with_best_params(train_df, best_params,total_timesteps=30000,model_path="a2c_eurjpy_15MIN.zip")

    model = A2C.load(r"C:\Users\cyber\Documents\Deep Learning\Deep Reinforcement Learning\A2C\a2c_eurjpy_15MIN.zip", device="cpu")
    test_a2c_model(model,test_df,nombre="test_plot_eurjpy_15MIN2.png")
    # metrics = test_and_validate_model(model, test_df, results_csv="test_results_eurjpy_15MIN.csv")
    # print("Final Balance EURJPY 15min:", metrics['Final Balance'])
    



    # # Cargar los datos GBPUSD
    # # Cargar los datos
    # file_path = r'C:\Users\cyber\Documents\Deep Learning\Deep Reinforcement Learning\A2C\dataset normalized\GBPUSD_15min_NORMALIZED_normalized'
    # df = pd.read_csv(file_path, sep=';')

    # # Dividir los datos en entrenamiento y prueba sin mezclar
    # train_size = int(len(df) * 0.7)  # Usar el 80% para entrenamiento
    # train_df = df[:train_size]       # Datos de entrenamiento (ordenados temporalmente)
    # test_df = df[train_size:]        


    # # # Preparar los datos con shuffling
    # # train_df = prepare_drl_training_data_sklearn(train_df)

    # # # Entrenar el modelo con los datos de entrenamiento
    # # best_params = optimize_hyperparameters(train_df,study_name="a2c_optimizationGBPUSD15m", n_trials=25)
    # # model = train_a2c_with_best_params(train_df, best_params,model_path="a2c_GBPUSD_15min.zip")

    # # #model = A2C.load(r"C:\Users\cyber\Documents\Deep Learning\Deep Reinforcement Learning\A2C\a2c_model.zip", device="cpu")
    # # metrics = test_and_validate_model(model, test_df, results_csv="test_results_GBPUSD_15min.csv")
    # # print("Final Balance GBPUSD 15min:", metrics['Final Balance'])

    # # # Cargar los datos GBPUSD
    # # # Cargar los datos
    # # file_path = r'C:\Users\cyber\Documents\Deep Learning\Deep Reinforcement Learning\A2C\dataset normalized\EURUSD_15min_NORMALIZED_normalized'
    # # df = pd.read_csv(file_path, sep=';')

    # # Dividir los datos en entrenamiento y prueba sin mezclar
    # train_size = int(len(df) * 0.7)  # Usar el 80% para entrenamiento
    # train_df = df[:train_size]       # Datos de entrenamiento (ordenados temporalmente)
    # test_df = df[train_size:]        


    # # Preparar los datos con shuffling
    # train_df = prepare_drl_training_data_sklearn(train_df)

    # # Entrenar el modelo con los datos de entrenamiento
    # best_params = optimize_hyperparameters(train_df, study_name="a2c_optimization_eurusd15min",n_trials=20)
    # model = train_a2c_with_best_params(train_df, best_params,model_path="a2c_EURUSD_15min.zip")

    # #model = A2C.load(r"C:\Users\cyber\Documents\Deep Learning\Deep Reinforcement Learning\A2C\a2c_model.zip", device="cpu")
    # metrics = test_and_validate_model(model, test_df, results_csv="test_results_EURUSD_15min.csv")
    # print("Final Balance: EURUSD 15 min", metrics['Final Balance'])


