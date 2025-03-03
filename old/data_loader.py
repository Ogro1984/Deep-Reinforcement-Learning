from PyEMD import EMD
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

def calcular_diferencia_promedio_historico(df, ventana_inicial, pasos, ventana_final, nombre_fichero, porcentaje_entrenamiento):
    # Eliminar columnas con 'Unnamed' en su nombre
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Eliminar las columnas especificadas
    columnas_a_eliminar = ['Open', 'High', 'Low', 'Real_volume', 'Spread', 'timeframe', 'symbol']
    df = df.drop(columns=columnas_a_eliminar)

    # Convertir la columna 'Time' a datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Crear directorios si no existen
    os.makedirs('training', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    os.makedirs('training_normalized', exist_ok=True)
    os.makedirs('test_normalized', exist_ok=True)
    os.makedirs('training_denoised_hard', exist_ok=True)
    os.makedirs('training_denoised_soft', exist_ok=True)
    os.makedirs('training_denoised_savgol', exist_ok=True)
    os.makedirs('training_denoised_hard_normalized', exist_ok=True)
    os.makedirs('training_denoised_soft_normalized', exist_ok=True)
    os.makedirs('training_denoised_savgol_normalized', exist_ok=True)

    for ventana in range(ventana_inicial, ventana_final + 1, pasos):
        # Calcular la diferencia contra el promedio móvil
        df[f'diferencia_media_{ventana}'] = df['Close'].rolling(window=ventana).mean() - df['Close']
        
        # Reemplazar NaN con 0
        df.fillna(0, inplace=True)
        
        # Descartar los valores hasta el primer dato diferente de 0 en la columna calculada
        df = df[df[f'diferencia_media_{ventana}'] != 0]
        
        # Seleccionar solo las columnas necesarias
        df = df[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        
        # Hacer un split de datos
        split_index = int(len(df) * porcentaje_entrenamiento)
        train_data = df.iloc[:split_index].copy()
        test_data = df.iloc[split_index:].copy()
        
        # Guardar los datasets en subcarpetas correspondientes
        train_data.to_csv(f'training/{nombre_fichero}_train_{ventana}.csv', index=False)
        test_data.to_csv(f'test/{nombre_fichero}_test_{ventana}.csv', index=False)
        
        # Normalizar los datos del grupo de entrenamiento
        scaler = MinMaxScaler()
        train_data_normalized = train_data.copy()
        train_data_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(train_data[['Close', 'Volume', f'diferencia_media_{ventana}']])
        train_data_normalized.to_csv(f'training_normalized/{nombre_fichero}_train_normalized_{ventana}.csv', index=False)
        
        # Aplicar métodos de denoising solo a la columna 'Close'
        emd = EMD()
        imfs = emd(train_data['Close'].values)
        
        # EMD con corte por umbral duro
        denoised_hard = pywt.threshold(imfs[-1], np.std(imfs[-1]), mode='hard')
        train_data_hard = train_data.copy()
        train_data_hard['Close'] = denoised_hard
        train_data_hard = train_data_hard[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        train_data_hard.to_csv(f'training_denoised_hard/{nombre_fichero}_train_denoised_hard_{ventana}.csv', index=False)
        
        # EMD con corte por umbral suave
        denoised_soft = pywt.threshold(imfs[-1], np.std(imfs[-1]), mode='soft')
        train_data_soft = train_data.copy()
        train_data_soft['Close'] = denoised_soft
        train_data_soft = train_data_soft[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        train_data_soft.to_csv(f'training_denoised_soft/{nombre_fichero}_train_denoised_soft_{ventana}.csv', index=False)
        
        # EMD con filtrado de Savitzky-Golay
        denoised_savgol = savgol_filter(imfs[-1], 11, 3)
        train_data_savgol = train_data.copy()
        train_data_savgol['Close'] = denoised_savgol
        train_data_savgol = train_data_savgol[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        train_data_savgol.to_csv(f'training_denoised_savgol/{nombre_fichero}_train_denoised_savgol_{ventana}.csv', index=False)
        
        # Normalizar las señales con denoising
        train_data_hard_normalized = train_data_hard.copy()
        train_data_soft_normalized = train_data_soft.copy()
        train_data_savgol_normalized = train_data_savgol.copy()
        
        train_data_hard_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(train_data_hard[['Close', 'Volume', f'diferencia_media_{ventana}']])
        train_data_soft_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(train_data_soft[['Close', 'Volume', f'diferencia_media_{ventana}']])
        train_data_savgol_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(train_data_savgol[['Close', 'Volume', f'diferencia_media_{ventana}']])
        
        # Guardar las señales con denoising normalizadas
        train_data_hard_normalized.to_csv(f'training_denoised_hard_normalized/{nombre_fichero}_train_denoised_hard_normalized_{ventana}.csv', index=False)
        train_data_soft_normalized.to_csv(f'training_denoised_soft_normalized/{nombre_fichero}_train_denoised_soft_normalized_{ventana}.csv', index=False)
        train_data_savgol_normalized.to_csv(f'training_denoised_savgol_normalized/{nombre_fichero}_train_denoised_savgol_normalized_{ventana}.csv', index=False)

def load_and_print_datasets():
    directories = [
        'training', 'test', 'training_normalized', 'test_normalized',
        'training_denoised_hard', 'training_denoised_soft', 'training_denoised_savgol',
        'training_denoised_hard_normalized', 'training_denoised_soft_normalized', 'training_denoised_savgol_normalized'
    ]
    
    for directory in directories:
        print(f"\nDirectory: {directory}")
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                print(f"\nFirst 10 lines of {filename}:\n")
                print(df.head(10))
                
                # Plotting the graph
                plt.figure(figsize=(10, 5))
                plt.plot(df['Time'], df['Close'], label='Close')
                plt.xlabel('Time')
                plt.ylabel('Close')
                plt.title(f'Close Prices for {filename}')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

# Ejemplo de uso
df_diario = pd.read_csv('fx_data\\GBPUSD_D1', sep=';', index_col=0)
calcular_diferencia_promedio_historico(df_diario, 5, 5, 20, 'GBPUSD_D1', 0.6)
load_and_print_datasets()