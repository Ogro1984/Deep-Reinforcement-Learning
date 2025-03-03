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

    # Determinar el timeframe del nombre del fichero
    if 'D1' in nombre_fichero:
        timeframe = 'D1'
    elif '15min' in nombre_fichero:
        timeframe = '15min'
    else:
        raise ValueError("Timeframe not found in the dataset name. Please include 'D1' or '15min' in the dataset name.")

    # Crear directorios si no existen
    base_dir = os.path.join('processed_data', nombre_fichero)
    timeframe_dir = os.path.join(base_dir, timeframe)
    os.makedirs(timeframe_dir, exist_ok=True)
    
    train_dir = os.path.join(timeframe_dir, 'train')
    test_dir = os.path.join(timeframe_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_subdirs = ['no_filtrado_no_normalizado', 'filtrado_no_normalizado', 'no_filtrado_normalizado', 'filtrado_normalizado']
    for subdir in train_subdirs:
        os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
    
    test_subdirs = ['no_filtrado_no_normalizado', 'filtrado_no_normalizado', 'no_filtrado_normalizado', 'filtrado_normalizado']
    for subdir in test_subdirs:
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
    
    filters = ['hard', 'soft', 'savgol', 'wavelet']
    for filter_method in filters:
        os.makedirs(os.path.join(train_dir, 'filtrado_no_normalizado', filter_method), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'filtrado_normalizado', filter_method), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_no_normalizado', filter_method), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_normalizado', filter_method), exist_ok=True)
    
    test_subdirs = ['normalized', 'not_normalized']
    for subdir in test_subdirs:
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)

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
        
        # Crear subcarpetas para cada ventana
        os.makedirs(os.path.join(train_dir, 'no_filtrado_no_normalizado', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'filtrado_no_normalizado', 'hard', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'filtrado_no_normalizado', 'soft', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'filtrado_no_normalizado', 'savgol', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'filtrado_no_normalizado', 'wavelet', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'no_filtrado_normalizado', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'filtrado_normalizado', 'hard', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'filtrado_normalizado', 'soft', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'filtrado_normalizado', 'savgol', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'filtrado_normalizado', 'wavelet', str(ventana)), exist_ok=True)

        os.makedirs(os.path.join(test_dir, 'no_filtrado_no_normalizado', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_no_normalizado', 'hard', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_no_normalizado', 'soft', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_no_normalizado', 'savgol', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_no_normalizado', 'wavelet', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'no_filtrado_normalizado', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_normalizado', 'hard', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_normalizado', 'soft', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_normalizado', 'savgol', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'filtrado_normalizado', 'wavelet', str(ventana)), exist_ok=True)

        os.makedirs(os.path.join(test_dir, 'normalized', str(ventana)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'not_normalized', str(ventana)), exist_ok=True)
        
        # Guardar los datasets en subcarpetas correspondientes
        train_data.to_csv(os.path.join(train_dir, 'no_filtrado_no_normalizado', str(ventana), f'{ventana}.csv'), index=False)
        test_data.to_csv(os.path.join(test_dir, 'no_filtrado_no_normalizado', str(ventana), f'{ventana}.csv'), index=False)
        
        # Normalizar los datos del grupo de entrenamiento
        scaler = MinMaxScaler()
        train_data_normalized = train_data.copy()
        train_data_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(train_data[['Close', 'Volume', f'diferencia_media_{ventana}']])
        train_data_normalized.to_csv(os.path.join(train_dir, 'no_filtrado_normalizado', str(ventana), f'{ventana}.csv'), index=False)
        
        test_data_normalized = test_data.copy()
        test_data_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(test_data[['Close', 'Volume', f'diferencia_media_{ventana}']])
        test_data_normalized.to_csv(os.path.join(test_dir, 'no_filtrado_normalizado', str(ventana), f'{ventana}.csv'), index=False)
        
        # Aplicar métodos de denoising solo a la columna 'Close'
        emd = EMD()
        imfs_train = emd(train_data['Close'].values)
        imfs_test = emd(test_data['Close'].values)
        
        # EMD con corte por umbral duro
        denoised_hard_train = pywt.threshold(imfs_train[-1], np.std(imfs_train[-1]) * 3, mode='hard')  # Much more aggressive threshold
        train_data_hard = train_data.copy()
        train_data_hard['Close'] = denoised_hard_train
        train_data_hard = train_data_hard[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        train_data_hard.to_csv(os.path.join(train_dir, 'filtrado_no_normalizado', 'hard', str(ventana), f'{ventana}.csv'), index=False)
        
        denoised_hard_test = pywt.threshold(imfs_test[-1], np.std(imfs_test[-1]) * 3, mode='hard')  # Much more aggressive threshold
        test_data_hard = test_data.copy()
        test_data_hard['Close'] = denoised_hard_test
        test_data_hard = test_data_hard[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        test_data_hard.to_csv(os.path.join(test_dir, 'filtrado_no_normalizado', 'hard', str(ventana), f'{ventana}.csv'), index=False)
        
        # EMD con corte por umbral suave
        denoised_soft_train = pywt.threshold(imfs_train[-1], np.std(imfs_train[-1]) * 3, mode='soft')  # Much more aggressive threshold
        train_data_soft = train_data.copy()
        train_data_soft['Close'] = denoised_soft_train
        train_data_soft = train_data_soft[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        train_data_soft.to_csv(os.path.join(train_dir, 'filtrado_no_normalizado', 'soft', str(ventana), f'{ventana}.csv'), index=False)
        
        denoised_soft_test = pywt.threshold(imfs_test[-1], np.std(imfs_test[-1]) * 3, mode='soft')  # Much more aggressive threshold
        test_data_soft = test_data.copy()
        test_data_soft['Close'] = denoised_soft_test
        test_data_soft = test_data_soft[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        test_data_soft.to_csv(os.path.join(test_dir, 'filtrado_no_normalizado', 'soft', str(ventana), f'{ventana}.csv'), index=False)
        
        # EMD con filtrado de Savitzky-Golay
        denoised_savgol_train = savgol_filter(imfs_train[-1], 7, 4)  # Much more aggressive window length and polynomial order
        train_data_savgol = train_data.copy()
        train_data_savgol['Close'] = denoised_savgol_train
        train_data_savgol = train_data_savgol[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        train_data_savgol.to_csv(os.path.join(train_dir, 'filtrado_no_normalizado', 'savgol', str(ventana), f'{ventana}.csv'), index=False)
        
        denoised_savgol_test = savgol_filter(imfs_test[-1], 7, 4)  # Much more aggressive window length and polynomial order
        test_data_savgol = test_data.copy()
        test_data_savgol['Close'] = denoised_savgol_test
        test_data_savgol = test_data_savgol[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        test_data_savgol.to_csv(os.path.join(test_dir, 'filtrado_no_normalizado', 'savgol', str(ventana), f'{ventana}.csv'), index=False)
        
        # Wavelet denoising
        coeffs_train = pywt.wavedec(train_data['Close'], 'db1', level=2)
        threshold_train = np.std(coeffs_train[-1]) * 3  # Much more aggressive threshold
        denoised_wavelet_train = pywt.waverec([pywt.threshold(c, threshold_train, mode='soft') for c in coeffs_train], 'db1')
        train_data_wavelet = train_data.copy()
        train_data_wavelet['Close'] = denoised_wavelet_train[:len(train_data_wavelet)]  # Ensure the length matches
        train_data_wavelet = train_data_wavelet[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        train_data_wavelet.to_csv(os.path.join(train_dir, 'filtrado_no_normalizado', 'wavelet', str(ventana), f'{ventana}.csv'), index=False)

        coeffs_test = pywt.wavedec(test_data['Close'], 'db1', level=2)
        threshold_test = np.std(coeffs_test[-1]) * 3  # Much more aggressive threshold
        denoised_wavelet_test = pywt.waverec([pywt.threshold(c, threshold_test, mode='soft') for c in coeffs_test], 'db1')
        test_data_wavelet = test_data.copy()
        test_data_wavelet['Close'] = denoised_wavelet_test[:len(test_data_wavelet)]  # Ensure the length matches
        test_data_wavelet = test_data_wavelet[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        test_data_wavelet.to_csv(os.path.join(test_dir, 'filtrado_no_normalizado', 'wavelet', str(ventana), f'{ventana}.csv'), index=False)
        
        # Normalizar las señales con denoising
        train_data_hard_normalized = train_data_hard.copy()
        train_data_soft_normalized = train_data_soft.copy()
        train_data_savgol_normalized = train_data_savgol.copy()
        train_data_wavelet_normalized = train_data_wavelet.copy()

        test_data_hard_normalized = test_data_hard.copy()
        test_data_soft_normalized = test_data_soft.copy()
        test_data_savgol_normalized = test_data_savgol.copy()
        test_data_wavelet_normalized = test_data_wavelet.copy()
        
        train_data_hard_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(train_data_hard[['Close', 'Volume', f'diferencia_media_{ventana}']])
        train_data_soft_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(train_data_soft[['Close', 'Volume', f'diferencia_media_{ventana}']])
        train_data_savgol_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(train_data_savgol[['Close', 'Volume', f'diferencia_media_{ventana}']])
        train_data_wavelet_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(train_data_wavelet[['Close', 'Volume', f'diferencia_media_{ventana}']])

        test_data_hard_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(test_data_hard[['Close', 'Volume', f'diferencia_media_{ventana}']])
        test_data_soft_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(test_data_soft[['Close', 'Volume', f'diferencia_media_{ventana}']])
        test_data_savgol_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(test_data_savgol[['Close', 'Volume', f'diferencia_media_{ventana}']])
        test_data_wavelet_normalized[['Close', 'Volume', f'diferencia_media_{ventana}']] = scaler.fit_transform(test_data_wavelet[['Close', 'Volume', f'diferencia_media_{ventana}']])
        
        # Guardar las señales con denoising normalizadas
        train_data_hard_normalized.to_csv(os.path.join(train_dir, 'filtrado_normalizado', 'hard', str(ventana), f'{ventana}.csv'), index=False)
        train_data_soft_normalized.to_csv(os.path.join(train_dir, 'filtrado_normalizado', 'soft', str(ventana), f'{ventana}.csv'), index=False)
        train_data_savgol_normalized.to_csv(os.path.join(train_dir, 'filtrado_normalizado', 'savgol', str(ventana), f'{ventana}.csv'), index=False)
        train_data_wavelet_normalized.to_csv(os.path.join(train_dir, 'filtrado_normalizado', 'wavelet', str(ventana), f'{ventana}.csv'), index=False)

        test_data_hard_normalized.to_csv(os.path.join(test_dir, 'filtrado_normalizado', 'hard', str(ventana), f'{ventana}.csv'), index=False)
        test_data_soft_normalized.to_csv(os.path.join(test_dir, 'filtrado_normalizado', 'soft', str(ventana), f'{ventana}.csv'), index=False)
        test_data_savgol_normalized.to_csv(os.path.join(test_dir, 'filtrado_normalizado', 'savgol', str(ventana), f'{ventana}.csv'), index=False)
        test_data_wavelet_normalized.to_csv(os.path.join(test_dir, 'filtrado_normalizado', 'wavelet', str(ventana), f'{ventana}.csv'), index=False)
        
        # Guardar gráficos de entrenamiento
        for data, label, filter_method in zip(
            [train_data, train_data_normalized, train_data_hard, train_data_soft, train_data_savgol, train_data_wavelet, train_data_hard_normalized, train_data_soft_normalized, train_data_savgol_normalized, train_data_wavelet_normalized],
            ['no_filtrado_no_normalizado', 'no_filtrado_normalizado', 'filtrado_no_normalizado', 'filtrado_no_normalizado', 'filtrado_no_normalizado', 'filtrado_no_normalizado', 'filtrado_normalizado', 'filtrado_normalizado', 'filtrado_normalizado', 'filtrado_normalizado'],
            [None, None, 'hard', 'soft', 'savgol', 'wavelet', 'hard', 'soft', 'savgol', 'wavelet']
        ):
            plt.figure(figsize=(10, 5))
            plt.plot(data['Time'], data['Close'], label='Close')
            plt.xlabel('Time')
            plt.ylabel('Close')
            plt.title(f'Close Prices for {nombre_fichero} - {label} - {ventana}')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            if filter_method:
                graph_path = os.path.join(train_dir, label, filter_method, str(ventana), f'{ventana}.png')
            else:
                graph_path = os.path.join(train_dir, label, str(ventana), f'{ventana}.png')
            plt.savefig(graph_path)
            plt.close()

        # Guardar gráficos de prueba
        for data, label, filter_method in zip(
            [test_data, test_data_normalized, test_data_hard, test_data_soft, test_data_savgol, test_data_wavelet, test_data_hard_normalized, test_data_soft_normalized, test_data_savgol_normalized, test_data_wavelet_normalized],
            ['no_filtrado_no_normalizado', 'no_filtrado_normalizado', 'filtrado_no_normalizado', 'filtrado_no_normalizado', 'filtrado_no_normalizado', 'filtrado_no_normalizado', 'filtrado_normalizado', 'filtrado_normalizado', 'filtrado_normalizado', 'filtrado_normalizado'],
            [None, None, 'hard', 'soft', 'savgol', 'wavelet', 'hard', 'soft', 'savgol', 'wavelet']
        ):
            plt.figure(figsize=(10, 5))
            plt.plot(data['Time'], data['Close'], label='Close')
            plt.xlabel('Time')
            plt.ylabel('Close')
            plt.title(f'Close Prices for {nombre_fichero} - {label} - {ventana}')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            if filter_method:
                graph_path = os.path.join(test_dir, label, filter_method, str(ventana), f'{ventana}.png')
            else:
                graph_path = os.path.join(test_dir, label, str(ventana), f'{ventana}.png')
            plt.savefig(graph_path)
            plt.close()

        # Guardar gráfico general en la carpeta del timeframe
        plt.figure(figsize=(10, 5))
        plt.plot(train_data['Time'], train_data['Close'], label='Close')
        plt.xlabel('Time')
        plt.ylabel('Close')
        plt.title(f'Close Prices for {nombre_fichero} - {ventana}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        graph_path = os.path.join(timeframe_dir, f'{ventana}.png')
        plt.savefig(graph_path)
        plt.close()

def load_and_print_datasets():
    directories = [
        'processed_data'
    ]
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.csv'):
                    filepath = os.path.join(root, filename)
                    df = pd.read_csv(filepath)
                    print(f"\nFirst 10 lines of {filename}:\n")
                    print(df.head(10))

# Ejemplo de uso
df_diario = pd.read_csv('fx_data\\GBPUSD_D1', sep=';', index_col=0)
calcular_diferencia_promedio_historico(df_diario, 5, 5, 20, 'GBPUSD_D1', 0.6)
load_and_print_datasets()