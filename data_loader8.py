from PyEMD import EMD
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def analyze_and_preprocess_data(df, nombre_fichero, ventana):

    # Directorio para guardar los resultados del análisis
    analysis_dir = os.path.join('analysis_results', nombre_fichero, str(ventana))
    os.makedirs(analysis_dir, exist_ok=True)

    # 1. Preprocesamiento de Datos
    print("Realizando preprocesamiento de datos...")
    # Eliminar duplicados
    df = df.drop_duplicates()

    # Manejar valores faltantes (rellenar con la media)
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # 2. Análisis Exploratorio de Datos (EDA)
    print("Realizando análisis exploratorio de datos...")
    # Resumen Estadístico
    print("\nDescriptivos Básicos:")
    print(df.describe())

    # Visualización
    print("\nVisualización de distribuciones...")
    for col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histograma de {col}')
        plt.savefig(os.path.join(analysis_dir, f'hist_{col}.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        plt.savefig(os.path.join(analysis_dir, f'boxplot_{col}.png'))
        plt.close()

    # Gráficos de dispersión para pares de variables
    print("\nGráficos de dispersión...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[col1], y=df[col2])
            plt.title(f'Gráfico de dispersión entre {col1} y {col2}')
            plt.savefig(os.path.join(analysis_dir, f'scatter_{col1}_{col2}.png'))
            plt.close()

    # 3. Análisis de Correlación
    print("Realizando análisis de correlación...")
    # Matriz de Correlación
    corr_matrix = df.corr()
    print("\nMatriz de Correlación:")
    print(corr_matrix)

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap de la Matriz de Correlación')
    plt.savefig(os.path.join(analysis_dir, 'heatmap.png'))
    plt.close()

    # 4. Identificación de Patrones
    print("Identificando patrones...")
    # Tendencias y Estacionalidad (solo para la columna 'Close')
    if 'Close' in df.columns and 'Time' in df.columns:
        df_time = df.set_index('Time')
        plt.figure(figsize=(14, 7))
        plt.plot(df_time['Close'])
        plt.title('Tendencias y Estacionalidad de la Columna Close')
        plt.xlabel('Tiempo')
        plt.ylabel('Close')
        plt.savefig(os.path.join(analysis_dir, 'time_series_close.png'))
        plt.close()

    # Análisis de Componentes Principales (PCA)
    print("Realizando análisis de componentes principales (PCA)...")
    if len(numeric_cols) > 1:
        # Escalar los datos
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])

        # Aplicar PCA
        pca = PCA(n_components=min(10, len(numeric_cols)))  # Limitar a 10 componentes o el número de columnas, lo que sea menor
        pca.fit(scaled_data)
        explained_variance = pca.explained_variance_ratio_

        print("\nComponentes Principales y Varianza Explicada:")
        print(explained_variance)

        # Visualizar la varianza explicada
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(explained_variance)), explained_variance, alpha=0.8, align='center', label='Varianza Individual Explicada')
        plt.ylabel('Varianza Explicada')
        plt.xlabel('Componentes Principales')
        plt.title('Varianza Explicada por Componentes Principales')
        plt.savefig(os.path.join(analysis_dir, 'pca_variance.png'))
        plt.close()

    # 5. Detección de Anomalías
    print("Detectando anomalías...")
    # Identificación de Outliers (usando el rango intercuartílico IQR)
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\nOutliers en la columna {col}:")
        print(outliers)

    # 6. Visualización Avanzada
    print("Realizando visualización avanzada...")
    # Gráficos Interactivos (usando Plotly)
    if 'Close' in df.columns and 'Time' in df.columns:
        fig = px.line(df, x='Time', y='Close', title='Gráfico Interactivo de Close Prices')
        fig.write_html(os.path.join(analysis_dir, 'interactive_close_prices.html'))

    print(f"Análisis completado. Los resultados se han guardado en: {analysis_dir}")


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
        
          # Realizar preprocesamiento y análisis de datos
        print(f"Realizando preprocesamiento y análisis para la ventana {ventana}...")
        analyze_and_preprocess_data(df, nombre_fichero, ventana)
       

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
        if '15min' in nombre_fichero:
            wavelet_threshold_multiplier = 25  # Even more aggressive threshold for 15min datasets
            wavelet_level = 10  # Increase wavelet decomposition level
            
        else:
           
            wavelet_threshold_multiplier = 3  # Default threshold
            wavelet_level = 2  # Default wavelet decomposition level

        # Train data wavelet denoising
        coeffs_train = pywt.wavedec(train_data['Close'], 'db1', level=wavelet_level)
        threshold_train = np.std(coeffs_train[-1]) * wavelet_threshold_multiplier  # Variable threshold
        denoised_wavelet_train = pywt.waverec([pywt.threshold(c, threshold_train, mode='soft') for c in coeffs_train], 'db1')
        train_data_wavelet = train_data.copy()
        train_data_wavelet['Close'] = denoised_wavelet_train[:len(train_data_wavelet)]  # Ensure the length matches
        train_data_wavelet = train_data_wavelet[['Time', 'Close', 'Volume', f'diferencia_media_{ventana}']]
        train_data_wavelet.to_csv(os.path.join(train_dir, 'filtrado_no_normalizado', 'wavelet', str(ventana), f'{ventana}.csv'), index=False)

        # Test data wavelet denoising
        coeffs_test = pywt.wavedec(test_data['Close'], 'db1', level=wavelet_level)
        threshold_test = np.std(coeffs_test[-1]) * wavelet_threshold_multiplier  # Variable threshold
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

def process_all_fx_data(folder_path='fx_data'):
    for filename in os.listdir(folder_path):
        
            filepath = os.path.join(folder_path, filename)
            try:
                df_diario = pd.read_csv(filepath, sep=';', index_col=0)
                nombre_fichero = filename.replace('.csv', '')
                
                if '15min' in filename:
                    calcular_diferencia_promedio_historico(df_diario, 250, 50, 300, nombre_fichero, 0.6)
                else:
                    calcular_diferencia_promedio_historico(df_diario, 250, 50, 300, nombre_fichero, 0.6)
                
                print(f"Dataset {filename} processed successfully.")
            except Exception as e:
                print(f"Error processing dataset {filename}: {e}")

# Ejemplo de uso
process_all_fx_data()
