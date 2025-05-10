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
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from ta.utils import dropna
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, TRIXIndicator
from ta.volatility import BollingerBands, DonchianChannel
from sklearn.utils import shuffle
import numpy as np

def process_csv(df,file_path, output_dir='./', separator=';', moving_average_window=5000):
    """
    Procesa un archivo CSV para preparar datos de trading.

    Args:
        file_path (str): Ruta al archivo CSV de entrada.
        output_dir (str): Directorio donde se guardará el archivo procesado.
        separator (str): Separador del archivo CSV (por defecto ';').
        moving_average_window (int): Ventana para calcular la media móvil.

    Returns:
        pd.DataFrame: DataFrame procesado.
    """

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)


    # Eliminar duplicados
    df = df.drop_duplicates()

    # Mantener solo las columnas 'Time', 'Volume' y 'Close'
    df = df[['Time', 'Volume', 'Close','Open', 'High', 'Low']]

    # Convertir la columna 'Time' a formato datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Calcular la diferencia a la media
    df['mean_close'] = df['Close'].rolling(window=moving_average_window).mean()
    df['diff_to_mean'] = df['Close'] - df['mean_close']

    # Calcular el volumen relativo
    df['mean_volume'] = df['Volume'].rolling(window=moving_average_window).mean()
    df['relative_volume'] = df['Volume'] / (df['mean_volume'] + 1e-8)  # Evitar división por cero

    # Calcular el cambio relativo (porcentual)
    df['relative_change'] = df['Close'].pct_change() * 100

    # Eliminar filas con valores NaN generados por la media móvil
    df = df.dropna()

    # Generar el nombre del archivo de salida
    base_name = os.path.basename(file_path)  # Nombre del archivo original
    name, ext = os.path.splitext(base_name)  # Separar nombre y extensión
    output_path = os.path.join(output_dir, f"{name}_procesado{ext}")

    # Guardar el DataFrame procesado en un nuevo archivo CSV
    df.to_csv(output_path, index=False, sep=separator)
    print(f"Archivo procesado guardado en: {output_path}")

    return df

def wavelet_column(df, output_dir='./', separator=';', wavelet='db1', level=1):
    """
    Aplica un filtrado Wavelet a la columna 'Close' de un archivo CSV y agrega la columna filtrada.

    Args:
        file_path (str): Ruta al archivo CSV de entrada.
        output_dir (str): Directorio donde se guardará el archivo procesado.
        separator (str): Separador del archivo CSV (por defecto ';').
        wavelet (str): Tipo de wavelet a usar (por defecto 'db1').
        level (int): Nivel de descomposición del wavelet (por defecto 1).

    Returns:
        pd.DataFrame: DataFrame con la columna 'Close_Filtrado_Wavelet' agregada.
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

   

    # Verificar si la columna 'Close' existe
    if 'Close' not in df.columns:
        raise ValueError("La columna 'Close' no existe en el archivo CSV.")

    # Aplicar el filtrado Wavelet
    close_values = df['Close'].values
    coeffs = pywt.wavedec(close_values, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, value=0.5 * max(c), mode='soft') for c in coeffs[1:]]  # Filtrado suave
    filtered_close = pywt.waverec(coeffs, wavelet)

    # Asegurarse de que la longitud coincida (puede haber diferencias por el padding)
    filtered_close = filtered_close[:len(df)]

    # Agregar la columna filtrada al DataFrame
    df['Close_Filtrado_Wavelet'] = filtered_close

    # Generar el nombre del archivo de salida
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_wavelet{ext}")

    # Guardar el DataFrame procesado en un nuevo archivo CSV
    df.to_csv(output_path, index=False, sep=separator)
    print(f"Archivo procesado con filtrado Wavelet guardado en: {output_path}")

    return df

def add_technical_indicators(df, output_dir='./', separator=';'):
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Verificar si las columnas necesarias existen
    required_columns = ['Close', 'High', 'Low', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' debe estar presente en el archivo CSV.")

    # Asegurarse de que no haya valores NaN
    df = dropna(df)

    # Calcular RSI (Relative Strength Index)
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # Calcular MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # Calcular Estocástico
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stochastic_K'] = stoch.stoch()
    df['Stochastic_D'] = stoch.stoch_signal()

    # Calcular Bandas de Bollinger
    bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()

    # Calcular Canales de Donchian
    donchian = DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
    df['Donchian_Upper'] = donchian.donchian_channel_hband()
    df['Donchian_Lower'] = donchian.donchian_channel_lband()
    df['Donchian_Middle'] = donchian.donchian_channel_mband()

    # Calcular ADX (Average Directional Index)
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['ADX_Pos'] = adx.adx_pos()
    df['ADX_Neg'] = adx.adx_neg()

    # Calcular Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # Calcular TRIX (Triple Exponential Moving Average)
    trix = TRIXIndicator(close=df['Close'], window=15)
    df['TRIX'] = trix.trix()




    # Generar el nombre del archivo de salida
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_with_indicators{ext}")

    # Guardar el DataFrame procesado en un nuevo archivo CSV
    df.to_csv(output_path, index=False, sep=separator)
    print(f"Archivo procesado con indicadores técnicos guardado en: {output_path}")

    return df

def add_all_ta(df):
     # Verificar si las columnas necesarias existen
    required_columns = ['Close', 'High', 'Low', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' debe estar presente en el archivo CSV.")

    # Asegurarse de que no haya valores NaN
    df = dropna(df)
    
      
    # Calcular RSI (Relative Strength Index)
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # Calcular MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # Calcular Estocástico
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stochastic_K'] = stoch.stoch()
    df['Stochastic_D'] = stoch.stoch_signal()

    # Calcular Bandas de Bollinger
    bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()

    # Calcular Canales de Donchian
    donchian = DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
    df['Donchian_Upper'] = donchian.donchian_channel_hband()
    df['Donchian_Lower'] = donchian.donchian_channel_lband()
    df['Donchian_Middle'] = donchian.donchian_channel_mband()

    # Calcular ADX (Average Directional Index)
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['ADX_Pos'] = adx.adx_pos()
    df['ADX_Neg'] = adx.adx_neg()

    # Calcular Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # Calcular TRIX (Triple Exponential Moving Average)
    trix = TRIXIndicator(close=df['Close'], window=15)
    df['TRIX'] = trix.trix()

    # Calcular Volatilidad (ATR - Average True Range)
    df['ATR'] = df['High'] - df['Low']

    # Calcular Media Móvil Simple (SMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # Calcular Media Móvil Exponencial (EMA)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Calcular Índice de Fuerza Relativa (Force Index)
    df['Force_Index'] = df['Close'].diff(1) * df['Volume']

    df = dropna(df)
    return df


def plot_all_in_one(df, columns, output_path='./combined_plot.png'):
    """
    Genera un único gráfico que contiene todas las columnas especificadas.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columns (list): Lista de nombres de columnas a graficar.
        output_path (str): Ruta donde se guardará el gráfico combinado.

    Returns:
        None
    """
    plt.figure(figsize=(15, 10))  # Tamaño del gráfico

    for column in columns:
        if column in df.columns:
            plt.plot(df[column], label=column)  # Graficar cada columna
        else:
            print(f"Columna no encontrada en el DataFrame: {column}")

    # Configurar título, etiquetas y leyenda
    plt.title('Gráfico Combinado de Indicadores', fontsize=16)
    plt.xlabel('Índice', fontsize=12)
    plt.ylabel('Valores', fontsize=12)
    plt.legend(loc='upper left', fontsize=8)  # Leyenda con las referencias
    plt.grid(True)

    # Guardar el gráfico
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Crear directorio si no existe
    plt.savefig(output_path, dpi=300)
    print(f'Gráfico combinado guardado en: {output_path}')

    # Mostrar el gráfico
    plt.show()

def drop_low_correlation_columns(df, target_column='Close', high_positive=(0.80, 1), low=(-0.2, 0.2), high_negative=(-1, -0.80)):
    """
    Elimina columnas que tienen una correlación con la columna objetivo en los rangos especificados.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        target_column (str): Columna objetivo para calcular la correlación.
        high_positive (tuple): Rango para correlaciones altas positivas (por defecto (0.8, 1)).
        low (tuple): Rango para correlaciones bajas (por defecto (-0.3, 0.3)).
        high_negative (tuple): Rango para correlaciones altas negativas (por defecto (-1, -0.8)).

    Returns:
        pd.DataFrame: DataFrame sin las columnas con correlación en los rangos especificados.
    """
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el DataFrame.")

    # Calcular la correlación con la columna objetivo
    correlations = df.corr()[target_column]

    # Identificar columnas con correlación en los rangos especificados
    columns_to_drop = correlations[
        ((correlations >= high_positive[0]) & (correlations <= high_positive[1])) |  # Correlación alta positiva
        ((correlations > low[0]) & (correlations < low[1])) |  # Correlación baja
        ((correlations >= high_negative[0]) & (correlations <= high_negative[1]))  # Correlación alta negativa
    ].index.tolist()

        # Asegurarse de no eliminar la columna objetivo
    if target_column in columns_to_drop:
        columns_to_drop.remove(target_column)

     # Guardar las columnas eliminadas en un archivo CSV
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    dropped_columns_path = os.path.join(os.path.dirname(file_path), f"{name}_dropped_columns.csv")
    pd.DataFrame(columns_to_drop, columns=['Dropped Columns']).to_csv(dropped_columns_path, index=False)
    print(f"Columnas eliminadas guardadas en: {dropped_columns_path}")

    # Eliminar las columnas identificadas
    df = df.drop(columns=columns_to_drop)
    print(f"Columnas eliminadas debido a correlación en los rangos especificados con '{target_column}': {columns_to_drop}")

    # Generar el nombre del archivo de salida basado en el nombre del archivo original
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(os.path.dirname(file_path), f"{name}_high_corr{ext}")

    # Guardar el DataFrame resultante en un archivo CSV
    df.to_csv(output_path, index=False,sep=';')
    print(f"DataFrame filtrado guardado en: {output_path}")


    return df

def drop_na_and_save(df, file_path='./'):
    """
    Elimina las filas con valores NaN o nulos al principio del DataFrame y guarda el resultado
    en un archivo CSV con el nombre del archivo original más '_dropna'.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        file_path (str): Ruta del archivo original para generar el nombre del archivo de salida.

    Returns:
        pd.DataFrame: DataFrame sin las filas con valores NaN o nulos al principio.
    """
    # Eliminar filas con valores NaN o nulos
    df_cleaned = df.dropna()

    # Generar el nombre del archivo de salida basado en el nombre del archivo original
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(os.path.dirname(file_path), f"{name}_dropna{ext}")

    # Guardar el DataFrame resultante en un archivo CSV
    df_cleaned.to_csv(output_path, index=False,sep=';')
    print(f"DataFrame sin valores NaN guardado en: {output_path}")

    return df_cleaned

def plot_correlation_heatmap(df, output_path='./correlation_heatmap.png'):
    """
    Genera y guarda un mapa de calor de correlación para un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        output_path (str): Ruta donde se guardará el gráfico del mapa de calor.

    Returns:
        None
    """
    # Calcular la matriz de correlación
    correlation_matrix = df.corr()

    # Configurar el tamaño del gráfico
    plt.figure(figsize=(15, 10))

    # Crear el mapa de calor
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)

    # Configurar título y etiquetas
    plt.title('Mapa de Calor de Correlación', fontsize=16)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)

    # Guardar el gráfico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Mapa de calor de correlación guardado en: {output_path}')

    # Mostrar el gráfico
    plt.show()
    return df 


def drop_date_column(df, date_column='Time'):
    """
    Elimina la columna de fecha especificada de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        date_column (str): Nombre de la columna de fecha a eliminar.

    Returns:
        pd.DataFrame: DataFrame sin la columna de fecha.
    """
    if date_column in df.columns:
        df = df.drop(columns=[date_column])
        print(f"Columna '{date_column}' eliminada del DataFrame.")
    else:
        print(f"La columna '{date_column}' no existe en el DataFrame.")
    return df

def plot_filtered_correlation_heatmap(df, threshold=0.6, output_path='./filtered_correlation_heatmap.png'):
    """
    Genera y guarda un mapa de calor de correlación filtrado para un DataFrame,
    mostrando solo las correlaciones mayores que el umbral positivo o menores que el umbral negativo.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        threshold (float): Umbral para filtrar las correlaciones (por defecto 0.6).
        output_path (str): Ruta donde se guardará el gráfico del mapa de calor.

    Returns:
        None
    """
    # Calcular la matriz de correlación
    correlation_matrix = df.corr()

    # Filtrar las correlaciones por el umbral
    filtered_matrix = correlation_matrix.applymap(
        lambda x: x if abs(x) > threshold else np.nan
    )

    # Configurar el tamaño del gráfico
    plt.figure(figsize=(15, 10))

    # Crear el mapa de calor
    sns.heatmap(filtered_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, mask=filtered_matrix.isnull())

    # Configurar título y etiquetas
    plt.title(f'Mapa de Calor de Correlación Filtrado (>|{threshold}|)', fontsize=16)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)

    # Guardar el gráfico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Mapa de calor de correlación filtrado guardado en: {output_path}')

    # Mostrar el gráfico
    plt.show()

def normalize_and_save(df, nombre,file_path='./'):
    """
    Normaliza los valores de todas las columnas numéricas en el DataFrame usando Min-Max Scaling
    y guarda el resultado en un archivo CSV con el nombre del archivo original más '_normalized'.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        file_path (str): Ruta del archivo original para generar el nombre del archivo de salida.

    Returns:
        pd.DataFrame: DataFrame con valores normalizados.
    """
    # Seleccionar solo las columnas numéricas
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Aplicar Min-Max Scaling
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Generar el nombre del archivo de salida basado en el nombre del archivo original
  
    output_path = os.path.join(os.path.dirname(file_path), f"{nombre}_normalized")

    # Guardar el DataFrame normalizado en un archivo CSV
    df.to_csv(output_path, index=False, sep=';')
    print(f"DataFrame normalizado guardado en: {output_path}")

    return df

def drop_specific_columns(df, columns_to_drop=['']):
    """
    Elimina columnas específicas de un DataFrame y guarda el resultado en un archivo CSV.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columns_to_drop (list): Lista de nombres de columnas a eliminar.
        file_path (str): Ruta del archivo original para generar el nombre del archivo de salida.

    Returns:
        pd.DataFrame: DataFrame sin las columnas especificadas.
    """
    # Lista de columnas a eliminar
    columns_to_drop = [
    'Time','Open', 'High', 'Low', 'relative_change', 'MACD_Hist', 'Stochastic_K', 'Stochastic_D',
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'Donchian_Upper', 'Donchian_Lower', 'Donchian_Middle',
    'ADX', 'ADX_Pos', 'ADX_Neg', 'Momentum']
    
    # Verificar si las columnas existen en el DataFrame
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if not existing_columns_to_drop:
        print("No se encontraron columnas para eliminar.")
        return df

    # Eliminar las columnas especificadas
    df = df.drop(columns=existing_columns_to_drop)
    print(f"Columnas eliminadas: {existing_columns_to_drop}")
    return df

def drop_specific_columns_15(df, columns_to_drop=['']):
    """
    Elimina columnas específicas de un DataFrame y guarda el resultado en un archivo CSV.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columns_to_drop (list): Lista de nombres de columnas a eliminar.
        file_path (str): Ruta del archivo original para generar el nombre del archivo de salida.

    Returns:
        pd.DataFrame: DataFrame sin las columnas especificadas.
    """
    # Lista de columnas a eliminar
    columns_to_drop = ['Open', 'High', 'Low', 'mean_close', 'relative_volume', 'relative_change', 
               'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
              'Stochastic_K', 'Stochastic_D', 'BB_Middle', 'BB_Upper', 'BB_Lower', 
              'Donchian_Upper', 'Donchian_Lower', 'Donchian_Middle', 'ADX', 'ADX_Pos', 
              'ADX_Neg', 'Momentum', 'TRIX', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'Force_Index']
    
    # Verificar si las columnas existen en el DataFrame
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if not existing_columns_to_drop:
        print("No se encontraron columnas para eliminar.")
        return df

    # Eliminar las columnas especificadas
    df = df.drop(columns=existing_columns_to_drop)
    print(f"Columnas eliminadas: {existing_columns_to_drop}")
    return df


def plot_and_save_financial_data(df, output_path=r'C:\Users\cyber\Documents\Deep Learning\Deep Reinforcement Learning\A2C\plots'):
    """
    Plots financial data with one graph for Close and Volume,
    and another separate graph for all other indicators, including TRIX.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        output_path (str): Path to save the resulting graphics.
    """
    # Scale volume to range 0-20
    scaler = MinMaxScaler(feature_range=(30, 120))
    df['Volume_Scaled'] = scaler.fit_transform(df[['Volume']])

    # Create the first figure for Close and Volume
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    ax1.plot(df['Close'], label='Close Price', color='blue', linewidth=2)
    ax1.bar(df.index, df['Volume_Scaled'], color='gray', alpha=0.5, label='Volume (Scaled)', width=1.0)
    ax1.set_title('Close Price and Scaled Volume', fontsize=14)
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Values', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Save the first plot
    close_volume_path = output_path.replace('.png', '_close_volume.png')
    os.makedirs(os.path.dirname(close_volume_path), exist_ok=True)
    plt.savefig(close_volume_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de Close y Volume guardado en: {close_volume_path}")

    # Show the first plot
    plt.show()

    # Create the second figure for all other indicators
    fig2, axes = plt.subplots(7, 1, figsize=(15, 25), sharex=True)
    fig2.tight_layout(pad=4.0)

    # Plot 1: Relative Volume
    axes[0].plot(df['relative_volume'], label='Relative Volume', color='orange')
    axes[0].set_title('Relative Volume', fontsize=12)
    axes[0].legend()

    # Plot 2: RSI with horizontal lines at 20 and 70
    axes[1].plot(df['RSI'], label='RSI', color='green')
    axes[1].axhline(y=20, color='red', linestyle='--', label='Lower Threshold (20)')
    axes[1].axhline(y=70, color='red', linestyle='--', label='Upper Threshold (70)')
    axes[1].set_title('RSI', fontsize=12)
    axes[1].legend()

    # Plot 3: MACD and MACD Signal
    axes[2].plot(df['MACD'], label='MACD', color='purple')
    axes[2].plot(df['MACD_Signal'], label='MACD Signal', color='brown')
    axes[2].set_title('MACD and MACD Signal', fontsize=12)
    axes[2].legend()

    # Plot 4: Stochastic K and D
    axes[3].plot(df['Stochastic_K'], label='Stochastic K', color='blue')
    axes[3].plot(df['Stochastic_D'], label='Stochastic D', color='orange')
    axes[3].set_title('Stochastic K and D', fontsize=12)
    axes[3].legend()

    # Plot 5: Momentum
    axes[4].plot(df['Momentum'], label='Momentum', color='cyan')
    axes[4].set_title('Momentum', fontsize=12)
    axes[4].legend()

    # Plot 6: diff_to_mean
    axes[5].plot(df['diff_to_mean'], label='Diff to Mean', color='lime')
    axes[5].set_title('Diff to Mean', fontsize=12)
    axes[5].legend()

    # Plot 7: TRIX
    axes[6].plot(df['TRIX'], label='TRIX', color='darkblue')
    axes[6].set_title('TRIX', fontsize=12)
    axes[6].legend()

    # Save the second plot
    indicators_path = output_path.replace('.png', '_indicators.png')
    plt.savefig(indicators_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de indicadores guardado en: {indicators_path}")

    # Show the second plot
    plt.show()

def plot_close_and_volume(df, output_path=r'./', low_volume_sc=0, high_volume_sc=1):
    """
    Plots the principal plot of Close price and scaled Volume.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        output_path (str): Path to save the resulting graphic.
    """
    # Scale volume to range 0-20
    scaler = MinMaxScaler(feature_range=(low_volume_sc, high_volume_sc))
    df['Volume_Scaled'] = scaler.fit_transform(df[['Volume']])

    # Create the figure for Close and Volume
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(df['Close'], label='Close Price', color='blue', linewidth=2)
    ax.bar(df.index, df['Volume_Scaled'], color='gray', alpha=0.5, label='Volume (Scaled)', width=1.0)
    ax.set_title('Close Price and Scaled Volume', fontsize=14)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.legend()
    ax.grid(True)

    # Save the plot
    close_volume_path = output_path.replace('.png', '_close_volume.png')
    os.makedirs(os.path.dirname(close_volume_path), exist_ok=True)
    plt.savefig(close_volume_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de Close y Volume guardado en: {close_volume_path}")

    # Show the plot
    plt.show()

def plot_indicators(df, output_path=r'./'):
    """
    Plots additional indicators: RSI, MACD, MACD Signal, mean_close, diff_to_mean, mean_volume, and TRIX.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        output_path (str): Path to save the resulting graphic.
    """
    # Create the figure for additional indicators
    fig, axes = plt.subplots(6, 1, figsize=(15, 25), sharex=True)
    fig.tight_layout(pad=4.0)

    # Plot 1: RSI with horizontal lines at 20 and 70
    axes[0].plot(df['RSI'], label='RSI', color='green')
    axes[0].axhline(y=0.2, color='red', linestyle='--', label='Lower Threshold (20)')
    axes[0].axhline(y=0.7, color='red', linestyle='--', label='Upper Threshold (70)')
    axes[0].set_title('RSI', fontsize=12)
    axes[0].legend()

    # Plot 2: MACD and MACD Signal
    axes[1].plot(df['MACD'], label='MACD', color='purple')
    axes[1].plot(df['MACD_Signal'], label='MACD Signal', color='brown')
    axes[1].set_title('MACD and MACD Signal', fontsize=12)
    axes[1].legend()

    # Plot 3: Mean Close and Diff to Mean
    axes[2].plot(df['mean_close'], label='Mean Close', color='blue')
    axes[2].plot(df['diff_to_mean'], label='Diff to Mean', color='orange')
    axes[2].set_title('Mean Close and Diff to Mean', fontsize=12)
    axes[2].legend()

    # Plot 4: Mean Volume
    axes[3].plot(df['mean_volume'], label='Mean Volume', color='cyan')
    axes[3].set_title('Mean Volume', fontsize=12)
    axes[3].legend()

    # Plot 5: TRIX
    axes[4].plot(df['TRIX'], label='TRIX', color='darkblue')
    axes[4].set_title('TRIX', fontsize=12)
    axes[4].legend()

   # Plot 6: Relative Volume
    axes[5].plot(df['relative_volume'], label='Relative Volume', color='orange')
    axes[5].set_title('Relative Volume', fontsize=12)
    axes[5].legend()

    # Save the plot
    indicators_path = output_path.replace('.png', '_indicators.png')
    plt.savefig(indicators_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de indicadores guardado en: {indicators_path}")

    # Show the plot
    plt.show()

def plot_indicators_15(df, output_path=r'./'):
    """
    Plots additional indicators: RSI, MACD, MACD Signal, mean_close, diff_to_mean, mean_volume, and TRIX.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        output_path (str): Path to save the resulting graphic.
    """
    # Create the figure for additional indicators
    fig, axes = plt.subplots(3, 1, figsize=(15, 25), sharex=True)
    fig.tight_layout(pad=4.0)

    # Plot 3:  Diff to Mean
    axes[0].plot(df['diff_to_mean'], label='Diff to Mean', color='orange')
    axes[0].set_title('Diff to Mean', fontsize=12)
    axes[0].legend()

    # Plot 4: Mean Volume
    axes[1].plot(df['mean_volume'], label='Mean Volume', color='cyan')
    axes[1].set_title('Mean Volume', fontsize=12)
    axes[1].legend()

    # Plot 5: ATR
    axes[2].plot(df['ATR'], label='ATR', color='darkblue')
    axes[2].set_title('ATR', fontsize=12)
    axes[2].legend()
   

    # Save the plot
    indicators_path = output_path.replace('.png', '_indicators.png')
    plt.savefig(indicators_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de indicadores guardado en: {indicators_path}")

    # Show the plot
    plt.show()



def prepare_drl_training_data_sklearn(df, batch_size=64, random_state=42):
    """
    Prepara un conjunto de datos para entrenar modelos de Deep Reinforcement Learning utilizando Scikit-learn para shuffling.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        batch_size (int): Tamaño de los lotes (batches) para el entrenamiento.
        random_state (int): Semilla para la aleatorización (por defecto 42).

    Returns:
        list: Lista de lotes (batches) de datos.
    """
    # Mezclar los datos utilizando sklearn.utils.shuffle
    df_shuffled = shuffle(df, random_state=random_state)

    # Convertir el DataFrame a un array de NumPy
    data = df_shuffled.to_numpy()

    # Dividir los datos en lotes (batches)
    num_batches = len(data) // batch_size
    batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    # Si hay datos restantes, añadirlos como un lote más pequeño
    if len(data) % batch_size != 0:
        batches.append(data[num_batches * batch_size:])

    print(f"Datos preparados en {len(batches)} lotes de tamaño {batch_size} (último lote puede ser más pequeño).")
    return batches





file_path = r'C:\Users\cyber\Documents\Deep Learning\Deep Reinforcement Learning\A2C\data 15 min\EURJPY_15min'
df = pd.read_csv(file_path, sep=';')
#final_file_path = r'C:\Users\cyber\Documents\Deep Learning\Deep Reinforcement Learning\A2C\processed_files\EURJPY_D1_procesado_wavelet_with_indicators_high_corr_dropna'

# final_df=plot_correlation_heatmap(drop_na_and_save(drop_specific_columns(add_technical_indicators(wavelet_column(process_csv(df,file_path))))),file_path)
# final_df=normalize_and_save(final_df,'GBPUSD_15min_NORMALIZED')
# plot_close_and_volume(final_df)
# plot_indicators(final_df)


final_df=plot_correlation_heatmap(drop_na_and_save(drop_specific_columns_15(drop_date_column(add_all_ta(wavelet_column(process_csv(df,file_path)))))),file_path)
final_df=normalize_and_save(final_df,'EURJPY_15min_NORMALIZED')
#plot_close_and_volume(final_df)
plot_indicators_15(final_df)





