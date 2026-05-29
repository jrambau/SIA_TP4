"""Utilidades compartidas para todos los experimentos de Kohonen."""
import sys
from pathlib import Path

# Add parent directory to path so we can import the KohonenSOM class
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# IMPORTANTE: Instalar con 'pip install adjustText'
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("Advertencia: Se recomienda 'pip install adjustText' para evitar superposición de textos.")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent.parent / 'data' / 'europe.csv'
OUTPUT_DIR = BASE_DIR.parent.parent.parent / 'outputs' / 'kohonen' / 'experiments'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')


def load_data():
    """Load and standardize the Europe dataset.
    
    Returns
    -------
    X_scaled : np.ndarray
        Standardized feature matrix (n_samples, n_features).
    countries : np.ndarray
        Array of country names.
    feature_names : pd.Index
        Names of the features.
    """
    df = pd.read_csv(DATA_PATH)
    countries = df['Country'].values
    df.set_index('Country', inplace=True)
    feature_names = df.columns
    
    # Fundamental: Estandarizar vectores para que las medidas de similitud (ej. Euclidiana) funcionen correctamente.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    return X_scaled, countries, feature_names


def save_figure(fig, filename):
    """Save a matplotlib figure to the experiments output directory."""
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight') # Subido a 300 dpi para calidad 4K en PPT
    print(f'[Utils] Gráfico guardado en {output_path}')
    plt.close(fig)


def draw_country_map(ax, som, X_scaled, countries, grid_y, grid_x, base_fontsize=10):
    """
    Dibuja el mapa de asignación de países con texto centrado y tamaño dinámico.
    """
    bmus = som.get_bmus(X_scaled)
    
    # Agrupamos países por BMU
    grid_countries = {(y, x): [] for y in range(grid_y) for x in range(grid_x)}
    for country, bmu in zip(countries, bmus):
        grid_countries[bmu].append(country)
    
    # Configuramos la grilla visual
    ax.set_xlim(-0.5, grid_x - 0.5)
    ax.set_ylim(grid_y - 0.5, -0.5)
    
    # Dibujamos las líneas separadoras de las neuronas
    ax.set_xticks(np.arange(-0.5, grid_x, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_y, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Recorremos la grilla y distribuimos los países
    for (y, x), country_list in grid_countries.items():
        if country_list:
            # Ordenamos alfabéticamente para mayor prolijidad en la lectura
            country_list.sort()
            
            # Unimos los nombres con un salto de línea
            text = "\n".join(country_list)
            
            # Ajuste dinámico de fuente: a mayor cantidad de países, letra más chica
            # Base = 10. Restamos 0.8 por cada país extra, con un mínimo de 4 pts.
            n_countries = len(country_list)
            dynamic_fontsize = max(4.0, base_fontsize - (n_countries * 0.8))
            
            # Colocamos el texto exactamente en el centro de la neurona (discreto)
            ax.text(x, y, text, fontsize=dynamic_fontsize, 
                    ha='center', va='center', weight='semibold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))