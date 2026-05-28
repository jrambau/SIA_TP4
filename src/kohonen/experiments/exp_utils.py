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
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    return X_scaled, countries, feature_names


def save_figure(fig, filename):
    """Save a matplotlib figure to the experiments output directory."""
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Gráfico guardado en {output_path}')
    plt.close(fig)


def draw_country_map(ax, som, X_scaled, countries, grid_y, grid_x, fontsize=7):
    """Draw the country assignment map on a given axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    som : KohonenSOM
        Trained SOM.
    X_scaled : np.ndarray
        Standardized data.
    countries : np.ndarray
        Country names.
    grid_y, grid_x : int
        Grid dimensions.
    fontsize : int
        Font size for country labels.
    """
    bmus = som.get_bmus(X_scaled)
    
    grid_countries = {}
    for y in range(grid_y):
        for x in range(grid_x):
            grid_countries[(y, x)] = []
    for country, bmu in zip(countries, bmus):
        grid_countries[bmu].append(country)
    
    ax.set_xlim(-0.5, grid_x - 0.5)
    ax.set_ylim(grid_y - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, grid_x, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_y, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for (y, x), country_list in grid_countries.items():
        if country_list:
            text = "\n".join(country_list)
            ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                    bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
