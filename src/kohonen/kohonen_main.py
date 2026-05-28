import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from kohonen import KohonenSOM

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent / 'data' / 'europe.csv'
OUTPUT_DIR = BASE_DIR.parent.parent / 'outputs' / 'kohonen'
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid')

def save_output(fig, filename):
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Gráfico guardado en {output_path}')

def main():
    # 1. Cargar y preparar datos
    #np.random.seed(42)  # Para reproducibilidad
    df = pd.read_csv(DATA_PATH)
    countries = df['Country'].values
    df.set_index('Country', inplace=True)
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # 2. Configurar y entrenar la red SOM
    # Para 28 países, una grilla de 4x4 o 5x5 es un buen tamaño.
    grid_size = 4
    som = KohonenSOM(grid_y=grid_size, grid_x=grid_size, input_dim=X_scaled.shape[1], epochs=5000, learning_rate=0.1)
    
    print("Entrenando red de Kohonen (SOM)...")
    som.train(X_scaled)
    print("Entrenamiento finalizado.")
    
    # Obtener la neurona ganadora (BMU) para cada país
    bmus = som.get_bmus(X_scaled)
    
    # Crear un mapa para guardar los países asignados a cada neurona
    grid_countries = {}
    for y in range(grid_size):
        for x in range(grid_size):
            grid_countries[(y, x)] = []
            
    for country, bmu in zip(countries, bmus):
        grid_countries[bmu].append(country)

    # 3. Gráfico 1: Mapa de asociaciones (Países por neurona)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Dibujar la cuadrícula
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5) # Invertimos el eje y para que (0,0) esté arriba a la izquierda
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for (y, x), country_list in grid_countries.items():
        if country_list:
            text = "\n".join(country_list)
            ax.text(x, y, text, ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
    ax.set_title(f'Red de Kohonen ({grid_size}x{grid_size}): Agrupación de Países', fontsize=16)
    save_output(fig, 'som_countries_map.png')
    
    # 4. Gráfico 2: Matriz U (Distancias promedio entre neuronas vecinas)
    u_matrix = som.get_u_matrix()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(u_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax, cbar_kws={'label': 'Distancia Euclidiana Promedio'})
    ax.set_title('Matriz U: Distancia Promedio entre Neuronas Vecinas')
    save_output(fig, 'som_u_matrix.png')
    
    # 5. Gráfico 3: Cantidad de elementos por neurona
    count_matrix = np.zeros((grid_size, grid_size))
    for (y, x), country_list in grid_countries.items():
        count_matrix[y, x] = len(country_list)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(count_matrix, cmap='Blues', annot=True, fmt="g", ax=ax, cbar_kws={'label': 'Cantidad de Países'})
    ax.set_title('Densidad de Agrupación: Cantidad de países por neurona')
    save_output(fig, 'som_density_map.png')
    
    # 6. Gráfico 4: Planos de Componentes (Observar una sola variable)
    feature_names = df.columns
    num_features = len(feature_names)
    
    # Creamos una grilla de subplots (2 filas y 4 columnas para acomodar las 7 variables)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten() # Aplanamos para iterar fácilmente
    
    for i in range(num_features):
        ax = axes[i]
        # La matriz de pesos de la red tiene tamaño (grid_y, grid_x, cantidad_variables)
        # Extraemos la capa "i" que corresponde a la variable actual
        feature_weights = som.weights[:, :, i]
        
        # Dibujamos el mapa de calor para esta variable específica
        sns.heatmap(feature_weights, cmap='coolwarm', ax=ax, cbar=True)
        ax.set_title(feature_names[i], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Borramos el último subplot que nos sobra (porque son 7 variables y 8 espacios)
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle('Planos de Componentes: Distribución de variables en la red SOM', fontsize=16)
    plt.tight_layout()
    save_output(fig, 'som_component_planes.png')

    # Imprimir un análisis por consola
    print("\n--- Análisis de la Red de Kohonen ---")
    total_active_neurons = np.sum(count_matrix > 0)
    print(f"Neuronas totales: {grid_size * grid_size}")
    print(f"Neuronas activas (con al menos un país): {total_active_neurons}")
    print(f"Porcentaje de activación: {(total_active_neurons / (grid_size**2)) * 100:.2f}%")
    print(f"Máxima cantidad de países en una neurona: {int(np.max(count_matrix))}")

if __name__ == '__main__':
    main()
