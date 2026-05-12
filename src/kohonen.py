import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'data' / 'europe.csv'
OUTPUT_DIR = BASE_DIR.parent / 'outputs'/'kohonen'
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid')

def save_output(fig, filename):
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Gráfico guardado en {output_path}')

class KohonenSOM:
    def __init__(self, grid_y, grid_x, input_dim, learning_rate=0.1, radius=None, epochs=1000):
        self.grid_y = grid_y
        self.grid_x = grid_x
        self.input_dim = input_dim
        self.lr_0 = learning_rate
        self.radius_0 = radius if radius is not None else max(grid_y, grid_x) / 2
        self.epochs = epochs
        
        # Initialize weights with random values from a normal distribution
        # Since data will be scaled to zero mean and unit variance
        self.weights = np.random.normal(0, 1, (grid_y, grid_x, input_dim))
        
        # Grid of coordinates for easy distance calculation
        y, x = np.mgrid[0:grid_y, 0:grid_x]
        self.grid_coords = np.c_[y.ravel(), x.ravel()].reshape((grid_y, grid_x, 2))

    def _get_bmu(self, x):
        # Calculate Euclidean distances between input x and all weights
        distances = np.linalg.norm(self.weights - x, axis=2)
        # Find the index of the minimum distance
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def train(self, data):
        lambda_r = self.epochs / np.log(self.radius_0)
        
        for epoch in range(self.epochs):
            # Update learning rate and radius
            lr = self.lr_0 * np.exp(-epoch / self.epochs)
            radius = self.radius_0 * np.exp(-epoch / lambda_r)
            
            # Shuffle data to avoid order bias
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            
            for idx in indices:
                x = data[idx]
                bmu = self._get_bmu(x)
                
                # Calculate distance from BMU to all other neurons in the grid
                dist_to_bmu = np.linalg.norm(self.grid_coords - np.array(bmu), axis=2)
                
                # Calculate neighborhood function (Gaussian)
                neighborhood = np.exp(-(dist_to_bmu**2) / (2 * (radius**2)))
                
                # Update weights
                # Reshape neighborhood to broadcast over input_dim
                neighborhood = neighborhood[:, :, np.newaxis]
                self.weights += lr * neighborhood * (x - self.weights)

    def get_bmus(self, data):
        bmus = []
        for x in data:
            bmus.append(self._get_bmu(x))
        return bmus

    def get_u_matrix(self):
        u_matrix = np.zeros((self.grid_y, self.grid_x))
        for y in range(self.grid_y):
            for x in range(self.grid_x):
                neighbors = []
                if y > 0: neighbors.append(self.weights[y-1, x])
                if y < self.grid_y - 1: neighbors.append(self.weights[y+1, x])
                if x > 0: neighbors.append(self.weights[y, x-1])
                if x < self.grid_x - 1: neighbors.append(self.weights[y, x+1])
                
                # Diagonal neighbors
                if y > 0 and x > 0: neighbors.append(self.weights[y-1, x-1])
                if y > 0 and x < self.grid_x - 1: neighbors.append(self.weights[y-1, x+1])
                if y < self.grid_y - 1 and x > 0: neighbors.append(self.weights[y+1, x-1])
                if y < self.grid_y - 1 and x < self.grid_x - 1: neighbors.append(self.weights[y+1, x+1])
                
                distances = [np.linalg.norm(self.weights[y, x] - neighbor) for neighbor in neighbors]
                u_matrix[y, x] = np.mean(distances)
        return u_matrix

def main():
    # 1. Cargar y preparar datos
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

    # Imprimir un análisis por consola
    print("\n--- Análisis de la Red de Kohonen ---")
    total_active_neurons = np.sum(count_matrix > 0)
    print(f"Neuronas totales: {grid_size * grid_size}")
    print(f"Neuronas activas (con al menos un país): {total_active_neurons}")
    print(f"Porcentaje de activación: {(total_active_neurons / (grid_size**2)) * 100:.2f}%")
    print(f"Máxima cantidad de países en una neurona: {int(np.max(count_matrix))}")

if __name__ == '__main__':
    main()
