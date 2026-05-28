import numpy as np

class KohonenSOM:
    """Self-Organizing Map (Kohonen) with rectangular 2D grid topology."""

    def __init__(self, grid_y, grid_x, input_dim, learning_rate=0.1, radius=None, epochs=1000):
        """Initialize SOM structure and training hyperparameters.

        Parameters
        ----------
        grid_y : int
            Number of rows in the neuron grid.
        grid_x : int
            Number of columns in the neuron grid.
        input_dim : int
            Dimensionality of each input sample.
        learning_rate : float, optional
            Initial learning rate.
        radius : float or None, optional
            Initial neighborhood radius. If None, defaults to half of max grid size.
        epochs : int, optional
            Number of training epochs.
        """
        self.grid_y = grid_y
        self.grid_x = grid_x
        self.input_dim = input_dim
        self.lr_0 = learning_rate
        self.radius_0 = radius if radius is not None else max(grid_y, grid_x) / 2
        self.epochs = epochs
        
        self.weights = None
        
        # Grid of coordinates for easy distance calculation
        y, x = np.mgrid[0:grid_y, 0:grid_x]
        self.grid_coords = np.c_[y.ravel(), x.ravel()].reshape((grid_y, grid_x, 2))

    def _get_bmu(self, x):
        """Return the index (y, x) of the Best Matching Unit for input vector `x`."""
        # Calculate Euclidean distances between input x and all weights
        distances = np.linalg.norm(self.weights - x, axis=2)
        # Find the index of the minimum distance
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def train(self, data):
        """Train the SOM using exact rules from Slide 26 and 27."""
        
        # --- NUEVO BLOQUE DE INICIALIZACIÓN ---
        # Si es la primera vez que se llama a train, inicializamos los pesos
        if self.weights is None:
            num_neurons = self.grid_y * self.grid_x
            
            # Elegimos índices al azar del dataset. 
            # replace=False asegura que no elijamos el mismo país/dato dos veces (si hay suficientes datos)
            replace_flag = data.shape[0] < num_neurons
            random_indices = np.random.choice(data.shape[0], size=num_neurons, replace=replace_flag)
            
            # Tomamos esos ejemplos reales y le damos la forma de nuestra grilla
            self.weights = data[random_indices].reshape((self.grid_y, self.grid_x, self.input_dim))
        # --------------------------------------
        
        # Factor de decaimiento para el radio (se mantiene exponencial para cumplir que R(i)->1)
        lambda_r = self.epochs / np.log(self.radius_0)
        
        for epoch in range(self.epochs):
            # 1. ACTUALIZACIÓN DE TASA DE APRENDIZAJE Y RADIO
            # Alineado a Diapo 27: Por ejemplo eta(i) = 1/i (usamos epoch + 1 para evitar div por 0)
            lr = self.lr_0 / (epoch + 1)
            
            # Alineado a Diapo 26: R(i) -> 1 cuando i -> inf
            radius = self.radius_0 * np.exp(-epoch / lambda_r)
            if radius < 1.0: 
                radius = 1.0 # Aseguramos que el límite sea 1 como dice la presentación
            
            # Shuffle data to avoid order bias
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            
            for idx in indices:
                x = data[idx]
                
                # 2. ENCONTRAR LA NEURONA GANADORA (Diapo 25, Paso 2)
                bmu = self._get_bmu(x)
                
                # Calcular distancia  ||n - n_k||
                dist_to_bmu = np.linalg.norm(self.grid_coords - np.array(bmu), axis=2)
                
                # 3. REGLA DE KOHONEN ALINEADA A DIAPO 26 Y 27
                # Diapo 26: N_k(i) = {n / ||n - n_k|| < R(i)}
                # Creamos una máscara: 1.0 si es menor estricto que el radio, 0.0 si está afuera
                neighborhood_mask = (dist_to_bmu < radius).astype(float)
                
                # Expandimos dimensiones para multiplicar por los pesos
                neighborhood_mask = neighborhood_mask[:, :, np.newaxis]
                
                # Diapo 27: 
                # Si j pertenece a N_k(i) -> W + lr * (X - W)
                # Si j NO pertenece       -> W + 0  (queda igual)
                self.weights += lr * neighborhood_mask * (x - self.weights)

    def get_bmus(self, data):
        """Compute BMU coordinates for each sample in `data`.

        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, input_dim).

        Returns
        -------
        list[tuple[int, int]]
            BMU grid coordinates for each input sample.
        """
        bmus = []
        for x in data:
            bmus.append(self._get_bmu(x))
        return bmus

    def get_u_matrix(self):
        """Return the U-Matrix (mean distance to neighboring neurons) for each neuron."""
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
