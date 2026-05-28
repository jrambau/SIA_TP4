import numpy as np

class KohonenSOM:
    """Self-Organizing Map (Kohonen) with rectangular 2D grid topology."""

    def __init__(self, grid_y, grid_x, input_dim, learning_rate=0.1, radius=None, epochs=1000,
                 init_method='normal', lr_decay='linear', track_qe=False):
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
        init_method : str, optional
            Weight initialization method: 'normal', 'uniform', or 'pca'.
        lr_decay : str, optional
            Learning rate decay strategy: 'linear', 'exponential', or 'inverse'.
        track_qe : bool, optional
            If True, compute and store quantization error after each epoch.
        """
        self.grid_y = grid_y
        self.grid_x = grid_x
        self.input_dim = input_dim
        self.lr_0 = learning_rate
        self.radius_0 = radius if radius is not None else max(grid_y, grid_x) / 2
        self.epochs = epochs
        self.init_method = init_method
        self.lr_decay = lr_decay
        self.track_qe = track_qe
        self.qe_history = []
        
        # Initialize weights (except for methods that need data)
        if init_method not in ('pca', 'sample'):
            self._init_weights()
        
        # Grid of coordinates for easy distance calculation
        y, x = np.mgrid[0:grid_y, 0:grid_x]
        self.grid_coords = np.c_[y.ravel(), x.ravel()].reshape((grid_y, grid_x, 2))

    def _init_weights(self, data=None):
        """Initialize weight vectors according to the chosen strategy."""
        if self.init_method == 'normal':
            # Random normal initialization (good for standardized data)
            self.weights = np.random.normal(0, 1, (self.grid_y, self.grid_x, self.input_dim))
        elif self.init_method == 'uniform':
            # Random uniform initialization in [-1, 1]
            self.weights = np.random.uniform(-1, 1, (self.grid_y, self.grid_x, self.input_dim))
        elif self.init_method == 'sample' and data is not None:
            # Initialize each neuron with a random sample from the training data
            n_neurons = self.grid_y * self.grid_x
            # Elegimos índices al azar del dataset. 
            # replace=False asegura que no elijamos el mismo país/dato dos veces (si hay suficientes datos)
            replace_flag = data.shape[0] < n_neurons
            random_indices = np.random.choice(data.shape[0], size=n_neurons, replace=replace_flag)
            
            # Tomamos esos ejemplos reales y le damos la forma de nuestra grilla
            self.weights = data[random_indices].reshape((self.grid_y, self.grid_x, self.input_dim))
        elif self.init_method == 'pca' and data is not None:
            # PCA-based initialization: span grid along first two principal components
            mean = data.mean(axis=0)
            centered = data - mean
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by descending eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = eigenvalues[idx]
            
            pc1 = eigenvectors[:, 0] * np.sqrt(abs(eigenvalues[0]))
            pc2 = eigenvectors[:, 1] * np.sqrt(abs(eigenvalues[1]))
            
            y_range = np.linspace(-1, 1, self.grid_y)
            x_range = np.linspace(-1, 1, self.grid_x)
            
            self.weights = np.zeros((self.grid_y, self.grid_x, self.input_dim))
            for i, y_val in enumerate(y_range):
                for j, x_val in enumerate(x_range):
                    self.weights[i, j] = mean + y_val * pc1 + x_val * pc2

    def _get_learning_rate(self, epoch):
        """Compute the learning rate for the given epoch using the chosen decay strategy."""
        if self.lr_decay == 'linear':
            return self.lr_0 * (1 - epoch / self.epochs)
        elif self.lr_decay == 'exponential':
            tau = self.epochs / 5
            return self.lr_0 * np.exp(-epoch / tau)
        elif self.lr_decay == 'inverse':
            # Alineado a Diapo 27: eta(i) = 1/i (usamos epoch + 1 para evitar div por 0)
            return self.lr_0 / (epoch + 1)
        return self.lr_0

    def _get_bmu(self, x):
        """Return the index (y, x) of the Best Matching Unit for input vector `x`."""
        # Calculate Euclidean distances between input x and all weights
        distances = np.linalg.norm(self.weights - x, axis=2)
        # Find the index of the minimum distance
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def train(self, data):
        """Train the SOM using online updates and Gaussian neighborhood decay.

        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, input_dim).
        """
        # If data-dependent init, do it now that we have data
        if self.init_method in ('pca', 'sample'):
            self._init_weights(data)
        
        if self.radius_0 > 1:
            lambda_r = self.epochs / np.log(self.radius_0)
        else:
            lambda_r = self.epochs  # Avoid log of values <= 1
        
        self.qe_history = []
        
        for epoch in range(self.epochs):
            # Update learning rate and radius
            lr = self._get_learning_rate(epoch)
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
                #======================================================
                # Calculate neighborhood function (Gaussian)
                #neighborhood = np.exp(-(dist_to_bmu**2) / (2 * (radius**2))) NO SE USA PERO LA DEJAMOS POR SI QUEREMOS USARLO DSP
                
                # Update weights
                # Reshape neighborhood to broadcast over input_dim
                #neighborhood = neighborhood[:, :, np.newaxis]
                #self.weights += lr * neighborhood * (x - self.weights)
                #======================================================
                
                #ACA SERIA COMO ESTA HECHO EN DIAPOSITIVAS
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
            
            # Track quantization error if requested
            if self.track_qe:
                qe = self.compute_quantization_error(data)
                self.qe_history.append(qe)

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

    def compute_quantization_error(self, data):
        """Compute the average Euclidean distance from each input to its BMU.

        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, input_dim).

        Returns
        -------
        float
            Mean quantization error.
        """
        total = 0
        for x in data:
            bmu = self._get_bmu(x)
            total += np.linalg.norm(x - self.weights[bmu[0], bmu[1]])
        return total / len(data)

    def compute_topographic_error(self, data):
        """Fraction of inputs whose second BMU is not adjacent to the first BMU.

        Measures how well the SOM preserves the topology of the input space.
        A value of 0 means perfect topology preservation.

        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, input_dim).

        Returns
        -------
        float
            Topographic error in [0, 1].
        """
        errors = 0
        for x in data:
            distances = np.linalg.norm(self.weights - x, axis=2)
            flat_indices = np.argsort(distances.ravel())
            bmu1 = np.unravel_index(flat_indices[0], distances.shape)
            bmu2 = np.unravel_index(flat_indices[1], distances.shape)
            
            # Check if bmu2 is a neighbor of bmu1 (Manhattan distance <= 1)
            grid_dist = abs(bmu1[0] - bmu2[0]) + abs(bmu1[1] - bmu2[1])
            if grid_dist > 1:
                errors += 1
        return errors / len(data)

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
