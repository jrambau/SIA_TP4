import numpy as np

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
            lr = self.lr_0 * (1 - epoch / self.epochs)
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
