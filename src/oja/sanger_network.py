import numpy as np


class SangerNetwork:
    def __init__(self, input_dim, n_components, learning_rate=0.001, epochs=2000, random_seed=42):
        self.lr = learning_rate
        self.epochs = epochs
        self.n_components = n_components

        if random_seed is not None:
            np.random.seed(random_seed)

        # Weight matrix: each row is the weight vector for one output neuron
        self.weights = np.random.uniform(0, 1, (n_components, input_dim))

    def train(self, data):
        """
        Train using Sanger's rule to find the top-k principal components.

        Sanger's rule (generalized Oja):
            y = W @ x
            ΔW[i] = lr * y[i] * (x - sum_{l=0}^{i} y[l] * W[l])

        Vectorized form using a lower-triangular outer product:
            ΔW = lr * (outer(y, x) - tril(outer(y, y)) @ W)

        Returns weight matrix of shape (n_components, input_dim),
        where row i is the i-th principal component.
        """
        for epoch in range(self.epochs):
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)

            for idx in indices:
                x = data[idx]

                y = self.weights @ x  # (n_components,)

                # Progressive deflation via lower-triangular trick:
                # tril(outer(y,y))[i,l] = y[i]*y[l] if l<=i else 0
                # so (tril @ W)[i,j] = y[i] * sum_{l=0}^{i} y[l]*W[l,j]
                tril_yy = np.tril(np.outer(y, y))
                delta_W = self.lr * (np.outer(y, x) - tril_yy @ self.weights)
                self.weights += delta_W

        # Normalize each component to unit length
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights = self.weights / norms
        return self.weights
