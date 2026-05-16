import numpy as np

class OjaNetwork:
    def __init__(self, input_dim, learning_rate=0.001, epochs=2000, random_seed=42):
        self.lr = learning_rate
        self.epochs = epochs
        
        # Fijamos semilla para reproducibilidad
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Inicializamos pesos con distribución 
        # uniforme entre 0 y 1.
        self.weights = np.random.uniform(0, 1, input_dim)

    def train(self, data):
        """
        Train the Oja's network using the provided data.
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, input_dim).
        Returns
        -------
        np.ndarray
            The final weight vector after training.
        """
        for epoch in range(self.epochs):
            # Opcional: mezclar los datos en cada epoch mejora la convergencia
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            
            for idx in indices:
                x = data[idx]
                
                # 1. Calcular la salida de la neurona (y = suma(x_i * w_i))
                y = np.dot(x, self.weights)
                
                # 2. Actualizar los pesos usando la Regla de Oja: dw = lr * (y*x - y^2 * w)
                delta_w = self.lr * (y * x - (y**2) * self.weights)
                self.weights += delta_w
                
        # Al finalizar, normalizamos el vector de pesos para que su longitud sea 1
        # Esto nos permite compararlo directamente con el autovector de scikit-learn
        self.weights = self.weights / np.linalg.norm(self.weights)
        return self.weights