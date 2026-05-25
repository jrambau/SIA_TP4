import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
        
    def train(self, patterns):
        """Fase de Aprendizaje: Regla de Hebb."""
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        for p in patterns:
            p_flat = p.flatten()
            self.weights += np.dot(p_flat[:, None], p_flat[None, :])
            
        self.weights /= self.num_neurons
        np.fill_diagonal(self.weights, 0) 
        
    def sync_predict(self, noisy_pattern, max_iterations=10):
        """Fase de Recuperación: Todos los píxeles cambian a la vez."""
        state = np.copy(noisy_pattern).flatten()
        
        for iteration in range(max_iterations):
            prev_state = np.copy(state)
            suma = np.dot(self.weights, state)
            state = np.sign(suma)
            state[state == 0] = 1 
            
            # Convergencia: El paso actual es idéntico al anterior
            if np.array_equal(state, prev_state):
                print(f"   [Sincrónico] Convergió en iteración {iteration}")
                break
        return state.reshape(noisy_pattern.shape)

    # def async_predict(self, noisy_pattern, max_iterations=100):
    #     """Fase de Recuperación: Los píxeles cambian de a uno al azar."""
    #     state = np.copy(noisy_pattern).flatten()
    #     history = [np.copy(state)]
        
    #     for iteration in range(max_iterations):
    #         random_neuron = np.random.randint(0, self.num_neurons)
    #         sum = np.dot(self.weights[random_neuron], state)
    #         state[random_neuron] = 1 if sum >= 0 else -1
            
    #         history.append(np.copy(state))
            
    #         # Convergencia Asincrónica: Las últimas 'num_neurons' iteraciones fueron iguales
    #         if len(history) > self.num_neurons:
    #             if np.array_equal(history[-1], history[-self.num_neurons]):
    #                 print(f"   [Asincrónico] Convergió en iteración {iteration}")
    #                 break
    #     return state.reshape(noisy_pattern.shape)