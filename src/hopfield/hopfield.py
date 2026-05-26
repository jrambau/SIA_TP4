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

    def compute_energy(self, state):
        """Calcula la energía del estado actual: E = -0.5 * s^T * W * s"""
        s = state.flatten()
        return -0.5 * np.dot(s, np.dot(self.weights, s))

    def sync_predict(self, noisy_pattern, max_iterations=20):
        """
        Fase de Recuperación sincrónica: todos los píxeles cambian a la vez.
        Devuelve (estado_final, historial_de_estados, historial_de_energías).
        """
        state = np.copy(noisy_pattern).flatten()
        shape = noisy_pattern.shape

        history = [state.reshape(shape).copy()]
        energies = [self.compute_energy(state)]

        for iteration in range(max_iterations):
            prev_state = np.copy(state)
            suma = np.dot(self.weights, state)
            state = np.sign(suma)
            state[state == 0] = 1

            history.append(state.reshape(shape).copy())
            energies.append(self.compute_energy(state))

            # Convergencia: El paso actual es idéntico al anterior
            if np.array_equal(state, prev_state):
                #print(f"   [Sincrónico] Convergió en iteración {iteration}")
                break

        return state.reshape(shape), history, energies

    def async_predict(self, noisy_pattern, max_iterations=500):
        """
        Fase de Recuperación asincrónica: los píxeles cambian de a uno al azar.
        Devuelve (estado_final, historial_de_estados, historial_de_energías).
        Se guarda un snapshot cada num_neurons actualizaciones (= 1 "época").
        """
        state = np.copy(noisy_pattern).flatten()
        shape = noisy_pattern.shape

        history = [state.reshape(shape).copy()]
        energies = [self.compute_energy(state)]

        for iteration in range(max_iterations):
            # Actualizar una neurona al azar
            random_neuron = np.random.randint(0, self.num_neurons)
            h = np.dot(self.weights[random_neuron], state)
            state[random_neuron] = 1 if h >= 0 else -1

            # Snapshot cada "época" (num_neurons actualizaciones individuales)
            if (iteration + 1) % self.num_neurons == 0:
                history.append(state.reshape(shape).copy())
                energies.append(self.compute_energy(state))

            # Convergencia: comparar con el estado de hace 2 épocas
            if len(history) > 2 and np.array_equal(history[-1], history[-2]):
                print(f"   [Asincrónico] Convergió en iteración {iteration}")
                break

        return state.reshape(shape), history, energies