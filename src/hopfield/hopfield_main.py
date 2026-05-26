import numpy as np

try:
    from .hopfield import HopfieldNetwork
    from .utils import add_noise, get_abc, orthogonality_test, plot_abc, plot_recovery
except Exception:
    from hopfield import HopfieldNetwork
    from utils import add_noise, get_abc, orthogonality_test, plot_abc, plot_recovery


def main():
    #np.random.seed(42)
    alphabet = get_abc()
    plot_abc(alphabet)

    base_letters = ["A", "L", "O", "W"]
    base_patterns = [alphabet[letter] for letter in base_letters]
    orthogonality_test(base_letters, base_patterns)

    network = HopfieldNetwork(num_neurons=25)
    network.train(base_patterns)

    test_pattern = alphabet["L"]
    noisy_pattern = add_noise(test_pattern, noise_level=0.1)
    recovered_pattern, history, energies = network.sync_predict(noisy_pattern)

    plot_recovery(
        test_pattern,
        noisy_pattern,
        recovered_pattern,
        "Hopfield learning demo: L trained and recovered",
        "hopfield_learning_demo.png",
    )

    print("Hopfield network initialized and trained with base patterns: A, L, O, W.")
    print("One recovery experiment completed for pattern L.")


if __name__ == "__main__":
    main()