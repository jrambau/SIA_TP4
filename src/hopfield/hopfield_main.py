import numpy as np

try:
    from .hopfield import HopfieldNetwork
    from .utils import add_noise, get_abc, plot_abc, plot_recovery
except Exception:
    from hopfield import HopfieldNetwork
    from utils import add_noise, get_abc, plot_abc, plot_recovery


def main():
    #np.random.seed(42)
    alphabet = get_abc()
    plot_abc(alphabet)

    base_letters = ["A", "I", "O", "X"]
    base_patterns = [alphabet[letter] for letter in base_letters]

    network = HopfieldNetwork(num_neurons=25)
    network.train(base_patterns)

    test_pattern = alphabet["I"]
    noisy_pattern = add_noise(test_pattern, noise_level=0.2)
    recovered_pattern = network.sync_predict(noisy_pattern)

    plot_recovery(
        test_pattern,
        noisy_pattern,
        recovered_pattern,
        "Hopfield learning demo: A trained and recovered",
        "hopfield_learning_demo.png",
    )

    print("Hopfield network initialized and trained with base patterns: A, I, O, X.")
    print("One recovery experiment completed for pattern A.")


if __name__ == "__main__":
    main()