import itertools
import csv
import numpy as np
import os
import sys

# allow running from repo root: add project root so `src` imports resolve
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.hopfield.utils import get_abc, get_hopfield_output_path
from src.hopfield.hopfield import HopfieldNetwork
import matplotlib.pyplot as plt


def quartet_orthogonality_score(patterns):
    # patterns: list of 4 arrays flattened
    dots = np.array([[np.dot(p, q) for q in patterns] for p in patterns])
    # sum absolute off-diagonal
    diag = np.abs(np.diag(dots))
    score = np.sum(np.abs(dots)) - np.sum(diag)
    return score


def benchmark_quartet(alphabet, quartet, noise_level=0.2, repeats=5):
    patterns = [alphabet[l] for l in quartet]
    names = list(quartet)
    net = HopfieldNetwork(num_neurons=25)
    net.train(patterns)

    correct = 0
    total = 0
    failures = []  # list of (pattern_name, trial_idx, num_flips, hamming)
    trial_idx = 0
    for r in range(repeats):
        for name, pat in zip(names, patterns):
            trial_idx += 1
            noisy = pat.copy()
            # apply noise by flipping a fraction of bits
            flat = noisy.flatten()
            num_flips = int(len(flat) * noise_level)
            if num_flips > 0:
                idx = np.random.choice(len(flat), num_flips, replace=False)
                flat[idx] *= -1
                noisy = flat.reshape(pat.shape)
            recovered, _, _ = net.sync_predict(noisy)
            total += 1
            if np.array_equal(recovered, pat):
                correct += 1
            else:
                # compute Hamming distance
                hamming = int(np.sum(recovered.flatten() != pat.flatten()))
                failures.append((name, trial_idx, num_flips, hamming))

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, correct, total, failures


if __name__ == '__main__':
    np.random.seed(42)
    alphabet = get_abc()
    letters = list(alphabet.keys())

    combos = list(itertools.combinations(letters, 4))
    print(f"Evaluating {len(combos)} four-letter combinations (orthogonality metric)...")

    scores = []
    # compute flattened patterns once
    flat_patterns = {l: alphabet[l].flatten() for l in letters}
    for combo in combos:
        pats = [flat_patterns[l] for l in combo]
        score = quartet_orthogonality_score(pats)
        scores.append((combo, int(score)))

    scores.sort(key=lambda x: x[1])

    top_k = 12
    top = scores[:top_k]

    print(f"Top {top_k} quartets by orthogonality score (lower = more orthogonal):")
    for i, (combo, score) in enumerate(top, start=1):
        print(f"  {i}. {combo}: {score}")

    # Save a theoretical Top-12 plot (orthogonality scores)
    th_labels = ["".join(c) for c, _ in top]
    th_scores = [s for _, s in top]
    fig_t, ax_t = plt.subplots(figsize=(8, 6))
    y = np.arange(len(th_labels))
    ax_t.barh(y, th_scores, color='#3498db')
    ax_t.set_yticks(y)
    ax_t.set_yticklabels(th_labels, fontsize=10)
    ax_t.invert_yaxis()
    ax_t.set_xlabel('Puntuación ortogonal (menor es mejor)')
    ax_t.set_title(f'Top {top_k} — ortogonalidad teórica (menor es mejor)')
    for i, v in enumerate(th_scores):
        ax_t.text(v + 0.5, i, str(v), va='center', fontsize=9)
    plt.tight_layout()
    out_th_png = get_hopfield_output_path('best_quartets_theoretical.png')
    plt.savefig(out_th_png, dpi=150)
    print(f"Saved theoretical top-12 plot to {out_th_png}")

    # Benchmark top N quartets empirically
    results = []
    for idx, (combo, score) in enumerate(top, start=1):
        acc, correct, total, failures = benchmark_quartet(alphabet, combo, noise_level=0.2, repeats=100)
        results.append((combo, score, acc, correct, total, failures))
        print(f"Benchmark {idx}. {combo}: score={score}, recovery_acc={acc:.1f}% ({correct}/{total})")
        if failures:
            for name, trial_idx, num_flips, hamming in failures:
                print(f"    FAIL: pattern={name}, trial={trial_idx}, flips={num_flips}, hamming={hamming}")

    # Save CSV
    out_csv = get_hopfield_output_path('best_quartets.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['quartet', 'orth_score', 'recovery_acc_pct', 'correct_count', 'total_trials'])
        for combo, score, acc, correct, total, failures in results:
            writer.writerow(["".join(combo), score, acc, correct, total])
    print(f"Saved results to {out_csv}")

    # Plot stacked bar chart of correct vs incorrect counts and annotate percent success
    labels = ["".join(r[0]) for r in results]
    corrects = [r[3] for r in results]
    totals = [r[4] for r in results]
    incorrects = [t - c for c, t in zip(corrects, totals)]

    x = np.arange(len(labels))
    width = 0.6
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, corrects, width, label='Correcto', color='#2ecc71')
    ax.bar(x, incorrects, width, bottom=corrects, label='Incorrecto', color='#e74c3c')
    ax.set_ylabel('Ensayos (correctos / incorrectos)')
    ax.set_title(f'Top {top_k}: recuento de recuperaciones y tasa de éxito (20% ruido)', pad=18, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=10)

    # adjust layout to leave space for the title and percent labels
    ymax = max(totals) + 4
    ax.set_ylim(0, ymax)
    fig.subplots_adjust(top=0.88)

    # place legend at bottom-left, add bottom margin so it doesn't overlap x labels
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=10)
    fig.subplots_adjust(bottom=0.15)

    # annotate each bar: percent above bar (not touching title) and counts inside
    for i, (c, t) in enumerate(zip(corrects, totals)):
        pct = 100.0 * c / t if t > 0 else 0.0
        # percent label slightly above the top of the stacked bar
        ax.text(i, t + 0.6, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        # counts inside the green portion (correct)
        ax.text(i, max(0.6, c / 2), f'{c}/{t}', ha='center', va='center', color='white', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = get_hopfield_output_path('best_quartets_accuracy.png')
    plt.savefig(out_png, dpi=150)
    print(f"Saved accuracy plot to {out_png}")
