"""
Hopfield Network — Experimentos completos
==========================================
Exp 0: Mundo del revés — A invertida como atractor espejo
Exp 1: Consigna (a) — Recuperar 4 letras con ruido, mostrando cada paso
Exp 2: Consigna (b) — Encontrar un estado espureo
Exp 3: Variación — Letras muy similares (D, O, Q, C)
Exp 4: Variación — Letras ortogonales (T, X, L, O)
Exp 5: Variación — Saturación de capacidad (4→10 patrones)
Exp 6: Variación — Comparación sincrónico vs asincrónico
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
import sys

# Agregar el path del proyecto para poder importar como paquete
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.hopfield.hopfield import HopfieldNetwork
from src.hopfield.utils import (
    get_abc, add_noise, orthogonality_test,
    plot_abc, plot_recovery, plot_step_by_step,
    plot_energy_evolution, is_spurious, find_closest_pattern,
    plot_spurious_comparison, plot_capacity_results,
    plot_multi_recovery, get_hopfield_output_path,
)

SEPARATOR = "=" * 70

# ──────────────────────────────────────────────────────────────────────
#  EXPERIMENTO 0 — Mundo del Revés: atractores espejo
# ──────────────────────────────────────────────────────────────────────
def experiment_0(alphabet):
    print(f"\n{SEPARATOR}")
    print("EXPERIMENTO 1 — Mundo del Revés")
    print("Entrenar con A y pasar A invertida (−A) sin ruido.")
    print(SEPARATOR)

    pattern = alphabet["A"]
    inverted_pattern = -pattern

    net = HopfieldNetwork(num_neurons=25)
    net.train([pattern])

    recovered, history, energies = net.sync_predict(inverted_pattern)

    exact_inverted = np.array_equal(recovered, inverted_pattern)
    exact_original = np.array_equal(recovered, pattern)
    print(f"  Entrada: −A")
    print(f"  Recuperada: {'−A' if exact_inverted else 'otra'}")
    print(f"  Coincide con −A: {'SI' if exact_inverted else 'NO'}")
    print(f"  Coincide con A: {'SI' if exact_original else 'NO'}")

    plot_recovery(
        pattern,
        inverted_pattern,
        recovered,
        "Exp 1 — A invertida como atractor espejo",
        "exp1_mundo_del_reves.png",
        input_label="Patrón invertido (Entrada)",
        output_label="Salida de la red",
    )
    plot_step_by_step(
        history,
        "Exp 1 — Recuperación de A invertida",
        "exp1_mundo_del_reves_steps.png",
        energies=energies,
    )
    plot_energy_evolution(
        energies,
        "Exp 1 — Energía para A invertida",
        "exp1_mundo_del_reves_energy.png",
    )


# ──────────────────────────────────────────────────────────────────────
#  EXPERIMENTO 1 — Consigna (a): Almacenar 4 patrones, recuperar con
#  ruido mostrando cada paso
# ──────────────────────────────────────────────────────────────────────
def experiment_1(alphabet):
    print(f"\n{SEPARATOR}")
    print("EXPERIMENTO 1 — Consigna (a)")
    print("Almacenar {I, J, O, R}. Recuperar con ruido. Mostrar cada paso.")
    print(SEPARATOR)

    letters = ["I", "J", "O", "R"]
    patterns = [alphabet[l] for l in letters]
    orthogonality_test(letters, patterns)

    net = HopfieldNetwork(num_neurons=25)
    net.train(patterns)

    noise_levels = [0.1, 0.2, 0.3, 0.4]

    for letter, pattern in zip(letters, patterns):
        for nl in noise_levels:
            noisy = add_noise(pattern, noise_level=nl)
            recovered, history, energies = net.sync_predict(noisy)

            ok = np.array_equal(recovered, pattern)
            tag = "OK" if ok else "FALLO"
            print(f"  Letra {letter} | Ruido {int(nl*100)}% | "
                  f"Pasos={len(history)-1} | {tag}")

            plot_step_by_step(
                history,
                f"Recuperación paso a paso: {letter} (ruido {int(nl*100)}%)",
                f"exp1_steps_{letter}_noise{int(nl*100)}.png",
                energies=energies,
            )
            plot_energy_evolution(
                energies,
                f"Energía por paso: {letter} (ruido {int(nl*100)}%)",
                f"exp1_energy_{letter}_noise{int(nl*100)}.png",
            )

    # Resumen visual: las 4 letras con 20% de ruido
    results = []
    for letter, pattern in zip(letters, patterns):
        noisy = add_noise(pattern, noise_level=0.2)
        recovered, _, _ = net.sync_predict(noisy)
        ok = np.array_equal(recovered, pattern)
        results.append((letter, pattern, noisy, recovered, ok))

    plot_multi_recovery(results, "Exp 1 — Recuperación de {I,J,O,R} con 20% ruido",
                        "exp1_summary_20pct.png")


# ──────────────────────────────────────────────────────────────────────
#  EXPERIMENTO 2 — Consigna (b): Encontrar un estado espureo
# ──────────────────────────────────────────────────────────────────────
def experiment_2(alphabet):
    print(f"\n{SEPARATOR}")
    print("EXPERIMENTO 2 — Consigna (b)")
    print("Ingresar un patrón muy ruidoso e identificar un estado espureo.")
    print(SEPARATOR)

    letters = ["I", "J", "O", "R"]
    patterns = [alphabet[l] for l in letters]

    net = HopfieldNetwork(num_neurons=25)
    net.train(patterns)

    # Estrategia 1: ruido alto sobre cada letra
    found = False
    for attempt in range(200):
        # Elegir una letra al azar y aplicar mucho ruido
        idx = np.random.randint(0, len(patterns))
        nl = np.random.uniform(0.4, 0.6)
        noisy = add_noise(patterns[idx], noise_level=nl)
        recovered, history, energies = net.sync_predict(noisy, max_iterations=30)

        if is_spurious(recovered, patterns):
            closest, hamming, dot = find_closest_pattern(
                recovered, letters, patterns)
            print(f"  ¡ESTADO ESPUREO encontrado! (intento {attempt+1})")
            print(f"    Base: {letters[idx]} con {int(nl*100)}% ruido")
            print(f"    Más cercano a '{closest}' (Hamming={hamming}, dot={dot})")
            print(f"    Energía final: {energies[-1]:.4f}")

            plot_spurious_comparison(
                patterns, letters, noisy, recovered,
                f"Estado Espureo (base={letters[idx]}, ruido={int(nl*100)}%)",
                "exp2_spurious_state.png",
            )
            plot_step_by_step(
                history,
                f"Pasos hacia estado espureo (base={letters[idx]})",
                "exp2_spurious_steps.png",
                energies=energies,
            )
            found = True
            break

    # Estrategia 2: combinaciones de patrones (mezclas)
    if not found:
        print("  Probando mezclas de patrones...")
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                # Promedio de dos patrones → signo
                mix = np.sign(patterns[i].astype(float) + patterns[j].astype(float))
                mix[mix == 0] = 1
                recovered, history, energies = net.sync_predict(mix, max_iterations=30)

                if is_spurious(recovered, patterns):
                    print(f"  ¡ESTADO ESPUREO con mezcla {letters[i]}+{letters[j]}!")
                    closest, hamming, dot = find_closest_pattern(
                        recovered, letters, patterns)
                    print(f"    Más cercano a '{closest}' (Hamming={hamming})")

                    plot_spurious_comparison(
                        patterns, letters, mix, recovered,
                        f"Estado Espureo (mezcla {letters[i]}+{letters[j]})",
                        "exp2_spurious_state.png",
                    )
                    plot_step_by_step(
                        history,
                        f"Pasos hacia espureo (mezcla {letters[i]}+{letters[j]})",
                        "exp2_spurious_steps.png",
                        energies=energies,
                    )
                    found = True
                    break
            if found:
                break

    # Estrategia 3: patrón aleatorio puro
    if not found:
        print("  Probando patrones aleatorios puros...")
        for attempt in range(500):
            random_pat = np.random.choice([-1, 1], size=(5, 5))
            recovered, history, energies = net.sync_predict(random_pat, max_iterations=30)

            if is_spurious(recovered, patterns):
                closest, hamming, dot = find_closest_pattern(
                    recovered, letters, patterns)
                print(f"  ¡ESTADO ESPUREO con patrón aleatorio! (intento {attempt+1})")
                print(f"    Más cercano a '{closest}' (Hamming={hamming})")

                plot_spurious_comparison(
                    patterns, letters, random_pat, recovered,
                    "Estado Espureo (patrón aleatorio)",
                    "exp2_spurious_state.png",
                )
                plot_step_by_step(
                    history,
                    "Pasos hacia espureo (patrón aleatorio)",
                    "exp2_spurious_steps.png",
                    energies=energies,
                )
                found = True
                break

    if not found:
        print("  No se encontró estado espureo en los intentos realizados.")


# ──────────────────────────────────────────────────────────────────────
#  EXPERIMENTO 3 — Letras muy similares (D, O, Q, C)
# ──────────────────────────────────────────────────────────────────────
def experiment_3(alphabet):
    print(f"\n{SEPARATOR}")
    print("EXPERIMENTO 3 — Variación: Letras similares {H, M, N, W}")
    print("Demostrar confusión entre patrones de baja ortogonalidad.")
    print(SEPARATOR)

    letters = ["H", "M", "N", "W"]
    patterns = [alphabet[l] for l in letters]
    orthogonality_test(letters, patterns)

    net = HopfieldNetwork(num_neurons=25)
    net.train(patterns)

    results = []
    for letter, pattern in zip(letters, patterns):
        noisy = add_noise(pattern, noise_level=0.0)
        recovered, history, energies = net.sync_predict(noisy)
        ok = np.array_equal(recovered, pattern)
        closest, hamming, _ = find_closest_pattern(recovered, letters, patterns)
        tag = "OK" if ok else f"CONFUNDIO con {closest}"
        print(f"  {letter} -> {tag} (Hamming al mas cercano: {hamming})")
        results.append((letter, pattern, noisy, recovered, ok))

    plot_multi_recovery(results, "Exp 3 — Letras similares {H,M,N,W} con 0% ruido",
                        "exp3_similar_letters.png")


# ──────────────────────────────────────────────────────────────────────
#  EXPERIMENTO 4 — Letras ortogonales (T, X, L, O)
# ──────────────────────────────────────────────────────────────────────
def experiment_4(alphabet):
    print(f"\n{SEPARATOR}")
    print("EXPERIMENTO 4 — Variación: Letras ortogonales {T, X, L, O}")
    print("Demostrar mejor recuperación con alta ortogonalidad.")
    print(SEPARATOR)

    letters = ["T", "X", "L", "O"]
    patterns = [alphabet[l] for l in letters]
    orthogonality_test(letters, patterns)

    net = HopfieldNetwork(num_neurons=25)
    net.train(patterns)

    # Probar con ruido más alto
    for nl in [0.2, 0.3, 0.4]:
        results = []
        for letter, pattern in zip(letters, patterns):
            noisy = add_noise(pattern, noise_level=nl)
            recovered, _, _ = net.sync_predict(noisy)
            ok = np.array_equal(recovered, pattern)
            results.append((letter, pattern, noisy, recovered, ok))

        n_ok = sum(1 for r in results if r[4])
        print(f"  Ruido {int(nl*100)}%: {n_ok}/{len(results)} recuperaciones correctas")

        plot_multi_recovery(
            results,
            f"Exp 4 — Letras ortogonales {{T,X,L,O}} con {int(nl*100)}% ruido",
            f"exp4_orthogonal_noise{int(nl*100)}.png")


# ──────────────────────────────────────────────────────────────────────
#  EXPERIMENTO 5 — Saturación de capacidad
# ──────────────────────────────────────────────────────────────────────
def experiment_5(alphabet):
    print(f"\n{SEPARATOR}")
    print("EXPERIMENTO 5 — Capacidad: ¿cuántos patrones puede almacenar?")
    print("Aumentar de 4 a 10 y medir tasa de recuperación.")
    print(SEPARATOR)
    all_letters = list(alphabet.keys())
    num_stored_list = [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    trials = 80  # repeticiones por configuración (aumentar para más estabilidad)

    summary_means = []
    summary_stds = []
    best_combos = []

    out_rows = []

    for n_stored in num_stored_list:
        per_trial_acc = []  # accuracy per trial for this n
        best_combo = None
        best_combo_score = -1.0

        for _ in range(trials):
            # Elegir n_stored letras al azar
            chosen = list(np.random.choice(all_letters, n_stored, replace=False))
            patterns = [alphabet[l] for l in chosen]

            net = HopfieldNetwork(num_neurons=25)
            net.train(patterns)

            # Probar recuperación de cada patrón con 20% ruido
            correct_in_trial = 0
            for pat in patterns:
                noisy = add_noise(pat, noise_level=0.2)
                recovered, _, _ = net.sync_predict(noisy, max_iterations=20)
                if np.array_equal(recovered, pat):
                    correct_in_trial += 1

            acc_trial = 100.0 * correct_in_trial / n_stored
            per_trial_acc.append(acc_trial)

            # track best combo observed
            if acc_trial > best_combo_score:
                best_combo_score = acc_trial
                best_combo = ("".join(chosen), int(correct_in_trial), n_stored)

        mean_acc = float(np.mean(per_trial_acc))
        std_acc = float(np.std(per_trial_acc))
        summary_means.append(mean_acc)
        summary_stds.append(std_acc)
        best_combos.append((best_combo, best_combo_score))

        out_rows.append((n_stored, mean_acc, std_acc, best_combo[0], best_combo[1], best_combo[2]))
        print(f"  {n_stored} patrones: media={mean_acc:.1f}% std={std_acc:.1f}% "
              f"(mejor combo observado: {best_combo[0]} -> {best_combo_score:.1f}% [{best_combo[1]}/{best_combo[2]}])")

    # Save CSV with summarized results
    out_csv = get_hopfield_output_path('exp5_capacity_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_stored', 'mean_acc_pct', 'std_acc_pct', 'best_combo', 'best_combo_correct', 'best_combo_n'])
        for row in out_rows:
            writer.writerow(row)
    print(f"Saved capacity summary to {out_csv}")

# Plot mean +/- std
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(num_stored_list)
    
    # 1. Calculamos los errores asimétricos para no pasarnos de 0 ni de 100
    means = np.array(summary_means)
    stds = np.array(summary_stds)
    lower_error = np.minimum(stds, means) 
    upper_error = np.minimum(stds, 100.0 - means)
    errores_asimetricos = [lower_error, upper_error]

    # 2. Graficamos con los nuevos errores asimétricos
    ax.errorbar(x, means, yerr=errores_asimetricos, fmt='-o', color='#2c3e50', ecolor='#95a5a6', capsize=4)
    ax.set_xlabel('Número de patrones almacenados')
    ax.set_ylabel('Recuperación media (%)')
    ax.set_title('Prueba de capacidad: recuperación media ± std (20% ruido)')
    
    # 3. Bloqueamos el eje Y para que el gráfico quede prolijo
    ax.set_ylim(-5, 105)

    # 4. Anotamos los porcentajes justo arriba del límite superior del error
    for xi, m, ue in zip(x, means, upper_error):
        ax.text(xi, m + ue + 2, f'{m:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    out_png = get_hopfield_output_path('exp5_capacity_summary.png')
    plt.savefig(out_png, dpi=150)
    print(f"Saved capacity plot to {out_png}")

# ──────────────────────────────────────────────────────────────────────
#  EXPERIMENTO 6 — Sincrónico vs Asincrónico
# ──────────────────────────────────────────────────────────────────────
def experiment_6(alphabet):
    print(f"\n{SEPARATOR}")
    print("EXPERIMENTO 6 — Comparación: Sincrónico vs Asincrónico")
    print(SEPARATOR)

    letters = ["A", "L", "O", "W"]
    patterns = [alphabet[l] for l in letters]

    net = HopfieldNetwork(num_neurons=25)
    net.train(patterns)

    for letter, pattern in zip(letters, patterns):
        noisy = add_noise(pattern, noise_level=0.3)

        # Sincrónico
        rec_sync, hist_sync, en_sync = net.sync_predict(noisy.copy())
        ok_sync = np.array_equal(rec_sync, pattern)

        # Asincrónico (misma entrada)
        rec_async, hist_async, en_async = net.async_predict(noisy.copy(),
                                                              max_iterations=1000)
        ok_async = np.array_equal(rec_async, pattern)

        print(f"  {letter}: Sync={'OK' if ok_sync else 'FALLO'} "
              f"({len(hist_sync)-1} pasos) | "
              f"Async={'OK' if ok_async else 'FALLO'} "
              f"({len(hist_async)-1} épocas)")

        # Plot comparativo
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))

        titles_row = ["Original", "Ruidoso (30%)", "Sync Result", "Async Result"]
        imgs_row1 = [pattern, noisy, rec_sync, rec_async]
        for j, (ax, img, t) in enumerate(zip(axes[0], imgs_row1, titles_row)):
            ax.imshow(img, cmap='binary', vmin=-1, vmax=1)
            ax.set_title(t, fontsize=10)
            ax.axis('off')
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

        # Energía sync
        axes[1][0].plot(en_sync, 'o-', color='#3498db')
        axes[1][0].set_title("Energía Sync", fontsize=10)
        axes[1][0].set_xlabel("Paso")
        axes[1][0].grid(True, alpha=0.3)

        # Energía async
        axes[1][1].plot(en_async, 's-', color='#e74c3c')
        axes[1][1].set_title("Energía Async", fontsize=10)
        axes[1][1].set_xlabel("Época")
        axes[1][1].grid(True, alpha=0.3)

        # Apagar los dos restantes
        axes[1][2].axis('off')
        axes[1][3].axis('off')

        plt.suptitle(f"Exp 6 — Sync vs Async: Letra {letter} (30% ruido)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_path = get_hopfield_output_path(f"exp6_sync_vs_async_{letter}.png")
        plt.savefig(out_path, dpi=150)
        print(f"  Guardado: {out_path}")
        plt.close()


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────
def main(selected_experiment=None):
    np.random.seed(42)
    alphabet = get_abc()
    plot_abc(alphabet)

    experiments = {
        0: experiment_0,
        1: experiment_1,
        2: experiment_2,
        3: experiment_3,
        4: experiment_4,
        5: experiment_5,
        6: experiment_6,
    }

    if selected_experiment is None:
        for exp_id in sorted(experiments):
            experiments[exp_id](alphabet)
    else:
        experiments[selected_experiment](alphabet)

    print(f"\n{SEPARATOR}")
    if selected_experiment is None:
        print("¡Todos los experimentos completados!")
    else:
        print(f"¡Experimento {selected_experiment} completado!")
    print(f"Resultados en: outputs/hopfield/")
    print(SEPARATOR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecutar uno o todos los experimentos de Hopfield."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        type=int,
        choices=range(0, 7),
        help="Número de experimento a ejecutar (0-6). Si se omite, ejecuta todos.",
    )
    args = parser.parse_args()
    main(args.experiment)
