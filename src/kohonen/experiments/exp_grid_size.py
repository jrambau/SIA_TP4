"""Experimento 1: Variación del Tamaño de Grilla
================================================
Compara los resultados de entrenar SOM con grillas de 3x3, 4x4, 5x5 y 6x6.
Analiza:
- Agrupación de países en cada configuración
- Error de cuantización (QE) y error topográfico (TE)
- Porcentaje de utilización de neuronas
- Matrices U para cada tamaño
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from exp_utils import load_data, save_figure, draw_country_map
from kohonen import KohonenSOM


def main():
    np.random.seed(42)
    X_scaled, countries, feature_names = load_data()

    grid_sizes = [3, 4, 5, 6]
    results = {}

    for size in grid_sizes:
        print(f"\nEntrenando SOM {size}x{size}...")
        som = KohonenSOM(grid_y=size, grid_x=size, input_dim=X_scaled.shape[1],
                         epochs=5000, learning_rate=0.1)
        som.train(X_scaled)

        qe = som.compute_quantization_error(X_scaled)
        te = som.compute_topographic_error(X_scaled)
        bmus = som.get_bmus(X_scaled)

        # Count active neurons
        active = set(bmus)
        utilization = len(active) / (size * size) * 100

        results[size] = {
            'som': som,
            'qe': qe,
            'te': te,
            'utilization': utilization,
        }
        print(f"  QE: {qe:.4f} | TE: {te:.4f} | Utilización: {utilization:.1f}%")

    # ── Gráfico 1: Mapas de países para cada tamaño ──────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    for ax, size in zip(axes, grid_sizes):
        r = results[size]
        draw_country_map(ax, r['som'], X_scaled, countries, size, size)
        ax.set_title(
            f'Grilla {size}x{size}  |  QE={r["qe"]:.3f}  |  TE={r["te"]:.3f}\n'
            f'Utilización: {r["utilization"]:.1f}%',
            fontsize=12,
        )

    fig.suptitle('Experimento 1: Variación del Tamaño de Grilla', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp1_grid_size_maps.png')

    # ── Gráfico 2: Comparación de métricas ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sizes_labels = [f'{s}x{s}' for s in grid_sizes]
    qe_vals = [results[s]['qe'] for s in grid_sizes]
    te_vals = [results[s]['te'] for s in grid_sizes]
    util_vals = [results[s]['utilization'] for s in grid_sizes]
    colors = sns.color_palette('viridis', len(grid_sizes))

    axes[0].bar(sizes_labels, qe_vals, color=colors)
    axes[0].set_title('Error de Cuantización (QE)')
    axes[0].set_ylabel('QE')
    axes[0].set_xlabel('Tamaño de grilla')
    for i, v in enumerate(qe_vals):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    axes[1].bar(sizes_labels, te_vals, color=colors)
    axes[1].set_title('Error Topográfico (TE)')
    axes[1].set_ylabel('TE')
    axes[1].set_xlabel('Tamaño de grilla')
    for i, v in enumerate(te_vals):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    axes[2].bar(sizes_labels, util_vals, color=colors)
    axes[2].set_title('Utilización de Neuronas')
    axes[2].set_ylabel('Utilización (%)')
    axes[2].set_xlabel('Tamaño de grilla')
    for i, v in enumerate(util_vals):
        axes[2].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

    fig.suptitle('Métricas de Calidad por Tamaño de Grilla', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp1_grid_size_metrics.png')

    # ── Gráfico 3: U-Matrices ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, size in zip(axes, grid_sizes):
        u_matrix = results[size]['som'].get_u_matrix()
        sns.heatmap(u_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Distancia promedio'})
        ax.set_title(f'U-Matrix {size}x{size}')

    fig.suptitle('Matrices U por Tamaño de Grilla', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp1_grid_size_umatrix.png')

    print("\n[OK] Experimento 1 completado. Graficos guardados.")


if __name__ == '__main__':
    main()
