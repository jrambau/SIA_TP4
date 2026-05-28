"""Experimento 7: Grilla Rectangular vs No Cuadrada
====================================================
Compara redes SOM con la misma cantidad total de neuronas (16) pero
diferentes formas de grilla: 4x4, 2x8, 8x2, 1x16.
Analiza:
- Cómo la topología de la grilla afecta la organización de los clusters
- Diferencias en U-matrices y distribución de países
- Métricas de calidad (QE y TE)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from exp_utils import load_data, save_figure
from kohonen import KohonenSOM


def draw_country_map_rect(ax, som, X_scaled, countries, grid_y, grid_x, fontsize=7):
    """Draw country map adapted for rectangular (non-square) grids."""
    bmus = som.get_bmus(X_scaled)

    grid_countries = {}
    for y in range(grid_y):
        for x in range(grid_x):
            grid_countries[(y, x)] = []
    for country, bmu in zip(countries, bmus):
        grid_countries[bmu].append(country)

    ax.set_xlim(-0.5, grid_x - 0.5)
    ax.set_ylim(grid_y - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, grid_x, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_y, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])

    for (y, x), country_list in grid_countries.items():
        if country_list:
            text = "\n".join(country_list)
            ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                    bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray'))


def main():
    np.random.seed(42)
    X_scaled, countries, feature_names = load_data()

    epochs = 5000
    # All shapes have 16 neurons total
    shapes = [(4, 4), (2, 8), (8, 2), (1, 16)]
    shape_labels = [f'{y}×{x}' for y, x in shapes]
    results = {}

    for (gy, gx), label in zip(shapes, shape_labels):
        print(f"\nEntrenando SOM con grilla {label}...")
        np.random.seed(42)
        som = KohonenSOM(grid_y=gy, grid_x=gx, input_dim=X_scaled.shape[1],
                         epochs=epochs, learning_rate=0.1)
        som.train(X_scaled)

        qe = som.compute_quantization_error(X_scaled)
        te = som.compute_topographic_error(X_scaled)
        bmus = som.get_bmus(X_scaled)
        active = set(bmus)
        utilization = len(active) / (gy * gx) * 100

        results[label] = {
            'som': som,
            'qe': qe,
            'te': te,
            'utilization': utilization,
            'grid_y': gy,
            'grid_x': gx,
        }
        print(f"  QE: {qe:.4f} | TE: {te:.4f} | Utilización: {utilization:.1f}%")

    # ── Gráfico 1: Mapas de países ───────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))

    # Use different subplot ratios for different grid shapes
    grid_spec_positions = [
        (4, 4, (1, 2, 5, 6)),   # 4x4: top-left 2x2
        (4, 4, (3, 4)),          # 2x8: top-right 1x2
        (4, 4, (7, 8)),          # 8x2: bottom-right 1x2
        (4, 4, (9, 10, 11, 12)), # 1x16: bottom full row
    ]

    # Simpler approach: use subplots with varying sizes
    axes = []
    for i, ((gy, gx), label) in enumerate(zip(shapes, shape_labels)):
        ax = fig.add_subplot(2, 2, i + 1)
        axes.append(ax)

    for ax, label in zip(axes, shape_labels):
        r = results[label]
        draw_country_map_rect(ax, r['som'], X_scaled, countries,
                              r['grid_y'], r['grid_x'], fontsize=6)
        ax.set_title(
            f'Grilla {label}  |  QE={r["qe"]:.3f}  |  TE={r["te"]:.3f}\n'
            f'Utilización: {r["utilization"]:.1f}%',
            fontsize=11,
        )
        ax.set_aspect('equal')

    fig.suptitle('Experimento 7: Grillas con Diferentes Formas (16 neuronas)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp7_grid_shape_maps.png')

    # ── Gráfico 2: U-Matrices ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, label in zip(axes, shape_labels):
        r = results[label]
        u_matrix = r['som'].get_u_matrix()
        sns.heatmap(u_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Distancia'})
        ax.set_title(f'U-Matrix  |  {label}')

    fig.suptitle('Matrices U por Forma de Grilla', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp7_grid_shape_umatrix.png')

    # ── Gráfico 3: Comparación de métricas ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = sns.color_palette('Set2', len(shape_labels))
    qe_vals = [results[l]['qe'] for l in shape_labels]
    te_vals = [results[l]['te'] for l in shape_labels]
    util_vals = [results[l]['utilization'] for l in shape_labels]

    axes[0].bar(shape_labels, qe_vals, color=colors)
    axes[0].set_title('Error de Cuantización (QE)')
    axes[0].set_ylabel('QE')
    for i, v in enumerate(qe_vals):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    axes[1].bar(shape_labels, te_vals, color=colors)
    axes[1].set_title('Error Topográfico (TE)')
    axes[1].set_ylabel('TE')
    for i, v in enumerate(te_vals):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    axes[2].bar(shape_labels, util_vals, color=colors)
    axes[2].set_title('Utilización de Neuronas')
    axes[2].set_ylabel('Utilización (%)')
    for i, v in enumerate(util_vals):
        axes[2].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

    fig.suptitle('Métricas de Calidad por Forma de Grilla', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp7_grid_shape_metrics.png')

    print("\n[OK] Experimento 7 completado. Graficos guardados.")


if __name__ == '__main__':
    main()
