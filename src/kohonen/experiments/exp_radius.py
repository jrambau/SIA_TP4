"""Experimento 3: Radio Inicial del Vecindario
===============================================
Compara el efecto del radio inicial del vecindario gaussiano.
Valores probados: 1, 2, grid_size/2 (default), grid_size.
Analiza:
- Estructura topológica resultante
- Matrices U (fronteras entre clusters)
- Mapas de países
- Métricas de calidad (QE y TE)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from exp_utils import load_data, save_figure, draw_country_map
from kohonen import KohonenSOM


def main():
    np.random.seed(42)
    X_scaled, countries, feature_names = load_data()

    grid_size = 4
    epochs = 5000
    radii = [1.0, 2.0, grid_size / 2, float(grid_size)]
    radius_labels = ['1.0', '2.0', f'{grid_size/2:.1f} (default)', f'{grid_size:.1f}']
    results = {}

    for radius, label in zip(radii, radius_labels):
        print(f"\nEntrenando SOM con radio inicial = {label}...")
        np.random.seed(42)
        som = KohonenSOM(grid_y=grid_size, grid_x=grid_size,
                         input_dim=X_scaled.shape[1],
                         epochs=epochs, learning_rate=0.1, radius=radius)
        som.train(X_scaled)

        qe = som.compute_quantization_error(X_scaled)
        te = som.compute_topographic_error(X_scaled)

        results[label] = {
            'som': som,
            'qe': qe,
            'te': te,
            'radius': radius,
        }
        print(f"  QE: {qe:.4f} | TE: {te:.4f}")

    # ── Gráfico 1: Mapas de países ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    for ax, label in zip(axes, radius_labels):
        r = results[label]
        draw_country_map(ax, r['som'], X_scaled, countries, grid_size, grid_size)
        ax.set_title(
            f'Radio = {label}\nQE={r["qe"]:.3f}  |  TE={r["te"]:.3f}',
            fontsize=12,
        )

    fig.suptitle('Experimento 3: Variación del Radio Inicial del Vecindario',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp3_radius_maps.png')

    # ── Gráfico 2: U-Matrices ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, label in zip(axes, radius_labels):
        u_matrix = results[label]['som'].get_u_matrix()
        sns.heatmap(u_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Distancia'})
        ax.set_title(f'U-Matrix  |  Radio = {label}')

    fig.suptitle('Matrices U por Radio Inicial', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp3_radius_umatrix.png')

    # ── Gráfico 3: Comparación de métricas ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = sns.color_palette('magma', len(radius_labels))
    qe_vals = [results[l]['qe'] for l in radius_labels]
    te_vals = [results[l]['te'] for l in radius_labels]

    short_labels = [f'r={results[l]["radius"]:.1f}' for l in radius_labels]

    axes[0].bar(short_labels, qe_vals, color=colors)
    axes[0].set_title('Error de Cuantización (QE)')
    axes[0].set_ylabel('QE')
    for i, v in enumerate(qe_vals):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    axes[1].bar(short_labels, te_vals, color=colors)
    axes[1].set_title('Error Topográfico (TE)')
    axes[1].set_ylabel('TE')
    for i, v in enumerate(te_vals):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    fig.suptitle('Metricas de Calidad por Radio Inicial', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp3_radius_metrics.png')

    # ── Gráfico 4: Densidad (cantidad de países por neurona) ─────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, label in zip(axes, radius_labels):
        r = results[label]
        bmus = r['som'].get_bmus(X_scaled)
        count_matrix = np.zeros((grid_size, grid_size))
        for (y, x) in bmus:
            count_matrix[y, x] += 1
        sns.heatmap(count_matrix, cmap='Blues', annot=True, fmt='g', ax=ax,
                    cbar_kws={'label': 'Cantidad de paises'})
        ax.set_title(f'Densidad  |  Radio = {label}')

    fig.suptitle('Densidad de Agrupacion por Radio Inicial', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp3_radius_density.png')

    print("\n[OK] Experimento 3 completado. Graficos guardados.")


if __name__ == '__main__':
    main()
