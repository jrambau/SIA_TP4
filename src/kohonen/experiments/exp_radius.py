"""Experimento 4: Radio Inicial del Vecindario
===============================================
Compara el efecto del radio inicial R₀ sobre la calidad del mapa resultante.
Valores probados: 1.0, 2.5, 4.0, 5.0 (grilla completa).
Promedia resultados sobre 8 semillas distintas para robustez estadística.
Analiza:
- Error de cuantización (QE) y error topográfico (TE) con barras de error
- Estructura topológica resultante (U-matrices)
- Distribución de países en el mapa
- Densidad de agrupación por neurona
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from exp_utils import load_data, save_figure, draw_country_map
from kohonen import KohonenSOM


def main():
    sns.set_theme(style="whitegrid", context="talk")

    X_scaled, countries, feature_names = load_data()

    grid_size = 5
    epochs = 5000
    seeds = [42, 7, 2021, 123, 999, 0, 555, 321]

    radii = [1.0, 2.0, 3.0, 4.0]
    radius_keys = ['1', '2', '3', '4']
    radius_xlabels = ['R0=1', 'R0=2', 'R0=3', 'R0=4']

    results = {}

    for radius, key, xlabel in zip(radii, radius_keys, radius_xlabels):
        print(f"\nEntrenando SOM con R₀={radius} sobre {len(seeds)} semillas...")

        qe_list, te_list, util_list = [], [], []
        best_som = None
        min_qe = float('inf')

        for seed in seeds:
            np.random.seed(seed)
            som = KohonenSOM(grid_y=grid_size, grid_x=grid_size,
                             input_dim=X_scaled.shape[1],
                             epochs=epochs, learning_rate=0.1,
                             radius=radius, init_method='sample')
            som.train(X_scaled)

            qe = som.compute_quantization_error(X_scaled)
            te = som.compute_topographic_error(X_scaled)
            bmus = som.get_bmus(X_scaled)
            utilization = len(set(bmus)) / (grid_size ** 2) * 100

            qe_list.append(qe)
            te_list.append(te)
            util_list.append(utilization)

            if qe < min_qe:
                min_qe = qe
                best_som = som

        results[key] = {
            'som': best_som,
            'xlabel': xlabel,
            'radius': radius,
            'qe_mean': np.mean(qe_list),
            'qe_std': np.std(qe_list),
            'te_mean': np.mean(te_list),
            'te_std': np.std(te_list),
            'util_mean': np.mean(util_list),
            'util_std': np.std(util_list),
        }
        print(f"  QE: {results[key]['qe_mean']:.3f} ± {results[key]['qe_std']:.3f} | "
              f"TE: {results[key]['te_mean']:.3f} ± {results[key]['te_std']:.3f} | "
              f"Util: {results[key]['util_mean']:.1f}%")

    # ── Gráfico 1: Barras de métricas con error ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = sns.color_palette('magma', len(radius_keys))

    qe_m = [results[k]['qe_mean'] for k in radius_keys]
    qe_s = [results[k]['qe_std'] for k in radius_keys]
    te_m = [results[k]['te_mean'] for k in radius_keys]
    te_s = [results[k]['te_std'] for k in radius_keys]
    ut_m = [results[k]['util_mean'] for k in radius_keys]
    ut_s = [results[k]['util_std'] for k in radius_keys]
    xlabels = [results[k]['xlabel'] for k in radius_keys]

    axes[0].bar(xlabels, qe_m, yerr=qe_s, color=colors, capsize=5, alpha=0.9)
    axes[0].set_title('Error de Cuantización (QE)')
    axes[0].set_ylabel('QE Promedio')
    for i, (m, s) in enumerate(zip(qe_m, qe_s)):
        axes[0].text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=10)

    te_lower = np.minimum(te_m, te_s)
    axes[1].bar(xlabels, te_m, yerr=[te_lower, te_s], color=colors, capsize=5, alpha=0.9)
    axes[1].set_title('Error Topográfico (TE)')
    axes[1].set_ylabel('TE Promedio')
    axes[1].set_ylim(bottom=0)
    for i, (m, s) in enumerate(zip(te_m, te_s)):
        axes[1].text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=10)

    axes[2].bar(xlabels, ut_m, yerr=ut_s, color=colors, capsize=5, alpha=0.9)
    axes[2].set_title('Utilización de Neuronas')
    axes[2].set_ylabel('Utilización Promedio (%)')
    for i, (m, s) in enumerate(zip(ut_m, ut_s)):
        axes[2].text(i, m + s + 0.5, f'{m:.1f}%', ha='center', fontsize=10)

    fig.suptitle(f'Impacto del Radio Inicial R₀ (Promedio de {len(seeds)} corridas, grilla {grid_size}×{grid_size})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp4_radius_metrics.png')

    # ── Gráfico 2: Mapas de países — una imagen por radio ────────────────────
    for key in radius_keys:
        r = results[key]
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_country_map(ax, r['som'], X_scaled, countries, grid_size, grid_size, base_fontsize=9)
        ax.set_title(
            f'Mapa SOM — R₀ = {int(r["radius"])} | Grilla {grid_size}×{grid_size}\n'
            f'QE = {r["qe_mean"]:.3f} ± {r["qe_std"]:.3f}   TE = {r["te_mean"]:.3f} ± {r["te_std"]:.3f}',
            fontsize=12,
        )
        plt.tight_layout()
        save_figure(fig, f'exp4_radius_map_r{int(r["radius"])}.png')

    # ── Gráfico 3: U-Matrices ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, key in zip(axes, radius_keys):
        r = results[key]
        u_matrix = r['som'].get_u_matrix()
        sns.heatmap(u_matrix, cmap='magma', annot=True, fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Distancia Promedio'},
                    annot_kws={'size': 9})
        ax.set_title(f'U-Matrix  |  R₀ = {r["radius"]}')

    fig.suptitle('Matrices U por Radio Inicial', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp4_radius_umatrix.png')

    # ── Gráfico 4: Densidad de agrupación — una imagen por radio ─────────────
    for key in radius_keys:
        r = results[key]
        bmus = r['som'].get_bmus(X_scaled)
        count_matrix = np.zeros((grid_size, grid_size))
        for (y, x) in bmus:
            count_matrix[y, x] += 1
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(count_matrix, cmap='Blues', annot=True, fmt='g', ax=ax,
                    linewidths=.5, cbar_kws={'label': 'Cantidad de países'})
        ax.set_title(f'Densidad de Agrupación — R₀ = {int(r["radius"])} | Grilla {grid_size}×{grid_size}',
                     fontsize=13)
        plt.tight_layout()
        save_figure(fig, f'exp4_radius_density_r{int(r["radius"])}.png')

    print(f"\n[OK] Experimento 4 completado ({len(seeds)} seeds). Gráficos guardados.")


if __name__ == '__main__':
    main()
