"""Experimento 5: Tipo de Inicialización de Pesos
==================================================
Compara tres estrategias de inicialización de pesos:
- Normal (N(0,1)) — la que usamos por defecto
- Uniforme (U(-1,1))
- PCA — inicializa la grilla sobre los dos primeros componentes principales
Analiza:
- Velocidad de convergencia (curvas QE)
- Calidad final de la agrupación
- Diferencias visuales en mapas y U-matrices
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from exp_utils import load_data, save_figure, draw_country_map
from kohonen import KohonenSOM


def main():
    X_scaled, countries, feature_names = load_data()

    grid_size = 4
    epochs = 5000
    init_methods = ['normal', 'uniform', 'pca']
    init_labels = {
        'normal': 'Normal N(0,1)',
        'uniform': 'Uniforme U(-1,1)',
        'pca': 'PCA',
    }
    results = {}

    for method in init_methods:
        print(f"\nEntrenando SOM con inicializacion: {method}...")
        np.random.seed(42)
        som = KohonenSOM(grid_y=grid_size, grid_x=grid_size,
                         input_dim=X_scaled.shape[1],
                         epochs=epochs, learning_rate=0.1,
                         init_method=method, track_qe=True)
        som.train(X_scaled)

        qe = som.compute_quantization_error(X_scaled)
        te = som.compute_topographic_error(X_scaled)

        results[method] = {
            'som': som,
            'qe': qe,
            'te': te,
            'qe_history': som.qe_history,
        }
        print(f"  QE: {qe:.4f} | TE: {te:.4f}")

    # ── Gráfico 1: Curvas de convergencia ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'normal': '#2196F3', 'uniform': '#FF9800', 'pca': '#4CAF50'}

    for method in init_methods:
        r = results[method]
        ax.plot(r['qe_history'], label=init_labels[method],
                color=colors[method], linewidth=1.5, alpha=0.85)

    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Error de Cuantización (QE)', fontsize=12)
    ax.set_title('Convergencia según Método de Inicialización', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    save_figure(fig, 'exp5_weight_init_convergence.png')

    # ── Gráfico 2: Mapas de países ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    for ax, method in zip(axes, init_methods):
        r = results[method]
        draw_country_map(ax, r['som'], X_scaled, countries, grid_size, grid_size, fontsize=8)
        ax.set_title(
            f'{init_labels[method]}\nQE={r["qe"]:.3f}  |  TE={r["te"]:.3f}',
            fontsize=12,
        )

    fig.suptitle('Experimento 5: Agrupación según Inicialización de Pesos',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp5_weight_init_maps.png')

    # ── Gráfico 3: U-Matrices ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, method in zip(axes, init_methods):
        u_matrix = results[method]['som'].get_u_matrix()
        sns.heatmap(u_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Distancia'})
        ax.set_title(f'U-Matrix  |  {init_labels[method]}')

    fig.suptitle('Matrices U por Método de Inicialización', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp5_weight_init_umatrix.png')

    # ── Gráfico 4: Comparación de métricas ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    labels = [init_labels[m] for m in init_methods]
    qe_vals = [results[m]['qe'] for m in init_methods]
    te_vals = [results[m]['te'] for m in init_methods]
    bar_colors = [colors[m] for m in init_methods]

    axes[0].bar(labels, qe_vals, color=bar_colors)
    axes[0].set_title('Error de Cuantización Final')
    axes[0].set_ylabel('QE')
    for i, v in enumerate(qe_vals):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    axes[1].bar(labels, te_vals, color=bar_colors)
    axes[1].set_title('Error Topográfico Final')
    axes[1].set_ylabel('TE')
    for i, v in enumerate(te_vals):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    fig.suptitle('Métricas Finales por Inicialización', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp5_weight_init_metrics.png')

    print("\n[OK] Experimento 5 completado. Graficos guardados.")


if __name__ == '__main__':
    main()
