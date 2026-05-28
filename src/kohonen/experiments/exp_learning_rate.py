"""Experimento 2: Impacto del Learning Rate Inicial
====================================================
Compara el entrenamiento con diferentes learning rates iniciales: 0.01, 0.1, 0.5, 1.0.
Analiza:
- Curvas de convergencia del error de cuantización
- Mapas de países resultantes
- Matrices U finales
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

    learning_rates = [0.01, 0.1, 0.5, 1.0]
    grid_size = 4
    epochs = 5000
    results = {}

    for lr in learning_rates:
        print(f"\nEntrenando SOM con learning rate = {lr}...")
        # Use the same seed for each run to isolate the LR effect
        np.random.seed(42)
        som = KohonenSOM(grid_y=grid_size, grid_x=grid_size,
                         input_dim=X_scaled.shape[1],
                         epochs=epochs, learning_rate=lr, track_qe=True)
        som.train(X_scaled)

        qe = som.compute_quantization_error(X_scaled)
        te = som.compute_topographic_error(X_scaled)

        results[lr] = {
            'som': som,
            'qe': qe,
            'te': te,
            'qe_history': som.qe_history,
        }
        print(f"  QE final: {qe:.4f} | TE: {te:.4f}")

    # ── Gráfico 1: Curvas de convergencia QE ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette('tab10', len(learning_rates))

    for (lr, r), color in zip(results.items(), colors):
        ax.plot(r['qe_history'], label=f'lr = {lr}', color=color, alpha=0.85)

    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Error de Cuantización (QE)', fontsize=12)
    ax.set_title('Convergencia del QE para diferentes Learning Rates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, epochs)
    plt.tight_layout()
    save_figure(fig, 'exp2_learning_rate_convergence.png')

    # ── Gráfico 2: Mapas de países ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    for ax, lr in zip(axes, learning_rates):
        r = results[lr]
        draw_country_map(ax, r['som'], X_scaled, countries, grid_size, grid_size)
        ax.set_title(f'lr = {lr}  |  QE={r["qe"]:.3f}  |  TE={r["te"]:.3f}', fontsize=12)

    fig.suptitle('Experimento 2: Agrupación de Países por Learning Rate', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp2_learning_rate_maps.png')

    # ── Gráfico 3: U-Matrices ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, lr in zip(axes, learning_rates):
        u_matrix = results[lr]['som'].get_u_matrix()
        sns.heatmap(u_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Distancia'})
        ax.set_title(f'U-Matrix  |  lr = {lr}')

    fig.suptitle('Matrices U por Learning Rate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp2_learning_rate_umatrix.png')

    # ── Gráfico 4: Comparación de métricas finales ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    lr_labels = [str(lr) for lr in learning_rates]
    qe_vals = [results[lr]['qe'] for lr in learning_rates]
    te_vals = [results[lr]['te'] for lr in learning_rates]

    axes[0].bar(lr_labels, qe_vals, color=colors)
    axes[0].set_title('Error de Cuantización Final')
    axes[0].set_ylabel('QE')
    axes[0].set_xlabel('Learning Rate')
    for i, v in enumerate(qe_vals):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    axes[1].bar(lr_labels, te_vals, color=colors)
    axes[1].set_title('Error Topográfico Final')
    axes[1].set_ylabel('TE')
    axes[1].set_xlabel('Learning Rate')
    for i, v in enumerate(te_vals):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    fig.suptitle('Métricas Finales por Learning Rate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp2_learning_rate_metrics.png')

    print("\n[OK] Experimento 2 completado. Graficos guardados.")


if __name__ == '__main__':
    main()
