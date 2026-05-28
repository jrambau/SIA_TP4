"""Experimento 6: Forma de Decaimiento del Learning Rate
========================================================
Compara tres estrategias de decaimiento del learning rate:
- Lineal:      lr(t) = lr_0 * (1 - t/T)
- Exponencial: lr(t) = lr_0 * exp(-t/τ)
- Potencia inversa: lr(t) = lr_0 / (1 + t/τ)
Analiza:
- Forma teórica de cada curva de decaimiento
- Convergencia del error de cuantización
- Calidad final de los clusters
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
    lr_0 = 0.1
    decay_types = ['linear', 'exponential', 'inverse']
    decay_labels = {
        'linear': r'Lineal: $lr_0 \cdot (1 - t/T)$',
        'exponential': r'Exponencial: $lr_0 \cdot e^{-t/\tau}$',
        'inverse': r'Inversa: $lr_0 / (1 + t/\tau)$',
    }
    results = {}

    # ── Gráfico 1: Curvas teóricas de decaimiento ────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {'linear': '#2196F3', 'exponential': '#E91E63', 'inverse': '#FF9800'}
    t = np.arange(epochs)
    tau = epochs / 5

    lr_curves = {
        'linear': lr_0 * (1 - t / epochs),
        'exponential': lr_0 * np.exp(-t / tau),
        'inverse': lr_0 / (1 + t / tau),
    }

    for decay in decay_types:
        ax.plot(t, lr_curves[decay], label=decay_labels[decay],
                color=colors[decay], linewidth=2)

    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Curvas Teóricas de Decaimiento del Learning Rate',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    save_figure(fig, 'exp6_lr_decay_curves.png')

    # ── Entrenamiento ────────────────────────────────────────────────────────
    for decay in decay_types:
        print(f"\nEntrenando SOM con decaimiento: {decay}...")
        np.random.seed(42)
        som = KohonenSOM(grid_y=grid_size, grid_x=grid_size,
                         input_dim=X_scaled.shape[1],
                         epochs=epochs, learning_rate=lr_0,
                         lr_decay=decay, track_qe=True)
        som.train(X_scaled)

        qe = som.compute_quantization_error(X_scaled)
        te = som.compute_topographic_error(X_scaled)

        results[decay] = {
            'som': som,
            'qe': qe,
            'te': te,
            'qe_history': som.qe_history,
        }
        print(f"  QE: {qe:.4f} | TE: {te:.4f}")

    # ── Gráfico 2: Curvas de convergencia QE ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for decay in decay_types:
        r = results[decay]
        ax.plot(r['qe_history'], label=decay_labels[decay],
                color=colors[decay], linewidth=1.5, alpha=0.85)

    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Error de Cuantización (QE)', fontsize=12)
    ax.set_title('Convergencia del QE según Estrategia de Decaimiento',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    save_figure(fig, 'exp6_lr_decay_convergence.png')

    # ── Gráfico 3: Mapas de países ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    for ax, decay in zip(axes, decay_types):
        r = results[decay]
        draw_country_map(ax, r['som'], X_scaled, countries, grid_size, grid_size, fontsize=8)
        ax.set_title(
            f'{decay_labels[decay]}\nQE={r["qe"]:.3f}  |  TE={r["te"]:.3f}',
            fontsize=11,
        )

    fig.suptitle('Experimento 6: Agrupación según Decaimiento del LR',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp6_lr_decay_maps.png')

    # ── Gráfico 4: U-Matrices ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, decay in zip(axes, decay_types):
        u_matrix = results[decay]['som'].get_u_matrix()
        sns.heatmap(u_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Distancia'})
        ax.set_title(f'U-Matrix  |  {decay.capitalize()}')

    fig.suptitle('Matrices U por Estrategia de Decaimiento', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp6_lr_decay_umatrix.png')

    # ── Gráfico 5: Comparación de métricas ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    labels = [decay.capitalize() for decay in decay_types]
    qe_vals = [results[d]['qe'] for d in decay_types]
    te_vals = [results[d]['te'] for d in decay_types]
    bar_colors = [colors[d] for d in decay_types]

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

    fig.suptitle('Métricas Finales por Estrategia de Decaimiento', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp6_lr_decay_metrics.png')

    print("\n[OK] Experimento 6 completado. Graficos guardados.")


if __name__ == '__main__':
    main()
