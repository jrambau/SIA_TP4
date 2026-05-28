"""Experimento 8: Métricas de Calidad – QE y Error Topográfico
==============================================================
Evaluación exhaustiva de la calidad del SOM combinando múltiples
configuraciones (tamaño de grilla × learning rate) y corriendo
múltiples semillas para obtener media y desviación estándar.
Genera:
- Heatmaps de QE y TE promedio por configuración
- Análisis de estabilidad (varianza entre semillas)
- Tabla resumen impresa en consola
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from exp_utils import load_data, save_figure
from kohonen import KohonenSOM


def main():
    X_scaled, countries, feature_names = load_data()

    # Configuration space
    grid_sizes = [3, 4, 5]
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    seeds = [42, 123, 456, 789, 1024]
    epochs = 3000  # Reduced for the grid search (many runs)

    # Store results
    records = []

    total_runs = len(grid_sizes) * len(learning_rates) * len(seeds)
    run = 0

    for size in grid_sizes:
        for lr in learning_rates:
            qe_list = []
            te_list = []
            for seed in seeds:
                run += 1
                print(f"\r  [{run}/{total_runs}] Grid {size}x{size}, lr={lr}, seed={seed}", end='')
                np.random.seed(seed)
                som = KohonenSOM(grid_y=size, grid_x=size,
                                 input_dim=X_scaled.shape[1],
                                 epochs=epochs, learning_rate=lr)
                som.train(X_scaled)

                qe = som.compute_quantization_error(X_scaled)
                te = som.compute_topographic_error(X_scaled)
                qe_list.append(qe)
                te_list.append(te)

            records.append({
                'grid': f'{size}x{size}',
                'grid_size': size,
                'lr': lr,
                'qe_mean': np.mean(qe_list),
                'qe_std': np.std(qe_list),
                'te_mean': np.mean(te_list),
                'te_std': np.std(te_list),
            })

    print("\n")
    df = pd.DataFrame(records)

    # ── Gráfico 1: Heatmap de QE medio ──────────────────────────────────────
    pivot_qe = df.pivot(index='grid', columns='lr', values='qe_mean')
    pivot_qe_std = df.pivot(index='grid', columns='lr', values='qe_std')

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.heatmap(pivot_qe, annot=True, fmt=".3f", cmap='YlOrRd', ax=axes[0],
                cbar_kws={'label': 'QE Medio'})
    axes[0].set_title('Error de Cuantización Medio (QE)', fontsize=13)
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Tamaño de Grilla')

    sns.heatmap(pivot_qe_std, annot=True, fmt=".4f", cmap='YlOrRd', ax=axes[1],
                cbar_kws={'label': 'QE Std'})
    axes[1].set_title('Desviación Estándar del QE', fontsize=13)
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('Tamaño de Grilla')

    fig.suptitle('Experimento 8: Error de Cuantización por Configuración',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp8_quality_qe_heatmap.png')

    # ── Gráfico 2: Heatmap de TE medio ──────────────────────────────────────
    pivot_te = df.pivot(index='grid', columns='lr', values='te_mean')
    pivot_te_std = df.pivot(index='grid', columns='lr', values='te_std')

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.heatmap(pivot_te, annot=True, fmt=".3f", cmap='YlGnBu', ax=axes[0],
                cbar_kws={'label': 'TE Medio'})
    axes[0].set_title('Error Topográfico Medio (TE)', fontsize=13)
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Tamaño de Grilla')

    sns.heatmap(pivot_te_std, annot=True, fmt=".4f", cmap='YlGnBu', ax=axes[1],
                cbar_kws={'label': 'TE Std'})
    axes[1].set_title('Desviación Estándar del TE', fontsize=13)
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('Tamaño de Grilla')

    fig.suptitle('Experimento 8: Error Topográfico por Configuración',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp8_quality_te_heatmap.png')

    # ── Gráfico 3: Boxplot de QE por grid size ───────────────────────────────
    # Re-run to get individual values for boxplots
    box_data = []
    for size in grid_sizes:
        for lr in learning_rates:
            for seed in seeds:
                np.random.seed(seed)
                som = KohonenSOM(grid_y=size, grid_x=size,
                                 input_dim=X_scaled.shape[1],
                                 epochs=epochs, learning_rate=lr)
                som.train(X_scaled)
                qe = som.compute_quantization_error(X_scaled)
                te = som.compute_topographic_error(X_scaled)
                box_data.append({
                    'Grid': f'{size}x{size}',
                    'lr': lr,
                    'QE': qe,
                    'TE': te,
                })

    df_box = pd.DataFrame(box_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df_box, x='Grid', y='QE', hue='lr', palette='Set2', ax=axes[0])
    axes[0].set_title('Distribución del QE', fontsize=13)
    axes[0].legend(title='LR', fontsize=9)

    sns.boxplot(data=df_box, x='Grid', y='TE', hue='lr', palette='Set2', ax=axes[1])
    axes[1].set_title('Distribución del TE', fontsize=13)
    axes[1].legend(title='LR', fontsize=9)

    fig.suptitle('Estabilidad de Métricas (múltiples semillas)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp8_quality_boxplots.png')

    # ── Tabla resumen en consola ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLA RESUMEN: Métricas de Calidad por Configuración")
    print("=" * 80)
    print(f"{'Grid':>6}  {'LR':>6}  {'QE(μ)':>8}  {'QE(σ)':>8}  {'TE(μ)':>8}  {'TE(σ)':>8}")
    print("-" * 54)
    for _, row in df.iterrows():
        print(f"{row['grid']:>6}  {row['lr']:>6.2f}  "
              f"{row['qe_mean']:>8.4f}  {row['qe_std']:>8.4f}  "
              f"{row['te_mean']:>8.4f}  {row['te_std']:>8.4f}")

    # Find best configuration
    best_qe = df.loc[df['qe_mean'].idxmin()]
    best_te = df.loc[df['te_mean'].idxmin()]
    print(f"\nMejor QE: Grid {best_qe['grid']}, LR={best_qe['lr']} -> QE={best_qe['qe_mean']:.4f}")
    print(f"Mejor TE: Grid {best_te['grid']}, LR={best_te['lr']} -> TE={best_te['te_mean']:.4f}")

    print("\n[OK] Experimento 8 completado. Graficos guardados.")


if __name__ == '__main__':
    main()
