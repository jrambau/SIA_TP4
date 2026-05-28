"""Experimento 4: Número de Épocas y Convergencia
==================================================
Entrena la red SOM con un alto número de épocas registrando el error de
cuantización (QE) en cada época para analizar la curva de convergencia.
Analiza:
- Curva de QE vs épocas
- Identificación de puntos de convergencia (90%, 95%, 99% del mínimo)
- Snapshots del mapa en diferentes momentos del entrenamiento
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
    total_epochs = 10000

    print(f"Entrenando SOM con {total_epochs} épocas (tracking QE)...")
    som = KohonenSOM(grid_y=grid_size, grid_x=grid_size,
                     input_dim=X_scaled.shape[1],
                     epochs=total_epochs, learning_rate=0.1, track_qe=True)
    som.train(X_scaled)
    print("Entrenamiento finalizado.")

    qe_history = np.array(som.qe_history)
    final_qe = qe_history[-1]

    # Find convergence milestones
    # Define convergence as reaching within X% of the final (minimum) QE
    qe_range = qe_history[0] - final_qe
    milestones = {}
    for pct in [0.90, 0.95, 0.99]:
        threshold = final_qe + (1 - pct) * qe_range
        # First epoch where QE drops below threshold
        idx = np.argmax(qe_history <= threshold)
        if qe_history[idx] <= threshold:
            milestones[pct] = idx

    # ── Gráfico 1: Curva de convergencia ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(qe_history, color='steelblue', linewidth=1.2, label='QE por época')

    # Mark milestones
    milestone_colors = {0.90: 'green', 0.95: 'orange', 0.99: 'red'}
    for pct, epoch_idx in milestones.items():
        ax.axvline(x=epoch_idx, color=milestone_colors[pct], linestyle='--', alpha=0.7)
        ax.annotate(
            f'{int(pct*100)}% convergencia\n(época {epoch_idx})',
            xy=(epoch_idx, qe_history[epoch_idx]),
            xytext=(epoch_idx + total_epochs * 0.03, qe_history[0] * 0.7),
            arrowprops=dict(arrowstyle='->', color=milestone_colors[pct]),
            fontsize=10, color=milestone_colors[pct],
        )

    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Error de Cuantización (QE)', fontsize=12)
    ax.set_title(f'Convergencia del Error de Cuantización ({total_epochs} épocas)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    save_figure(fig, 'exp4_epochs_convergence.png')

    # ── Gráfico 2: Convergencia en escala logarítmica ────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(qe_history, color='steelblue', linewidth=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('QE (escala log)', fontsize=12)
    ax.set_title('Convergencia del QE – Escala Logarítmica', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp4_epochs_convergence_log.png')

    # ── Gráfico 3: Snapshots a diferentes cantidades de épocas ───────────────
    snapshot_epochs = [100, 500, 1000, 2000, 5000, total_epochs]
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    for ax, ep in zip(axes, snapshot_epochs):
        np.random.seed(42)
        som_snap = KohonenSOM(grid_y=grid_size, grid_x=grid_size,
                              input_dim=X_scaled.shape[1],
                              epochs=ep, learning_rate=0.1)
        som_snap.train(X_scaled)
        qe_snap = som_snap.compute_quantization_error(X_scaled)
        draw_country_map(ax, som_snap, X_scaled, countries, grid_size, grid_size, fontsize=6)
        ax.set_title(f'Épocas: {ep}  |  QE={qe_snap:.3f}', fontsize=11)

    fig.suptitle('Snapshots del Mapa SOM en Diferentes Momentos del Entrenamiento',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp4_epochs_snapshots.png')

    # Print summary
    print("\n--- Resumen de Convergencia ---")
    print(f"QE inicial (época 0): {qe_history[0]:.4f}")
    print(f"QE final (época {total_epochs - 1}): {final_qe:.4f}")
    for pct, epoch_idx in milestones.items():
        print(f"  {int(pct*100)}% convergencia alcanzada en época {epoch_idx}")

    print("\n[OK] Experimento 4 completado. Graficos guardados.")


if __name__ == '__main__':
    main()
