"""Experimento 1: Variación del Tamaño de Grilla (Versión Rigurosa - Estática)
=================================================================================
Compara los resultados de entrenar SOM con grillas de 3x3, 4x4, 5x5 y 6x6.
Analiza promedios sobre múltiples inicializaciones (seeds):
- Agrupación de países
- Error de cuantización (QE) y error topográfico (TE) con barras de error
- Porcentaje de utilización de neuronas
- Matrices U para cada tamaño
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from exp_utils import load_data, save_figure, draw_country_map
from kohonen import KohonenSOM

def main():
    # Configuramos el estilo para presentaciones (fuentes más grandes y claras)
    sns.set_theme(style="whitegrid", context="talk")
    
    X_scaled, countries, feature_names = load_data()

    grid_sizes = [3, 4, 5, 6]
    # Usamos 10 semillas aleatorias para sacar promedios y desvíos estándar (Rigor científico)
    seeds = [42, 7, 2021, 123, 999, 0, 555, 321, 88, 777]
    
    results = {}

    for size in grid_sizes:
        print(f"\nEntrenando SOM {size}x{size} sobre {len(seeds)} semillas...")
        
        qe_list, te_list, util_list = [], [], []
        best_som = None # Guardaremos el de menor QE para dibujar los mapas/matrices
        min_qe = float('inf')

        for seed in seeds:
            np.random.seed(seed)
            som = KohonenSOM(grid_y=size, grid_x=size, input_dim=X_scaled.shape[1],
                             epochs=5000, learning_rate=0.1, init_method='sample')
            som.train(X_scaled)

            qe = som.compute_quantization_error(X_scaled)
            te = som.compute_topographic_error(X_scaled)
            bmus = som.get_bmus(X_scaled)

            utilization = len(set(bmus)) / (size * size) * 100
            
            qe_list.append(qe)
            te_list.append(te)
            util_list.append(utilization)
            
            # Guardamos el mejor modelo de estas corridas para los mapas visuales
            if qe < min_qe:
                min_qe = qe
                best_som = som

        results[size] = {
            'som': best_som,
            'qe_mean': np.mean(qe_list),
            'qe_std': np.std(qe_list),
            'te_mean': np.mean(te_list),
            'te_std': np.std(te_list),
            'util_mean': np.mean(util_list),
            'util_std': np.std(util_list),
        }
        print(f"  Promedios -> QE: {results[size]['qe_mean']:.3f} | TE: {results[size]['te_mean']:.3f} | Util: {results[size]['util_mean']:.1f}%")

    # ── Gráfico 1: Comparación de métricas con Barras de Error ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sizes_labels = [f'{s}x{s}' for s in grid_sizes]
    
    # Extraemos medias y desvíos
    qe_m = [results[s]['qe_mean'] for s in grid_sizes]
    qe_s = [results[s]['qe_std'] for s in grid_sizes]
    te_m = [results[s]['te_mean'] for s in grid_sizes]
    te_s = [results[s]['te_std'] for s in grid_sizes]
    ut_m = [results[s]['util_mean'] for s in grid_sizes]
    ut_s = [results[s]['util_std'] for s in grid_sizes]
    
    colors = sns.color_palette('viridis', len(grid_sizes))

    # Graficamos con yerr (barras de error)
    axes[0].bar(sizes_labels, qe_m, yerr=qe_s, color=colors, capsize=5, alpha=0.9)
    axes[0].set_title('Error de Cuantización (QE)')
    axes[0].set_ylabel('QE Promedio')
    
    axes[1].bar(sizes_labels, te_m, yerr=te_s, color=colors, capsize=5, alpha=0.9)
    axes[1].set_title('Error Topográfico (TE)')
    axes[1].set_ylabel('TE Promedio')

    axes[2].bar(sizes_labels, ut_m, yerr=ut_s, color=colors, capsize=5, alpha=0.9)
    axes[2].set_title('Utilización de Neuronas')
    axes[2].set_ylabel('Utilización Promedio (%)')

    fig.suptitle('Estabilidad y Métricas de Calidad (Promedio de 10 corridas)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp1_metrics_bars.png')

    # ── Gráfico 2: U-Matrices ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, size in zip(axes, grid_sizes):
        u_matrix = results[size]['som'].get_u_matrix()
        # Usamos 'magma' que es excelente para topografías y contrastes
        sns.heatmap(u_matrix, cmap='magma', annot=True, fmt=".2f", ax=ax,
                    cbar_kws={'label': 'Distancia Promedio'},
                    annot_kws={"size": 10}) # Ajusta el tamaño del número adentro
        ax.set_title(f'U-Matrix {size}x{size}')

    fig.suptitle('Matrices U (Topología de Distancias)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp1_umatrix.png')

    # ── Gráfico 3: Densidad (cantidad de países por neurona) ─────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, size in zip(axes, grid_sizes):
        bmus = results[size]['som'].get_bmus(X_scaled)
        count_matrix = np.zeros((size, size))
        for (y, x) in bmus:
            count_matrix[y, x] += 1
        
        # 'Blues' es limpio, le ponemos linewidths para separar las celdas
        sns.heatmap(count_matrix, cmap='Blues', annot=True, fmt='g', ax=ax,
                    linewidths=.5, cbar_kws={'label': 'Cantidad de países'})
        ax.set_title(f'Densidad {size}x{size}')

    fig.suptitle('Densidad de Agrupación', fontsize=18, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp1_density.png')

    # Nota: Mantenemos tu llamada al mapa de países (asumiendo que en exp_utils.py lo resolviste bien)
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    for ax, size in zip(axes, grid_sizes):
        r = results[size]
        draw_country_map(ax, r['som'], X_scaled, countries, size, size)
        ax.set_title(f'Mapa de Países {size}x{size}\n(Mejor ejecución: QE={r["som"].compute_quantization_error(X_scaled):.3f})')
    plt.tight_layout()
    save_figure(fig, 'exp1_country_maps.png')

    print("\n[OK] Experimento 1 completado. Gráficos estáticos de alta calidad guardados.")

if __name__ == '__main__':
    main()