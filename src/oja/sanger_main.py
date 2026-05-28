import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from pca_pipeline.plotting import (
    set_plot_theme,
    save_output,
    plot_correlation_heatmap,
    plot_pc1_loadings,
    plot_pc2_loadings,
    plot_pc1_ranking,
    plot_biplot,
)
from sanger_network import SangerNetwork

DATA_PATH = BASE_DIR.parent.parent / 'data' / 'europe.csv'
OUTPUT_DIR = BASE_DIR.parent.parent / 'outputs' / 'oja'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_COMPONENTS = 2


def align_signs(sanger_components, pca_components):
    aligned = sanger_components.copy()
    for i in range(len(aligned)):
        if np.sign(aligned[i, 0]) != np.sign(pca_components[i, 0]):
            aligned[i] *= -1
    return aligned


def main_sanger():
    set_plot_theme()

    # 1. Load and standardize
    df = pd.read_csv(DATA_PATH)
    countries = df['Country'].values
    features = df.drop(columns=['Country'])
    feature_names = features.columns.tolist()
    features_df = features.set_index(df['Country'])  # Country as index, for biplot

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 2. Train Sanger
    print(f"Entrenando red de Sanger ({N_COMPONENTS} componentes)...")
    sanger_net = SangerNetwork(
        input_dim=X_scaled.shape[1],
        n_components=N_COMPONENTS,
        learning_rate=0.001,
        epochs=2000,
    )
    W_sanger = sanger_net.train(X_scaled)

    # 3. Reference: sklearn PCA (for sign alignment and comparison)
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(X_scaled)
    W_pca = pca.components_

    W_sanger_aligned = align_signs(W_sanger, W_pca)

    for i in range(N_COMPONENTS):
        err = np.linalg.norm(W_sanger_aligned[i] - W_pca[i])
        print(f"\n  PC{i+1} — Error euclidiano Sanger vs PCA: {err:.6f}")
        print(f"  {'Variable':<14} {'Implementación':>16} {'Librería':>12}")
        print(f"  {'-'*44}")
        for fname, vs, vp in zip(feature_names, W_sanger_aligned[i], W_pca[i]):
            print(f"  {fname:<14} {vs:>16.4f} {vp:>12.4f}")

    # Scores and eigenvalues
    scores_sanger = X_scaled @ W_sanger_aligned.T
    for i in range(N_COMPONENTS):
        if np.sign(W_sanger_aligned[i, 0]) != np.sign(W_pca[i, 0]):
            scores_sanger[:, i] *= -1

    eigenvalues = np.var(scores_sanger, axis=0, ddof=0)
    total_var = float(X_scaled.shape[1])
    ev_ratio = eigenvalues / total_var

    print(f"\nVarianza explicada — PC1: {ev_ratio[0]*100:.2f}%  PC2: {ev_ratio[1]*100:.2f}%")

    # ----------------------------------------------------------------
    # Plot 1: Loadings comparison Sanger vs PCA
    # ----------------------------------------------------------------
    x = np.arange(len(feature_names))
    width = 0.35
    fig, axes = plt.subplots(1, N_COMPONENTS, figsize=(7 * N_COMPONENTS, 5))
    for i, ax in enumerate(axes):
        ax.bar(x - width / 2, W_sanger_aligned[i], width, label='Implementación', color='mediumseagreen')
        ax.bar(x + width / 2, W_pca[i], width, label='Librería', color='salmon')
        ax.set_title(f'PC{i+1}: Implementación vs Librería')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('Carga (loading)')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
    fig.suptitle('Comparación de Componentes: Implementación vs Librería', fontsize=13)
    plt.tight_layout()
    save_output(fig, OUTPUT_DIR, 'sanger_vs_pca_loadings.png')
    plt.close()

    # ----------------------------------------------------------------
    # Plots estilo preentrega (usando pca_pipeline/plotting.py)
    # ----------------------------------------------------------------

    # Correlation matrix
    corr = pd.DataFrame(X_scaled, columns=feature_names).corr()
    fig = plot_correlation_heatmap(corr)
    save_output(fig, OUTPUT_DIR, 'sanger_correlation_matrix.png')
    plt.close()

    # PC1 loadings
    loadings_pc1 = pd.Series(W_sanger_aligned[0], index=feature_names).sort_values(ascending=False)
    print(f"\nCargas PC1:\n{loadings_pc1.round(6)}")
    print(f"Autovalor PC1: {eigenvalues[0]:.6f}  — explica el {ev_ratio[0]*100:.2f}% de la varianza")
    fig = plot_pc1_loadings(loadings_pc1)
    save_output(fig, OUTPUT_DIR, 'sanger_pc1_loadings.png')
    plt.close()

    # PC2 loadings
    loadings_pc2 = pd.Series(W_sanger_aligned[1], index=feature_names).sort_values(ascending=False)
    print(f"\nCargas PC2:\n{loadings_pc2.round(6)}")
    print(f"Autovalor PC2: {eigenvalues[1]:.6f}  — explica el {ev_ratio[1]*100:.2f}% de la varianza")
    fig = plot_pc2_loadings(loadings_pc2)
    save_output(fig, OUTPUT_DIR, 'sanger_pc2_loadings.png')
    plt.close()

    # Country ranking by PC1
    df_sorted = pd.DataFrame({'PC1': scores_sanger[:, 0]}, index=countries).sort_values('PC1', ascending=False)
    fig = plot_pc1_ranking(df_sorted)
    save_output(fig, OUTPUT_DIR, 'sanger_pc1_ranking.png')
    plt.close()

    # Biplot estilo preentrega
    score_x = scores_sanger[:, 0]
    score_y = scores_sanger[:, 1]
    loading_x = W_sanger_aligned.T[:, 0]
    loading_y = W_sanger_aligned.T[:, 1]
    scale_x = (score_x.max() - score_x.min()) * 0.35
    scale_y = (score_y.max() - score_y.min()) * 0.35
    country_positions = list(zip(countries, score_x, score_y))
    country_refs = {pos + 1: country for pos, (country, _, _) in enumerate(country_positions)}

    biplot_data = {
        'score_x': score_x,
        'score_y': score_y,
        'loading_x': loading_x,
        'loading_y': loading_y,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'country_positions': country_positions,
        'country_refs': country_refs,
    }
    fig = plot_biplot(features_df, ev_ratio, biplot_data, title='Biplot de Sanger: Países y Variables en PC1 vs PC2')
    save_output(fig, OUTPUT_DIR, 'sanger_biplot.png')
    plt.close()


if __name__ == '__main__':
    main_sanger()
