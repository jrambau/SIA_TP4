from pathlib import Path

from pca_pipeline import (
	load_and_scale_data,
	compute_correlation,
	fit_pca_model,
	compute_loadings,
	compute_sorted_pc1_dataframe,
	compute_biplot_components,
	set_plot_theme,
	save_output,
	plot_correlation_heatmap,
	plot_pc1_loadings,
	plot_pc2_loadings,
	plot_pc1_ranking,
	plot_biplot,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'data' / 'europe.csv'
OUTPUT_DIR = BASE_DIR.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

set_plot_theme()

# 1. Cargar el dataset
# Es buena práctica setear el país como índice para no perder la etiqueta al transformar la matriz
df, df_scaled = load_and_scale_data(DATA_PATH)

# Matriz de correlaciones (originales)
corr = compute_correlation(df)
print("\n--- Matriz de correlaciones ---")
print(corr.round(3))
# Guardar la matriz en CSV
corr.to_csv(OUTPUT_DIR / 'correlation_matrix.csv')

# Heatmap de la matriz de correlaciones
fig = plot_correlation_heatmap(corr)
save_output(fig, OUTPUT_DIR, 'correlation_matrix_heatmap.png')

# 3. Instanciar y ajustar el modelo PCA
# Como solo nos piden interpretar la PC1, podríamos pasarle n_components=1, 
# pero lo dejamos por defecto para ver cuánta varianza explica.
pca = fit_pca_model(df_scaled)

# Proporción de varianza explicada por las primeras componentes
var_explicada_pc1 = pca.explained_variance_ratio_[0] * 100
var_explicada_pc2 = pca.explained_variance_ratio_[1] * 100
print(f"La PC1 explica el {var_explicada_pc1:.2f}% de la varianza total de los datos.")
print(f"La PC2 explica el {var_explicada_pc2:.2f}% de la varianza total de los datos.")

print("\n--- Autovalores de las primeras componentes ---")
for i, val in enumerate(pca.explained_variance_[:5], start=1):
	print(f"Autovalor PC{i}: {val:.6f}")

# 4. Extraer los loadings de las primeras componentes
loadings_pc1, loadings_pc2 = compute_loadings(pca, df.columns)

print("\n--- Cargas (Loadings) de la PC1 ---")
print(loadings_pc1)

print("\n--- Cargas (Loadings) de la PC2 ---")
print(loadings_pc2)

# Graficamos los loadings para entender visualmente la influencia de cada variable
fig = plot_pc1_loadings(loadings_pc1)
save_output(fig, OUTPUT_DIR, 'pc1_loadings.png')

fig = plot_pc2_loadings(loadings_pc2)
save_output(fig, OUTPUT_DIR, 'pc2_loadings.png')

# 5. Calcular los valores de la PC1 para cada país
_, df_sorted = compute_sorted_pc1_dataframe(df, pca, df_scaled)

# 6. Gráfico de barras de la PC1 por país
fig = plot_pc1_ranking(df_sorted)
save_output(fig, OUTPUT_DIR, 'pc1_ranking_paises.png')

# 7. Biplot usando las dos primeras componentes principales
biplot_data = compute_biplot_components(df, pca, df_scaled)
fig = plot_biplot(df, pca.explained_variance_ratio_, biplot_data)
save_output(fig, OUTPUT_DIR, 'pc1_pc2_biplot.png')