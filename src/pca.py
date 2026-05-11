from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'data' / 'europe.csv'
OUTPUT_DIR = BASE_DIR.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid')

def save_and_show(fig, filename):
	output_path = OUTPUT_DIR / filename
	fig.savefig(output_path, dpi=150, bbox_inches='tight')
	plt.show()
	plt.close(fig)
	print(f'Gráfico guardado en {output_path}')

# 1. Cargar el dataset
# Es buena práctica setear el país como índice para no perder la etiqueta al transformar la matriz
df = pd.read_csv(DATA_PATH)
df.set_index('Country', inplace=True)

# 2. Estandarizar las variables (Paso fundamental)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Volvemos a armar un DataFrame para mantener la prolijidad con los nombres de las columnas
df_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)

# Matriz de correlaciones (originales)
corr = df.corr()
print("\n--- Matriz de correlaciones ---")
print(corr.round(3))
# Guardar la matriz en CSV
corr.to_csv(OUTPUT_DIR / 'correlation_matrix.csv')

# Heatmap de la matriz de correlaciones
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={'shrink': .8}, ax=ax)
ax.set_title('Matriz de correlaciones de variables')
fig.tight_layout()
save_and_show(fig, 'correlation_matrix_heatmap.png')

# 3. Instanciar y ajustar el modelo PCA
# Como solo nos piden interpretar la PC1, podríamos pasarle n_components=1, 
# pero lo dejamos por defecto para ver cuánta varianza explica.
pca = PCA()
pca.fit(df_scaled)

# Extraemos la proporción de la varianza que explica la primera componente
var_explicada_pc1 = pca.explained_variance_ratio_[0] * 100
print(f"La PC1 explica el {var_explicada_pc1:.2f}% de la varianza total de los datos.")

# 4. Extraer los loadings de la PC1 (los coeficientes del primer autovector)
loadings_pc1 = pd.Series(pca.components_[0], index=df.columns).sort_values(ascending=False)

print("\n--- Cargas (Loadings) de la PC1 ---")
print(loadings_pc1)

# Graficamos los loadings para entender visualmente la influencia de cada variable
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=loadings_pc1.values, y=loadings_pc1.index, palette="viridis", ax=ax)
ax.set_title('Pesos de las variables originales en la PC1')
ax.set_xlabel('Carga (Loading)')
ax.set_ylabel('Variable')
ax.grid(axis='x', linestyle='--', alpha=0.7)
fig.tight_layout()
save_and_show(fig, 'pc1_loadings.png')

# 5. Calcular los valores de la PC1 para cada país
df['PC1'] = pca.transform(df_scaled)[:, 0]

# Ordenamos el dataframe por este nuevo índice
df_sorted = df.sort_values(by='PC1', ascending=False)

# 6. Gráfico de barras de la PC1 por país
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='PC1', y=df_sorted.index, data=df_sorted, palette="coolwarm", ax=ax)
ax.set_title('Ranking de Países Europeos según la PC1')
ax.set_xlabel('Valor de la Componente Principal 1')
ax.set_ylabel('País')
ax.axvline(0, color='black', linewidth=1)
ax.grid(axis='x', linestyle='--', alpha=0.7)
fig.tight_layout()
save_and_show(fig, 'pc1_ranking_paises.png')

# 7. Biplot usando las dos primeras componentes principales
scores = pca.transform(df_scaled)[:, :2]
loadings = pca.components_[:2].T

score_x = scores[:, 0]
score_y = scores[:, 1]
loading_x = loadings[:, 0]
loading_y = loadings[:, 1]

scale_x = (score_x.max() - score_x.min()) * 0.35
scale_y = (score_y.max() - score_y.min()) * 0.35

fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(score_x, score_y, color='steelblue', alpha=0.7, s=45)

for country, x_coord, y_coord in zip(df.index, score_x, score_y):
	ax.annotate(country, (x_coord, y_coord), xytext=(4, 4), textcoords='offset points', fontsize=8, alpha=0.8)

for variable, x_loading, y_loading in zip(df.columns, loading_x, loading_y):
	ax.arrow(0, 0, x_loading * scale_x, y_loading * scale_y, color='darkred', alpha=0.8, head_width=0.08, length_includes_head=True)
	ax.text(x_loading * scale_x * 1.08, y_loading * scale_y * 1.08, variable, color='darkred', fontsize=9, ha='center', va='center')

ax.axhline(0, color='gray', linewidth=1)
ax.axvline(0, color='gray', linewidth=1)
ax.set_title('Biplot de PCA: Países y Variables en PC1 vs PC2')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
save_and_show(fig, 'pc1_pc2_biplot.png')