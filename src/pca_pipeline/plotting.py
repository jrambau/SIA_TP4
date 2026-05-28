import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text


def set_plot_theme():
	sns.set_theme(style='whitegrid')


def save_output(fig, output_dir, filename):
	output_path = output_dir / filename
	fig.savefig(output_path, dpi=150, bbox_inches='tight')
	print(f'Gráfico guardado en {output_path}')


def plot_correlation_heatmap(corr):
	fig, ax = plt.subplots(figsize=(8, 6))
	sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'shrink': 0.8}, ax=ax)
	ax.set_title('Matriz de correlaciones de variables')
	fig.tight_layout()
	return fig


def plot_pc1_loadings(loadings_pc1):
	fig, ax = plt.subplots(figsize=(10, 5))
	sns.barplot(x=loadings_pc1.values, y=loadings_pc1.index, hue=loadings_pc1.index, palette='viridis', ax=ax, legend=False)
	ax.set_title('Pesos de las variables originales en la PC1')
	ax.set_xlabel('Carga (Loading)')
	ax.set_ylabel('Variable')
	ax.grid(axis='x', linestyle='--', alpha=0.7)
	fig.tight_layout()
	return fig


def plot_pc2_loadings(loadings_pc2):
	fig, ax = plt.subplots(figsize=(10, 5))
	sns.barplot(x=loadings_pc2.values, y=loadings_pc2.index, hue=loadings_pc2.index, palette='mako', ax=ax, legend=False)
	ax.set_title('Pesos de las variables originales en la PC2')
	ax.set_xlabel('Carga (Loading)')
	ax.set_ylabel('Variable')
	ax.grid(axis='x', linestyle='--', alpha=0.7)
	fig.tight_layout()
	return fig


def plot_pc1_ranking(df_sorted):
	fig, ax = plt.subplots(figsize=(12, 8))
	sns.barplot(x='PC1', y=df_sorted.index, data=df_sorted, hue=df_sorted.index, palette='coolwarm', ax=ax, legend=False)
	ax.set_title('Ranking de Países Europeos según la PC1')
	ax.set_xlabel('Valor de la Componente Principal 1')
	ax.set_ylabel('País')
	ax.axvline(0, color='black', linewidth=1)
	ax.grid(axis='x', linestyle='--', alpha=0.7)
	fig.tight_layout()
	return fig


def plot_biplot(df, explained_variance_ratio, biplot_data, title='Biplot de PCA: Países y Variables en PC1 vs PC2'):
	score_x = biplot_data['score_x']
	score_y = biplot_data['score_y']
	loading_x = biplot_data['loading_x']
	loading_y = biplot_data['loading_y']
	scale_x = biplot_data['scale_x']
	scale_y = biplot_data['scale_y']
	country_positions = biplot_data['country_positions']

	fig, ax = plt.subplots(figsize=(12, 9))
	ax.scatter(score_x, score_y, color='steelblue', alpha=0.7, s=45)

	country_labels = []
	for country, x_coord, y_coord in country_positions:
		label = ax.text(
			x_coord,
			y_coord,
			country,
			fontsize=7,
			color='black',
			ha='left',
			va='bottom',
		)
		country_labels.append(label)

	adjust_text(
		country_labels,
		ax=ax,
		force_text=(0.4, 0.6),
		force_points=(0.2, 0.3),
		expand_text=(1.05, 1.1),
		expand_points=(1.05, 1.1),
		only_move={'text': 'xy', 'points': 'xy', 'objects': 'xy'},
		lim=250,
		ensure_inside_axes=True,
	)

	for variable, x_loading, y_loading in zip(df.columns, loading_x, loading_y):
		ax.arrow(0, 0, x_loading * scale_x, y_loading * scale_y, color='darkred', alpha=0.75, head_width=0.06, length_includes_head=True)
		ax.text(x_loading * scale_x * 1.08, y_loading * scale_y * 1.08, variable, color='darkred', fontsize=9, ha='center', va='center')

	ax.axhline(0, color='gray', linewidth=1)
	ax.axvline(0, color='gray', linewidth=1)
	ax.set_title('Biplot de PCA: Países y Variables en PC1 vs PC2')
	ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.1f}%)')
	ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.1f}%)')
	ax.grid(True, linestyle='--', alpha=0.5)
	fig.tight_layout()
	return fig
