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


def plot_biplot(df, explained_variance_ratio, biplot_data):
	score_x = biplot_data['score_x']
	score_y = biplot_data['score_y']
	loading_x = biplot_data['loading_x']
	loading_y = biplot_data['loading_y']
	scale_x = biplot_data['scale_x']
	scale_y = biplot_data['scale_y']
	country_positions = biplot_data['country_positions']
	country_refs = biplot_data['country_refs']

	fig, ax = plt.subplots(figsize=(12, 9))
	ax.scatter(score_x, score_y, color='steelblue', alpha=0.7, s=45)
	fig.subplots_adjust(right=0.80)

	country_labels = []
	for ref_number, (_, x_coord, y_coord) in enumerate(country_positions, start=1):
		label = ax.text(
			x_coord,
			y_coord,
			str(ref_number),
			fontsize=7,
			fontweight='bold',
			color='black',
			ha='center',
			va='center',
			bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.15'),
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

	legend_x_left = 0.79
	legend_x_right = 0.90
	legend_y_start = 0.88
	legend_line_step = 0.022
	half = (len(country_refs) + 1) // 2
	left_items = list(country_refs.items())[:half]
	right_items = list(country_refs.items())[half:]

	fig.text(legend_x_left, legend_y_start + 0.03, 'Referencias', fontsize=9, fontweight='bold', ha='left', va='top')
	for row_index, (ref_number, country) in enumerate(left_items):
		fig.text(legend_x_left, legend_y_start - row_index * legend_line_step, f'{ref_number}. {country}', fontsize=6.5, ha='left', va='top')

	for row_index, (ref_number, country) in enumerate(right_items):
		fig.text(legend_x_right, legend_y_start - row_index * legend_line_step, f'{ref_number}. {country}', fontsize=6.5, ha='left', va='top')

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
