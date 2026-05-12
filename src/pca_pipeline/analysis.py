import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_and_scale_data(data_path):
	df = pd.read_csv(data_path)
	df.set_index('Country', inplace=True)

	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(df)
	df_scaled = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
	return df, df_scaled


def compute_correlation(df):
	return df.corr()


def fit_pca_model(df_scaled):
	pca_model = PCA()
	pca_model.fit(df_scaled)
	return pca_model


def compute_loadings(pca_model, feature_names):
	loadings_pc1 = pd.Series(pca_model.components_[0], index=feature_names).sort_values(ascending=False)
	loadings_pc2 = pd.Series(pca_model.components_[1], index=feature_names).sort_values(ascending=False)
	return loadings_pc1, loadings_pc2


def compute_sorted_pc1_dataframe(df, pca_model, df_scaled):
	df_with_pc1 = df.copy()
	df_with_pc1['PC1'] = pca_model.transform(df_scaled)[:, 0]
	df_sorted = df_with_pc1.sort_values(by='PC1', ascending=False)
	return df_with_pc1, df_sorted


def compute_biplot_components(df, pca_model, df_scaled):
	scores = pca_model.transform(df_scaled)[:, :2]
	loadings = pca_model.components_[:2].T

	score_x = scores[:, 0]
	score_y = scores[:, 1]
	loading_x = loadings[:, 0]
	loading_y = loadings[:, 1]

	scale_x = (score_x.max() - score_x.min()) * 0.35
	scale_y = (score_y.max() - score_y.min()) * 0.35

	country_positions = list(zip(df.index, score_x, score_y))
	country_refs = {position + 1: country for position, (country, _, _) in enumerate(country_positions)}

	return {
		'score_x': score_x,
		'score_y': score_y,
		'loading_x': loading_x,
		'loading_y': loading_y,
		'scale_x': scale_x,
		'scale_y': scale_y,
		'country_positions': country_positions,
		'country_refs': country_refs,
	}
