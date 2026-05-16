import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from oja_network import OjaNetwork


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent / 'data' / 'europe.csv'
OUTPUT_DIR = BASE_DIR.parent.parent / 'outputs' / 'oja'
OUTPUT_DIR.mkdir(exist_ok=True)

def main_oja():
    # 1. Cargar y preparar los datos
    df = pd.read_csv(DATA_PATH)
    features = df.drop(columns=['Country'])
    feature_names = features.columns
    
    # IMPORTANTE: Para Oja y PCA, los datos deben estar (estandarizados)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # ---------------------------------------------------------
    # Consigna 1: Calcular la 1ra componente principal con Oja
    # ---------------------------------------------------------
    print("Entrenando red de Oja...")
    oja_net = OjaNetwork(input_dim=X_scaled.shape[1], learning_rate=0.001, epochs=2000)
    w_oja = oja_net.train(X_scaled)
    print("Pesos finales (Regla de Oja):", np.round(w_oja, 4))
    
    # ---------------------------------------------------------
    # Consigna 3: Calcular con librería (scikit-learn) TODO: Podría usarse lo preentregado pero es lo mismo. 
    # ---------------------------------------------------------
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    w_pca = pca.components_[0]
    diference = np.linalg.norm(w_oja - w_pca)
    print(f"Diferencia (Error con distancia euclidiana) entre pesos Oja y PCA: {diference:.6f}")
    print("Pesos finales (PCA sklearn): ", np.round(w_pca, 4))
    
    # Los autovectores pueden apuntar en direcciones opuestas 
    # (multiplicados por -1) y significar exactamente lo mismo. 
    # Si los signos están invertidos entre Oja y PCA, los alineamos para el gráfico:
    if np.sign(w_oja[0]) != np.sign(w_pca[0]):
        w_pca = w_pca * -1
        
    # ---------------------------------------------------------
    # Visualización Comparativa
    # ---------------------------------------------------------
    x = np.arange(len(feature_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, w_oja, width, label='Regla de Oja', color='skyblue')
    ax.bar(x + width/2, w_pca, width, label='Sklearn PCA', color='salmon')
    
    ax.set_ylabel('Valor del Peso (Carga de la Componente)')
    ax.set_title('Comparación PC1: Regla de Oja vs PCA (Librería)')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'oja_vs_pca.png')
if __name__ == '__main__':
    main_oja()