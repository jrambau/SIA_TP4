import numpy as np
import matplotlib.pyplot as plt
import os

def get_hopfield_output_path(filename):
    """Devuelve una ruta absoluta dentro de outputs/hopfield."""
    outputs_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'hopfield'))
    os.makedirs(outputs_dir, exist_ok=True)

    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return filename

    return os.path.join(outputs_dir, filename)

def create_pattern(ascii_art):
    """Convierte un dibujo de texto (* y .) en una matriz de Hopfield (1 y -1)."""
    lineas = ascii_art.strip().split('\n')
    matriz = []
    for linea in lineas:
        fila = [1 if char == '*' else -1 for char in linea.strip()]
        matriz.append(fila)
    return np.array(matriz)

# def ortogonality_test(nombres, patrones):
#     """Calcula y muestra la matriz de ortogonalidad (Producto Punto)"""
#     print(f"\n--- MATRIZ DE ORTOGONALIDAD: {', '.join(nombres)} ---")
#     num_patterns = len(patrones)
    
#     header = "      " + "  ".join(f"{n:^4}" for n in nombres)
#     print(header)
#     print("-" * len(header))
    
#     for i in range(num_patterns):
#         row_str = f"{nombres[i]:^4}|"
#         for j in range(num_patterns):
#             dot_product = np.dot(patrones[i].flatten(), patrones[j].flatten())
#             row_str += f"{dot_product:^6}"
#         print(row_str)
#     print("-" * len(header))

def add_noise(pattern, noise_level=0.2):
    """Invierte un porcentaje (noise_level) de los píxeles de un patrón."""
    noisy = np.copy(pattern).flatten()
    num_flips = int(len(noisy) * noise_level)
    indices = np.random.choice(len(noisy), num_flips, replace=False)
    for idx in indices:
        noisy[idx] *= -1
    return noisy.reshape(pattern.shape)

def plot_recovery(original, ruidoso, recuperado, titulo, filename):
    """Genera un gráfico comparativo: Original -> Ruidoso -> Recuperado"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    
    imagenes = [original, ruidoso, recuperado]
    titulos = ['Patrón Original', 'Patrón Ruidoso (Entrada)', 'Patrón Recuperado (Salida)']
    
    for ax, img, t in zip(axes, imagenes, titulos):
        ax.imshow(img, cmap='binary', vmin=-1, vmax=1)
        ax.set_title(t)
        ax.axis('off')
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        
    plt.suptitle(titulo, fontsize=14)
    plt.tight_layout()

    # Ensure outputs/hopfield directory exists relative to repository root
    outputs_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'hopfield'))
    os.makedirs(outputs_dir, exist_ok=True)

    # If filename contains a path, use it; otherwise save under outputs/hopfield
    if os.path.dirname(filename):
        out_path = filename
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        out_path = os.path.join(outputs_dir, filename)

    plt.savefig(out_path, dpi=150)
    print(f"Gráfico guardado: {out_path}")
    plt.close()
    
def get_abc():
    alfabeto_ascii = {
        'A': " .***.\n *...*\n *****\n *...*\n *...*",
        'B': " ****.\n *...*\n ****.\n *...*\n ****.",
        'C': " .****\n *....\n *....\n *....\n .****",
        'D': " ****.\n *...*\n *...*\n *...*\n ****.",
        'E': " *****\n *....\n ****.\n *....\n *****",
        'F': " *****\n *....\n ****.\n *....\n *....",
        'G': " .****\n *....\n *.***\n *...*\n .***.",
        'H': " *...*\n *...*\n *****\n *...*\n *...*",
        'I': " *****\n ..*..\n ..*..\n ..*..\n *****",
        'J': " ..***\n ...*.\n ...*.\n *..*.\n .**..",
        'K': " *...*\n *..*.\n ***..\n *..*.\n *...*",
        'L': " *....\n *....\n *....\n *....\n *****",
        'M': " *...*\n **.**\n *.*.*\n *...*\n *...*",
        'N': " *...*\n **..*\n *.*.*\n *..**\n *...*",
        'O': " .***.\n *...*\n *...*\n *...*\n .***.",
        'P': " ****.\n *...*\n ****.\n *....\n *....",
        'Q': " .***.\n *...*\n *.*.*\n *..**\n .***.",
        'R': " ****.\n *...*\n ****.\n *..*.\n *...*",
        'S': " .****\n *....\n .***.\n ....*\n ****.",
        'T': " *****\n ..*..\n ..*..\n ..*..\n ..*..",
        'U': " *...*\n *...*\n *...*\n *...*\n .***.",
        'V': " *...*\n *...*\n *...*\n .*.*.\n ..*..",
        'W': " *...*\n *...*\n *.*.*\n **.**\n *...*",
        'X': " *...*\n .*.*.\n ..*..\n .*.*.\n *...*",
        'Y': " *...*\n .*.*.\n ..*..\n ..*..\n ..*..",
        'Z': " *****\n ...*.\n ..*..\n .*...\n *****"
    }
    
    # Convertimos el diccionario de texto a un diccionario de matrices NumPy
    abecedario_matrices = {}
    for letra, ascii_art in alfabeto_ascii.items():
        abecedario_matrices[letra] = create_pattern(ascii_art)
        
    return abecedario_matrices

def  plot_abc(abecedario_dict, filename="plancha_abecedario.png"):
    """Genera un PNG con las 26 letras en una cuadrícula."""
    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    axes = axes.flatten()
    
    for ax, (letra, matriz) in zip(axes, abecedario_dict.items()):
        ax.imshow(matriz, cmap='binary', vmin=-1, vmax=1)
        ax.set_title(f"Letra {letra}")
        ax.axis('off')
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
    # Apagamos los subplots vacíos al final
    for i in range(26, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    out_path = get_hopfield_output_path(filename)
    plt.savefig(out_path, dpi=150)
    print(f"¡Plancha del abecedario generada: {out_path}!")
    plt.close()