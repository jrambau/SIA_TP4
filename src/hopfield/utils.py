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

def orthogonality_test(names, patterns):
    """Calcula y muestra la matriz de ortogonalidad (producto punto)."""
    print(f"\n--- MATRIZ DE ORTOGONALIDAD: {', '.join(names)} ---")
    num_patterns = len(patterns)

    header = "      " + "  ".join(f"{name:^4}" for name in names)
    print(header)
    print("-" * len(header))

    for i in range(num_patterns):
        row_str = f"{names[i]:^4}|"
        for j in range(num_patterns):
            dot_product = np.dot(patterns[i].flatten(), patterns[j].flatten())
            row_str += f"{dot_product:^6}"
        print(row_str)
    print("-" * len(header))

def add_noise(pattern, noise_level=0.2):
    """Invierte un porcentaje (noise_level) de los píxeles de un patrón."""
    noisy = np.copy(pattern).flatten()
    num_flips = int(len(noisy) * noise_level)
    indices = np.random.choice(len(noisy), num_flips, replace=False)
    for idx in indices:
        noisy[idx] *= -1
    return noisy.reshape(pattern.shape)

def plot_recovery(original, ruidoso, recuperado, titulo, filename,
                  input_label='Patrón Ruidoso (Entrada)',
                  output_label='Patrón Recuperado (Salida)'):
    """Genera un gráfico comparativo: Original -> Ruidoso -> Recuperado"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    
    imagenes = [original, ruidoso, recuperado]
    titulos = ['Patrón Original', input_label, output_label]
    
    for ax, img, t in zip(axes, imagenes, titulos):
        ax.imshow(img, cmap='binary', vmin=-1, vmax=1)
        ax.set_title(t)
        ax.axis('off')
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        
    plt.suptitle(titulo, fontsize=14)
    plt.tight_layout()

    out_path = get_hopfield_output_path(filename)
    plt.savefig(out_path, dpi=150)
    print(f"Gráfico guardado: {out_path}")
    plt.close()

def plot_step_by_step(history, title, filename, energies=None):
    """
    Muestra cada paso de la recuperación como una grilla de matrices.
    history: lista de arrays 2D (cada paso).
    energies: lista opcional de energías por paso.
    """
    n = len(history)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(history[i], cmap='binary', vmin=-1, vmax=1)
            label = f"Paso {i}"
            if energies is not None and i < len(energies):
                label += f"\nE={energies[i]:.2f}"
            ax.set_title(label, fontsize=9)
            ax.axis('off')
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        else:
            ax.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = get_hopfield_output_path(filename)
    plt.savefig(out_path, dpi=150)
    print(f"Gráfico paso a paso guardado: {out_path}")
    plt.close()

def plot_energy_evolution(energies, title, filename):
    """Grafica la evolución de la energía por paso."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(energies)), energies, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    
    ax.set_xlabel("Paso")
    ax.set_ylabel("Energía")
    
    ax.set_xticks(range(len(energies)))
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = get_hopfield_output_path(filename)
    plt.savefig(out_path, dpi=150)
    print(f"Gráfico de energía guardado: {out_path}")
    plt.close()
def is_spurious(state, patterns, tolerance=0):
    """
    Verifica si un estado es espureo: no coincide con ninguno de los
    patrones almacenados (ni con su inverso).
    tolerance: cantidad de bits diferentes permitidos (0 = coincidencia exacta).
    """
    s = state.flatten()
    for p in patterns:
        pf = p.flatten()
        diff = np.sum(s != pf)
        if diff <= tolerance:
            return False
        # Chequear inverso (también es punto fijo)
        diff_inv = np.sum(s != -pf)
        if diff_inv <= tolerance:
            return False
    return True

def find_closest_pattern(state, pattern_names, patterns):
    """
    Retorna el nombre del patrón más cercano (por producto punto) y la distancia.
    """
    s = state.flatten()
    best_name = None
    best_dot = -np.inf
    for name, p in zip(pattern_names, patterns):
        dot = np.dot(s, p.flatten())
        if dot > best_dot:
            best_dot = dot
            best_name = name
    total = len(s)
    # Hamming distance = (total - dot) / 2
    hamming = int((total - best_dot) / 2)
    return best_name, hamming, best_dot

def plot_spurious_comparison(original_patterns, pattern_names, noisy_input,
                              recovered, title, filename):
    """
    Plot especial para mostrar un estado espureo:
    Fila 1: patrones almacenados
    Fila 2: entrada ruidosa → recuperado (espureo)
    """
    n_pat = len(original_patterns)
    cols = max(n_pat, 2)
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 7))

    # Fila 1: patrones almacenados
    for i in range(cols):
        ax = axes[0][i]
        if i < n_pat:
            ax.imshow(original_patterns[i], cmap='binary', vmin=-1, vmax=1)
            ax.set_title(f"Almacenado: {pattern_names[i]}", fontsize=10)
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.axis('off')

    # Fila 2: entrada y salida
    ax_in = axes[1][0]
    ax_in.imshow(noisy_input, cmap='binary', vmin=-1, vmax=1)
    ax_in.set_title("Entrada muy ruidosa", fontsize=10)
    ax_in.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax_in.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax_in.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax_in.axis('off')

    ax_out = axes[1][1]
    ax_out.imshow(recovered, cmap='binary', vmin=-1, vmax=1)
    ax_out.set_title("Estado ESPUREO", fontsize=10, color='red', fontweight='bold')
    ax_out.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax_out.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax_out.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax_out.axis('off')

    for i in range(2, cols):
        axes[1][i].axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = get_hopfield_output_path(filename)
    plt.savefig(out_path, dpi=150)
    print(f"Gráfico espureo guardado: {out_path}")
    plt.close()

def plot_capacity_results(num_stored_list, accuracy_list, filename):
    """Gráfico de tasa de recuperación vs. cantidad de patrones almacenados."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(num_stored_list, accuracy_list, 's-', color='#2ecc71', linewidth=2,
            markersize=8, markeredgecolor='#27ae60', markerfacecolor='white')
    ax.set_xlabel("Cantidad de patrones almacenados", fontsize=12)
    ax.set_ylabel("Tasa de recuperación correcta (%)", fontsize=12)
    ax.set_title("Capacidad de la red de Hopfield (25 neuronas)", fontsize=13)
    ax.set_ylim(-5, 105)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100%')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = get_hopfield_output_path(filename)
    plt.savefig(out_path, dpi=150)
    print(f"Gráfico de capacidad guardado: {out_path}")
    plt.close()

def plot_multi_recovery(results, suptitle, filename):
    """
    results: lista de tuplas (letra, original, ruidoso, recuperado, correcto)
    Muestra una grilla con N filas (una por letra) × 3 columnas.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original', 'Ruidoso', 'Recuperado']
    for i, (letra, orig, noisy, recov, ok) in enumerate(results):
        imgs = [orig, noisy, recov]
        for j, (ax, img) in enumerate(zip(axes[i], imgs)):
            ax.imshow(img, cmap='binary', vmin=-1, vmax=1)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=11)
            ax.axis('off')
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        status = "✓" if ok else "✗"
        color = 'green' if ok else 'red'
        axes[i][0].set_ylabel(f"{letra} {status}", fontsize=14, color=color,
                               fontweight='bold', rotation=0, labelpad=30)

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = get_hopfield_output_path(filename)
    plt.savefig(out_path, dpi=150)
    print(f"Gráfico multi-recovery guardado: {out_path}")
    plt.close()

def plot_abc(abecedario_dict, filename="plancha_abecedario.png"):
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
        'I': " *****\n ..*.. \n ..*.. \n ..*.. \n *****",
        'J': " ..***\n ...*.\n ...*.\n *..*.\n .**..",
        'K': " *...*\n *..*.\n ***.. \n *..*.\n *...*",
        'L': " *....\n *....\n *....\n *....\n *****",
        'M': " *...*\n **.**\n *.*.*\n *...*\n *...*",
        'N': " *...*\n **..*\n *.*.*\n *..**\n *...*",
        'O': " .***.\n *...*\n *...*\n *...*\n .***.",
        'P': " ****.\n *...*\n ****.\n *....\n *....",
        'Q': " .***.\n *...*\n *.*.*\n *..**\n .***.",
        'R': " ****.\n *...*\n ****.\n *..*.\n *...*",
        'S': " .****\n *....\n .***.\n ....*\n ****.",
        'T': " *****\n ..*.. \n ..*.. \n ..*.. \n ..*..",
        'U': " *...*\n *...*\n *...*\n *...*\n .***.",
        'V': " *...*\n *...*\n *...*\n .*.*.\n ..*..",
        'W': " *...*\n *...*\n *.*.*\n **.**\n *...*",
        'X': " *...*\n .*.*.\n ..*.. \n .*.*.\n *...*",
        'Y': " *...*\n .*.*.\n ..*.. \n ..*.. \n ..*..",
        'Z': " *****\n ...*.\n ..*.. \n .*...\n *****"
    }
    
    # Convertimos el diccionario de texto a un diccionario de matrices NumPy
    abecedario_matrices = {}
    for letra, ascii_art in alfabeto_ascii.items():
        abecedario_matrices[letra] = create_pattern(ascii_art)
        
    return abecedario_matrices