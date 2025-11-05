# ga_mejorado_completo.py
import random
import copy
import time
import statistics
import os
import matplotlib.pyplot as plt

# Requiere el paquete 'blosum' que expone BLOSUM(62) como blosum.BLOSUM(62)
import blosum

# Semilla por defecto para reproducibilidad
DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)
blosum62 = blosum.BLOSUM(62)

# ---------- Secuencias originales (constantes) ----------
def get_sequences():
    seq1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDRWGKLVVLGAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
    seq2 = "MKTLLVAAAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQKELQKQLGQKAKEL"
    seq3 = "MAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKALCVFAIN"
    return [list(seq1), list(seq2), list(seq3)]

ORIG_SEQS = get_sequences()

# ---------- Validación de integridad ----------
def validar_poblacion_sin_gaps(poblacion, originales):
    for individuo in poblacion:
        for seq, seq_orig in zip(individuo, originales):
            seq_sin_gaps = [a for a in seq if a != '-']
            seq_orig_sin_gaps = [a for a in seq_orig if a != '-']
            if seq_sin_gaps != seq_orig_sin_gaps:
                return False
    return True

# ---------- Utilidades básicas ----------
def igualar_longitud_secuencias(individuo, gap='-'):
    max_len = max(len(fila) for fila in individuo)
    individuo_igualado = [fila + [gap] * (max_len - len(fila)) for fila in individuo]
    return individuo_igualado

def evaluar_individuo_blosum62(individuo):
    score = 0
    n_seqs = len(individuo)
    seq_len = len(individuo[0])
    for col in range(seq_len):
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                a = individuo[i][col]
                b = individuo[j][col]
                if a == '-' or b == '-':
                    score -= 4
                else:
                    score += blosum62[a][b]
    return score

# ---------- Creación y mutación inicial (compatibles con original) ----------
def crear_individuo():
    return get_sequences()

def crear_poblacion_inicial(n=10):
    individuo_base = crear_individuo()
    poblacion = [[row[:] for row in individuo_base] for _ in range(n)]
    return poblacion

def mutar_poblacion_v2(poblacion, num_gaps=1):
    poblacion_mutada = []
    for individuo in poblacion:
        nuevo_individuo = []
        for fila in individuo:
            fila_mutada = fila[:]
            posiciones = set()
            for _ in range(num_gaps):
                pos = random.randint(0, len(fila_mutada))
                while pos in posiciones:
                    pos = random.randint(0, len(fila_mutada))
                posiciones.add(pos)
                fila_mutada.insert(pos, '-')
            nuevo_individuo.append(fila_mutada)
        poblacion_mutada.append(nuevo_individuo)
    return poblacion_mutada

# Garantizar no eliminar por debajo de 2 individuos
def eliminar_peores(poblacion, scores, porcentaje=0.5):
    idx_ordenados = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n_seleccionados = max(2, int(len(poblacion) * porcentaje))
    ind_seleccionados = [poblacion[i] for i in idx_ordenados[:n_seleccionados]]
    scores_seleccionados = [scores[i] for i in idx_ordenados[:n_seleccionados]]
    return ind_seleccionados, scores_seleccionados

# ---------- Cruza original (doble punto ignorando gaps) ----------
def cruzar_individuos_doble_punto_original(ind1, ind2):
    hijo1 = []
    hijo2 = []
    for seq1, seq2 in zip(ind1, ind2):
        aa_indices = [i for i, a in enumerate(seq1) if a != '-']
        if len(aa_indices) < 6:
            hijo1.append(seq1[:])
            hijo2.append(seq2[:])
            continue
        p1, p2 = sorted(random.sample(aa_indices, 2))
        if p2 - p1 < 1:
            p2 = min(len(aa_indices) - 1, p1 + 1)
        def cruza(seqA, seqB):
            aaA = [a for a in seqA if a != '-']
            aaB = [a for a in seqB if a != '-']
            nueva = aaA[:p1] + aaB[p1:p2] + aaA[p2:]
            resultado = []
            idx = 0
            for a in seqA:
                if a == '-':
                    resultado.append('-')
                else:
                    resultado.append(nueva[idx])
                    idx += 1
            return resultado
        nueva_seq1 = cruza(seq1, seq2)
        nueva_seq2 = cruza(seq2, seq1)
        hijo1.append(nueva_seq1)
        hijo2.append(nueva_seq2)
    return hijo1, hijo2

def cruzar_poblacion_doble_punto_original(poblacion):
    if len(poblacion) < 2:
        return [copy.deepcopy(ind) for ind in poblacion]
    nueva = []
    n = len(poblacion)
    indices = list(range(n))
    random.shuffle(indices)
    parejas = [(indices[i], indices[i+1]) for i in range(0, n - 1, 2)]
    if n % 2 == 1:
        parejas.append((indices[-1], indices[0]))
    for a, b in parejas:
        padre1 = poblacion[a]
        padre2 = poblacion[b]
        hijo1, hijo2 = cruzar_individuos_doble_punto_original(padre1, padre2)
        nueva.append(copy.deepcopy(padre1))
        nueva.append(copy.deepcopy(padre2))
        nueva.append(hijo1)
        nueva.append(hijo2)
    return nueva[:2 * n]

# ---------- Mejoras: selección torneo elitista, cruza reproyectada, mutación adaptativa ----------
def seleccion_torneo(poblacion, scores, k=3, elite_frac=0.1):
    n = len(poblacion)
    if n == 0:
        return []
    k_eff = min(k, max(1, n))
    elite_n = max(1, int(n * elite_frac))
    idx_ordenados = sorted(range(n), key=lambda i: scores[i], reverse=True)
    elites = [copy.deepcopy(poblacion[i]) for i in idx_ordenados[:elite_n]]
    seleccion = elites[:]
    while len(seleccion) < n:
        candidatos = random.sample(range(n), k_eff)
        mejor = max(candidatos, key=lambda i: scores[i])
        seleccion.append(copy.deepcopy(poblacion[mejor]))
    return seleccion

def estructura_gaps_de(secuencia):
    return [1 if a == '-' else 0 for a in secuencia]

def reproyectar_gaps_por_estructura(secuencia_con_gaps, estructura_ref):
    seq_aa = [a for a in secuencia_con_gaps if a != '-']
    resultado = []
    idx = 0
    for is_gap in estructura_ref:
        if is_gap:
            resultado.append('-')
        else:
            if idx < len(seq_aa):
                resultado.append(seq_aa[idx])
                idx += 1
            else:
                resultado.append('-')
    # si quedan aa extras, intentar colocarlas en posiciones no-gap
    if idx < len(seq_aa):
        for i in range(len(resultado)):
            if resultado[i] == '-' and idx < len(seq_aa):
                resultado[i] = seq_aa[idx]
                idx += 1
    return resultado

def mutar_individuo_mejorado(individuo, p_base=0.2, n_gaps=1):
    nuevo = []
    for seq in individuo:
        s = seq[:]
        if random.random() < p_base:
            posiciones = set()
            for _ in range(n_gaps):
                pos = random.randint(0, len(s))
                while pos in posiciones:
                    pos = random.randint(0, len(s))
                posiciones.add(pos)
                s.insert(pos, '-')
        nuevo.append(s)
    return nuevo

def cruzar_individuos_mejorado(ind1, ind2):
    hijo1 = []
    hijo2 = []
    for seq1, seq2 in zip(ind1, ind2):
        aa1 = [a for a in seq1 if a != '-']
        aa2 = [a for a in seq2 if a != '-']
        if min(len(aa1), len(aa2)) < 4:
            hijo1.append(seq1[:])
            hijo2.append(seq2[:])
            continue
        # puntos sobre índices de aminoácidos
        max_idx = min(len(aa1), len(aa2)) - 1
        p1 = random.randint(0, max_idx // 2)
        p2 = random.randint(max_idx // 2 + 1, max_idx) if max_idx // 2 + 1 <= max_idx else max_idx
        if p1 > p2:
            p1, p2 = p2, p1
        def cruza_reproyectar(sA, sB, refA):
            aaA = [a for a in sA if a != '-']
            aaB = [a for a in sB if a != '-']
            # proteger límites
            p1_loc = min(p1, len(aaA))
            p2_loc = min(p2, len(aaA))
            nueva = aaA[:p1_loc] + aaB[p1_loc:p2_loc] + aaA[p2_loc:]
            resultado = reproyectar_gaps_por_estructura(nueva, refA)
            return resultado
        refA = seq1
        refB = seq2
        nueva1 = cruza_reproyectar(seq1, seq2, refA)
        nueva2 = cruza_reproyectar(seq2, seq1, refB)
        hijo1.append(nueva1)
        hijo2.append(nueva2)
    hijo1 = mutar_individuo_mejorado(hijo1, p_base=0.15, n_gaps=1)
    hijo2 = mutar_individuo_mejorado(hijo2, p_base=0.15, n_gaps=1)
    return hijo1, hijo2

def cruzar_poblacion_mejorada(poblacion, scores):
    if len(poblacion) < 2:
        return [copy.deepcopy(ind) for ind in poblacion]
    seleccion = seleccion_torneo(poblacion, scores, k=3, elite_frac=0.2)
    nueva = []
    n = len(seleccion)
    indices = list(range(n))
    random.shuffle(indices)
    parejas = [(indices[i], indices[i+1]) for i in range(0, n - 1, 2)]
    if n % 2 == 1:
        parejas.append((indices[-1], indices[0]))
    for a, b in parejas:
        padre1 = seleccion[a]
        padre2 = seleccion[b]
        hijo1, hijo2 = cruzar_individuos_mejorado(padre1, padre2)
        nueva.append(copy.deepcopy(padre1))
        nueva.append(copy.deepcopy(padre2))
        nueva.append(hijo1)
        nueva.append(hijo2)
    return nueva[:2 * n]

def diversidad_poblacional(poblacion):
    if not poblacion:
        return 0.0
    L = max(len(seq) for seq in poblacion[0])
    col_sets = []
    for col in range(L):
        s = set()
        for ind in poblacion:
            for seq in ind:
                if col < len(seq):
                    s.add(seq[col])
        col_sets.append(len(s))
    return statistics.mean(col_sets)

# ---------- Ejecución de algoritmos (original y mejorado) ----------
def ejecutar_algoritmo_original(seed=DEFAULT_SEED, ngen=60, pop_size=20):
    random.seed(seed)
    poblacion = crear_poblacion_inicial(pop_size)
    poblacion = mutar_poblacion_v2(poblacion, num_gaps=1)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
    poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
    historial_mean = []
    historial_best = []
    historial_valid = []
    for g in range(ngen):
        poblacion = cruzar_poblacion_doble_punto_original(poblacion)
        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
        historial_mean.append(statistics.mean(scores))
        historial_best.append(max(scores))
        historial_valid.append(validar_poblacion_sin_gaps(poblacion, ORIG_SEQS))
    return historial_mean, historial_best, historial_valid

def ejecutar_algoritmo_mejorado(seed=DEFAULT_SEED, ngen=60, pop_size=20):
    random.seed(seed)
    poblacion = crear_poblacion_inicial(pop_size)
    poblacion = mutar_poblacion_v2(poblacion, num_gaps=1)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
    poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
    historial_mean = []
    historial_best = []
    historial_valid = []
    stagnation_counter = 0
    best_prev = max(scores)
    for g in range(ngen):
        div = diversidad_poblacional(poblacion)
        poblacion = cruzar_poblacion_mejorada(poblacion, scores)
        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
        mean_s = statistics.mean(scores)
        best_s = max(scores)
        historial_mean.append(mean_s)
        historial_best.append(best_s)
        historial_valid.append(validar_poblacion_sin_gaps(poblacion, ORIG_SEQS))
        if best_s <= best_prev:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_prev = best_s
        if stagnation_counter >= 6:
            poblacion = [mutar_individuo_mejorado(ind, p_base=0.4, n_gaps=1) for ind in poblacion]
            poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
            scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
            poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
            stagnation_counter = 0
    return historial_mean, historial_best, historial_valid

# ---------- Comparación y guardado de gráfica ----------
def comparar_y_graficar(seed=DEFAULT_SEED, ngen=60, pop_size=20, outdir="resultados"):
    os.makedirs(outdir, exist_ok=True)
    mo, bo, vo = ejecutar_algoritmo_original(seed=seed, ngen=ngen, pop_size=pop_size)
    mm, bm, vm = ejecutar_algoritmo_mejorado(seed=seed, ngen=ngen, pop_size=pop_size)
    generaciones = list(range(1, ngen + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(generaciones, bo, label="Original - mejor", color='red', alpha=0.8)
    plt.plot(generaciones, mo, label="Original - medio", linestyle='--', color='red', alpha=0.5)
    plt.plot(generaciones, bm, label="Mejorado - mejor", color='blue', alpha=0.9)
    plt.plot(generaciones, mm, label="Mejorado - medio", linestyle='--', color='blue', alpha=0.6)
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Comparación: Algoritmo original vs Algoritmo mejorado")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(outdir, "comparacion_fitness.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    resumen = {
        "original_best_final": bo[-1],
        "original_mean_final": mo[-1],
        "mejorado_best_final": bm[-1],
        "mejorado_mean_final": mm[-1],
        "validacion_original_all": all(vo),
        "validacion_mejorado_all": all(vm)
    }
    return resumen, fname, (generaciones, bo, mo, bm, mm, vo, vm)

# ---------- Ejecución principal ----------
if __name__ == "__main__":
    start = time.time()
    resumen, img_path, datos = comparar_y_graficar(seed=DEFAULT_SEED, ngen=60, pop_size=24)
    end = time.time()
    print("Resumen comparativo:")
    for k, v in resumen.items():
        print(f"  {k}: {v}")
    print("Gráfica guardada en:", img_path)
    print("Tiempo de ejecución (s):", round(end - start, 2))

