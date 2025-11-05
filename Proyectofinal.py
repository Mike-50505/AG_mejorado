import random
import pandas as pd
import blosum
import copy
import time
import matplotlib.pyplot as plt
import numpy as np

blosum62 = blosum.BLOSUM(62)
NFE = 0
start_time = time.time()

# Para guardar el historial de fitness para comparación
fitness_history_original = []
fitness_history_improved = []

def get_sequences():
    seq1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDRWGKLVVLGAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
    seq2 = "MKTLLVAAAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQKELQKQLGQKAKEL"
    seq3 = "MAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKALCVFAIN"
    return [list(seq1), list(seq2), list(seq3)]

def crear_individuo():
    return get_sequences()

def crear_poblacion_inicial(n=10):
    individuo_base = crear_individuo()
    poblacion = [ [row[:] for row in individuo_base] for _ in range(n) ]
    return poblacion

# ========== MEJORA 1: Mutación adaptativa ==========
def mutar_poblacion_adaptativa(poblacion, generacion, max_generaciones, num_gaps_min=1, num_gaps_max=3):
    """
    Mutación adaptativa: reduce la cantidad de gaps a medida que avanzan las generaciones
    """
    # Reducir progresivamente el número máximo de gaps
    progreso = generacion / max_generaciones
    num_gaps_actual = max(num_gaps_max - int(progreso * (num_gaps_max - num_gaps_min)), num_gaps_min)
    
    poblacion_mutada = []
    for individuo in poblacion:
        nuevo_individuo = []
        for fila in individuo:
            fila_mutada = fila[:]
            posiciones = set()
            
            # MEJORA: Probabilidad de mutación adaptativa
            prob_mutacion = 0.7 * (1 - progreso * 0.5)  # Reduce probabilidad con el tiempo
            if random.random() < prob_mutacion:
                for _ in range(num_gaps_actual):
                    pos = random.randint(0, len(fila_mutada))
                    while pos in posiciones:
                        pos = random.randint(0, len(fila_mutada))
                    posiciones.add(pos)
                    fila_mutada.insert(pos, '-')
            nuevo_individuo.append(fila_mutada)
        poblacion_mutada.append(nuevo_individuo)
    return poblacion_mutada

def igualar_longitud_secuencias(individuo, gap='-'):
    max_len = max(len(fila) for fila in individuo)
    individuo_igualado = [fila + [gap]*(max_len - len(fila)) for fila in individuo]
    return individuo_igualado

def evaluar_individuo_blosum62(individuo):
    global NFE
    NFE += 1
    score = 0
    n_seqs = len(individuo)
    seq_len = len(individuo[0])
    
    # MEJORA 2: Penalización adaptativa por gaps consecutivos
    for i in range(n_seqs):
        gaps_consecutivos = 0
        for col in range(seq_len):
            if individuo[i][col] == '-':
                gaps_consecutivos += 1
                # Penalización exponencial por gaps consecutivos
                if gaps_consecutivos > 1:
                    score -= 2 ** (gaps_consecutivos - 1)  # Penalización creciente
            else:
                gaps_consecutivos = 0
    
    # Scoring original BLOSUM
    for col in range(seq_len):
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                a = individuo[i][col]
                b = individuo[j][col]
                if a == '-' or b == '-':
                    score -= 4
                else:
                    score += blosum62[a][b]
                    
    return score

def eliminar_peores(poblacion, scores, porcentaje=0.5):
    idx_ordenados = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n_seleccionados = int(len(poblacion) * porcentaje)
    
    ind_seleccionados = [poblacion[i] for i in idx_ordenados[:n_seleccionados]]
    scores_seleccionados = [scores[i] for i in idx_ordenados[:n_seleccionados]]
    
    return ind_seleccionados, scores_seleccionados

# ========== MEJORA 3: Cruza con selección por torneo ==========
def seleccion_por_torneo(poblacion, scores, k=3):
    """Selección por torneo para elegir padres"""
    seleccionados = []
    for _ in range(len(poblacion)):
        # Escoger k individuos aleatorios
        torneo_indices = random.sample(range(len(poblacion)), k)
        # Escoger el mejor del torneo
        mejor_idx = max(torneo_indices, key=lambda i: scores[i])
        seleccionados.append(poblacion[mejor_idx])
    return seleccionados

def cruzar_individuos_doble_punto_mejorado(ind1, ind2):
    """Versión mejorada del cruce con más puntos de corte"""
    hijo1 = []
    hijo2 = []
    for seq1, seq2 in zip(ind1, ind2):
        aa_indices = [i for i, a in enumerate(seq1) if a != '-']
        if len(aa_indices) < 6:
            hijo1.append(seq1[:])
            hijo2.append(seq2[:])
            continue

        # MEJORA: Múltiples puntos de corte (2-4 puntos)
        num_puntos = random.randint(2, 4)
        puntos_corte = sorted(random.sample(aa_indices, min(num_puntos, len(aa_indices))))
        
        def cruza_mejorada(seqA, seqB):
            aaA = [a for a in seqA if a != '-']
            aaB = [a for a in seqB if a != '-']
            
            resultado = aaA[:]
            # Aplicar cruce en segmentos alternados
            for i in range(0, len(puntos_corte)-1, 2):
                start = puntos_corte[i]
                end = puntos_corte[i+1] if i+1 < len(puntos_corte) else len(aaA)
                if end > start:
                    resultado[start:end] = aaB[start:end]
            
            # Reconstruir con gaps
            resultado_final = []
            idx = 0
            for a in seqA:
                if a == '-':
                    resultado_final.append('-')
                else:
                    if idx < len(resultado):
                        resultado_final.append(resultado[idx])
                        idx += 1
                    else:
                        resultado_final.append(a)
            return resultado_final

        nueva_seq1 = cruza_mejorada(seq1, seq2)
        nueva_seq2 = cruza_mejorada(seq2, seq1)

        hijo1.append(nueva_seq1)
        hijo2.append(nueva_seq2)

    # MEJORA: Mutación con probabilidad adaptativa
    hijo1 = mutar_individuo_mejorado(hijo1, prob_mutacion=0.3)
    hijo2 = mutar_individuo_mejorado(hijo2, prob_mutacion=0.3)
    return hijo1, hijo2

def mutar_individuo_mejorado(individuo, prob_mutacion=0.3):
    """Mutación mejorada con diferentes tipos de mutación"""
    nuevo_individuo = []
    for secuencia in individuo:
        sec = secuencia[:]
        if random.random() < prob_mutacion:
            # MEJORA: Diferentes tipos de mutación
            tipo_mutacion = random.choice(['insertar', 'eliminar', 'cambiar'])
            
            if tipo_mutacion == 'insertar' and len(sec) > 0:
                # Insertar gap
                pos = random.randint(0, len(sec))
                sec.insert(pos, '-')
                
            elif tipo_mutacion == 'eliminar' and any(a == '-' for a in sec):
                # Eliminar gap aleatorio
                gaps_pos = [i for i, a in enumerate(sec) if a == '-']
                if gaps_pos:
                    pos = random.choice(gaps_pos)
                    sec.pop(pos)
                    
            elif tipo_mutacion == 'cambiar' and len(sec) > 1:
                # Cambiar posición de gap
                gaps_pos = [i for i, a in enumerate(sec) if a == '-']
                if gaps_pos:
                    gap_pos = random.choice(gaps_pos)
                    sec.pop(gap_pos)
                    nueva_pos = random.randint(0, len(sec))
                    sec.insert(nueva_pos, '-')
                    
        nuevo_individuo.append(sec)
    return nuevo_individuo

def cruzar_poblacion_mejorada(poblacion, scores):
    """
    Versión mejorada del cruce con selección por torneo
    """
    # MEJORA: Selección por torneo para padres
    padres = seleccion_por_torneo(poblacion, scores)
    
    nueva_poblacion = copy.deepcopy(poblacion)  # Mantener padres
    n = len(padres)
    indices = list(range(n))
    random.shuffle(indices)
    
    for i in range(0, n-1, 2):
        padre1 = padres[indices[i]]
        padre2 = padres[indices[i+1]]
        hijo1, hijo2 = cruzar_individuos_doble_punto_mejorado(padre1, padre2)
        nueva_poblacion.extend([hijo1, hijo2])
    
    return nueva_poblacion

# ========== MEJORA 4: Algoritmo original para comparación ==========
def algoritmo_original():
    global NFE
    NFE = 0
    start_time_orig = time.time()
    
    veryBest = None
    fitnessVeryBest = None
    poblacion = crear_poblacion_inicial(10)
    poblacion = mutar_poblacion_v2(poblacion, num_gaps=1)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
    poblacion, scores = eliminar_peores(poblacion, scores)
    
    fitness_history = []
    
    for generaciones in range(100):
        poblacion = cruzar_poblacion_doble_punto(poblacion)
        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        poblacion, scores = eliminar_peores(poblacion, scores)
        best, fitness_best = obtener_best(scores, poblacion)
        if veryBest is None or fitness_best > fitnessVeryBest:
            veryBest = best
            fitnessVeryBest = fitness_best
        fitness_history.append(fitnessVeryBest)
        
    end_time = time.time()
    transcurrido = end_time - start_time_orig
    print(f"ALGORITMO ORIGINAL - fitness: {fitnessVeryBest}, NFE: {NFE}, time: {transcurrido:.2f}s")
    
    return fitness_history, veryBest, fitnessVeryBest

# ========== MEJORA 5: Algoritmo mejorado ==========
def algoritmo_mejorado():
    global NFE
    NFE = 0
    start_time_improved = time.time()
    
    veryBest = None
    fitnessVeryBest = None
    poblacion = crear_poblacion_inicial(10)
    max_generaciones = 100
    
    fitness_history = []
    
    for generacion in range(max_generaciones):
        # MEJORA: Aplicar mutación adaptativa
        poblacion = mutar_poblacion_adaptativa(poblacion, generacion, max_generaciones)
        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        
        # MEJORA: Selección elitista - mantener el mejor siempre
        best, fitness_best = obtener_best(scores, poblacion)
        if veryBest is None or fitness_best > fitnessVeryBest:
            veryBest = best
            fitnessVeryBest = fitness_best
        
        poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.6)  # MEJORA: Mantener más diversidad
        
        # MEJORA: Cruza mejorada con selección por torneo
        poblacion = cruzar_poblacion_mejorada(poblacion, scores)
        
        fitness_history.append(fitnessVeryBest)
        
    end_time = time.time()
    transcurrido = end_time - start_time_improved
    print(f"ALGORITMO MEJORADO - fitness: {fitnessVeryBest}, NFE: {NFE}, time: {transcurrido:.2f}s")
    
    return fitness_history, veryBest, fitnessVeryBest

# ========== Funciones auxiliares originales (para comparación) ==========
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

def cruzar_individuos_doble_punto(ind1, ind2):
    hijo1 = []
    hijo2 = []
    for seq1, seq2 in zip(ind1, ind2):
        aa_indices = [i for i, a in enumerate(seq1) if a != '-']
        if len(aa_indices) < 6:
            hijo1.append(seq1[:])
            hijo2.append(seq2[:])
            continue

        intentos = 0
        while True:
            p1, p2 = sorted(random.sample(aa_indices, 2))
            if p2 - p1 >= 5 or intentos > 10:
                break
            intentos += 1

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

    hijo1 = mutar_individuo(hijo1, 1, 0.8)
    hijo2 = mutar_individuo(hijo2, 1, 0.8)
    return hijo1, hijo2

def mutar_individuo(individuo, n_gaps, p):
    nuevo_individuo = []
    for secuencia in individuo:
        sec = secuencia[:]
        if random.random() < p:
            posiciones = set()
            for _ in range(n_gaps):
                pos = random.randint(0, len(sec))
                while pos in posiciones:
                    pos = random.randint(0, len(sec))
                posiciones.add(pos)
                sec.insert(pos, '-')
        nuevo_individuo.append(sec)
    return nuevo_individuo

def cruzar_poblacion_doble_punto(poblacion):
    nueva_poblacion = []
    n = len(poblacion)
    indices = list(range(n))
    random.shuffle(indices)
    parejas = [(indices[i], indices[i+1]) for i in range(0, n-1, 2)]
    if n % 2 == 1:
        parejas.append((indices[-1], indices[0]))
    for idx1, idx2 in parejas:
        padre1 = poblacion[idx1]
        padre2 = poblacion[idx2]
        hijo1, hijo2 = cruzar_individuos_doble_punto(padre1, padre2)
        nueva_poblacion.append(copy.deepcopy(padre1))
        nueva_poblacion.append(copy.deepcopy(padre2))
        nueva_poblacion.append(hijo1)
        nueva_poblacion.append(hijo2)
    return nueva_poblacion[:2*n]

def validar_poblacion_sin_gaps(poblacion, originales):
    for individuo in poblacion:
        for seq, seq_orig in zip(individuo, originales):
            seq_sin_gaps = [a for a in seq if a != '-']
            seq_orig_sin_gaps = [a for a in seq_orig if a != '-']
            if seq_sin_gaps != seq_orig_sin_gaps:
                return False
    return True

def obtener_best(scores, poblacion):
    idx_mejor = scores.index(max(scores))
    fitness_best = scores[idx_mejor]
    best = copy.deepcopy(poblacion[idx_mejor])
    return best, fitness_best

# ========== Función para graficar comparación ==========
def graficar_comparacion(hist_original, hist_mejorado):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(hist_original, 'r-', label='Algoritmo Original', alpha=0.7)
    plt.plot(hist_mejorado, 'b-', label='Algoritmo Mejorado', alpha=0.7)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Comparación de Evolución del Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # Graficar mejora porcentual
    mejora_relativa = []
    for i in range(min(len(hist_original), len(hist_mejorado))):
        if hist_original[i] != 0:
            mejora = ((hist_mejorado[i] - hist_original[i]) / abs(hist_original[i])) * 100
            mejora_relativa.append(mejora)
    
    plt.plot(mejora_relativa, 'g-', label='Mejora Relativa (%)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Generación')
    plt.ylabel('Mejora (%)')
    plt.title('Mejora Relativa del Algoritmo Mejorado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacion_algoritmos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estadísticas finales
    mejora_final = ((hist_mejorado[-1] - hist_original[-1]) / abs(hist_original[-1])) * 100
    print(f"\n=== ESTADÍSTICAS FINALES ===")
    print(f"Fitness Original: {hist_original[-1]}")
    print(f"Fitness Mejorado: {hist_mejorado[-1]}")
    print(f"Mejora: {mejora_final:.2f}%")

if __name__ == "__main__":
    print("Ejecutando comparación de algoritmos...")
    
    # Ejecutar algoritmo original
    print("\n--- EJECUTANDO ALGORITMO ORIGINAL ---")
    hist_original, best_orig, fitness_orig = algoritmo_original()
    
    # Ejecutar algoritmo mejorado
    print("\n--- EJECUTANDO ALGORITMO MEJORADO ---")
    hist_mejorado, best_improved, fitness_improved = algoritmo_mejorado()
    
    # Validar integridad
    secuencias_originales = get_sequences()
    print(f"\nValidación integridad algoritmo original: {validar_poblacion_sin_gaps([best_orig], secuencias_originales)}")
    print(f"Validación integridad algoritmo mejorado: {validar_poblacion_sin_gaps([best_improved], secuencias_originales)}")
    
    # Graficar comparación
    graficar_comparacion(hist_original, hist_mejorado)