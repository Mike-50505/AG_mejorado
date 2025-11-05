#!/usr/bin/env python3
# ga_mejorado_blosum_reparado_fix.py
# Versión corregida: evita IndexError en cruza mezclando longitudes entre padres.

import random
import copy
import json
import time
from typing import List, Tuple
import math
import os

# ----- BLOSUM62 (tabla estándar) -----
BLOSUM62_TEXT = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""

def parse_blosum62(text: str):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    header = lines[0].split()
    mat = {}
    for line in lines[1:]:
        parts = line.split()
        row_letter = parts[0]
        values = list(map(int, parts[1:]))
        mat[row_letter] = {col: val for col, val in zip(header, values)}
    return mat

BLOSUM62 = parse_blosum62(BLOSUM62_TEXT)
GAP_PENALTY = -4

# ----- Secuencias originales -----
def get_sequences() -> List[List[str]]:
    seq1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDRWGKLVVLGAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
    seq2 = "MKTLLVAAAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQKELQKQLGQKAKEL"
    seq3 = "MAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKALCVFAIN"
    return [list(seq1), list(seq2), list(seq3)]

# ----- Utilidades -----
def igualar_longitud_secuencias(individuo: List[List[str]], gap: str='-') -> List[List[str]]:
    max_len = max(len(s) for s in individuo)
    return [s + [gap] * (max_len - len(s)) for s in individuo]

def pad_to_length(individuo: List[List[str]], target_len:int, gap:str='-') -> List[List[str]]:
    return [s + [gap]*(max(0, target_len - len(s))) for s in individuo]

def validar_poblacion_sin_gaps(poblacion: List[List[List[str]]], originales: List[List[str]]) -> bool:
    for individuo in poblacion:
        for seq, seq_orig in zip(individuo, originales):
            if [a for a in seq if a != '-'] != [a for a in seq_orig if a != '-']:
                return False
    return True

def reparar_integridad(individuo: List[List[str]], originales: List[List[str]]) -> List[List[str]]:
    repaired = []
    for idx, seq in enumerate(individuo):
        base = seq[:]
        orig_aa = [c for c in originales[idx] if c != '-']
        new_seq = []
        orig_i = 0
        for ch in base:
            if ch == '-':
                new_seq.append('-')
            else:
                if orig_i < len(orig_aa):
                    new_seq.append(orig_aa[orig_i]); orig_i += 1
                else:
                    new_seq.append('-')
        while orig_i < len(orig_aa):
            new_seq.append(orig_aa[orig_i]); orig_i += 1
        repaired.append(new_seq)
    repaired = igualar_longitud_secuencias(repaired)
    return repaired

# ----- Scoring BLOSUM62 -----
def evaluar_individuo_blosum62(individuo: List[List[str]]) -> int:
    individuo_eq = igualar_longitud_secuencias(individuo)
    L = len(individuo_eq[0])
    score = 0
    n = len(individuo_eq)
    for c in range(L):
        col = [seq[c] for seq in individuo_eq]
        for i in range(n):
            for j in range(i+1, n):
                a = col[i]; b = col[j]
                if a == '-' or b == '-':
                    score += GAP_PENALTY
                else:
                    score += BLOSUM62[a][b]
    return score

# ----- Operadores -----
def mutar_individuo_insert_gaps(individuo: List[List[str]], n_gaps: int=1, p_insert: float=0.6) -> List[List[str]]:
    nuevo = []
    for seq in individuo:
        s = seq[:]
        if random.random() < p_insert:
            for _ in range(n_gaps):
                pos = random.randint(0, len(s))
                s.insert(pos, '-')
        nuevo.append(s)
    return igualar_longitud_secuencias(nuevo)

def mover_gap(seq: List[str]) -> List[str]:
    s = seq[:]
    gaps = [i for i, ch in enumerate(s) if ch == '-']
    if not gaps:
        return s
    g = random.choice(gaps)
    dir = random.choice([-1, 1])
    new_pos = g + dir
    if 0 <= new_pos < len(s):
        s[g], s[new_pos] = s[new_pos], s[g]
    return s

def refinamiento_local(individuo: List[List[str]], originales: List[List[str]], max_iters: int=40) -> Tuple[List[List[str]], int]:
    curr = igualar_longitud_secuencias(individuo)
    curr = reparar_integridad(curr, originales)
    best_score = evaluar_individuo_blosum62(curr)
    best = curr
    for _ in range(max_iters):
        candidate = copy.deepcopy(best)
        op = random.random()
        seq_idx = random.randrange(len(candidate))
        seq = candidate[seq_idx]
        if op < 0.4:
            seq = mover_gap(seq)
        elif op < 0.7:
            pos = random.randint(0, len(seq))
            seq.insert(pos, '-')
        else:
            gaps = [i for i, c in enumerate(seq) if c == '-']
            if gaps:
                pos = random.choice(gaps)
                seq.pop(pos)
        candidate[seq_idx] = seq
        candidate = igualar_longitud_secuencias(candidate)
        candidate = reparar_integridad(candidate, originales)
        sc = evaluar_individuo_blosum62(candidate)
        if sc > best_score:
            best_score = sc
            best = candidate
    return best, best_score

def cruza_mejorada_con_reparacion(p1: List[List[str]], p2: List[List[str]], originales: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    # FIX: asegurarse de que ambos padres se paddeen al mismo target length
    max_len_p1 = max(len(s) for s in p1)
    max_len_p2 = max(len(s) for s in p2)
    L = max(max_len_p1, max_len_p2)
    p1 = pad_to_length(p1, L)
    p2 = pad_to_length(p2, L)

    if L < 2:
        return copy.deepcopy(p1), copy.deepcopy(p2)

    cut = random.randint(1, L-1)
    child1 = []
    child2 = []
    for seqA, seqB in zip(p1, p2):
        new1 = [seqA[i] if i < cut else seqB[i] for i in range(L)]
        new2 = [seqB[i] if i < cut else seqA[i] for i in range(L)]
        child1.append(new1)
        child2.append(new2)

    child1 = reparar_integridad(child1, originales)
    child2 = reparar_integridad(child2, originales)
    return child1, child2

# ----- Selección -----
def torneo_seleccion(poblacion: List[List[List[str]]], scores: List[int], k: int=3) -> List[List[str]]:
    n = len(poblacion)
    candidatos = random.sample(range(n), min(k, n))
    best_idx = max(candidatos, key=lambda i: scores[i])
    return copy.deepcopy(poblacion[best_idx])

# ----- GA -----
def run_ga_one(originales: List[List[str]],
               generations: int = 60,
               pop_size: int = 30,
               initial_gaps: int = 2,
               elitism_ratio: float = 0.1,
               tournament_k: int = 3,
               verbose: bool = False) -> Tuple[List[int], List[List[List[str]]], List[int]]:
    def crear_poblacion_inicial(n):
        base = originales
        return [ [row[:] for row in base] for _ in range(n) ]

    poblacion = crear_poblacion_inicial(pop_size)
    poblacion = [mutar_individuo_insert_gaps(ind, n_gaps=initial_gaps, p_insert=1.0) for ind in poblacion]
    poblacion = [reparar_integridad(ind, originales) for ind in poblacion]
    scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]

    bests_per_gen = []
    elitismo = max(1, int(math.ceil(elitism_ratio * pop_size)))
    for gen in range(generations):
        nueva = []
        idx_ord = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        elites = [copy.deepcopy(poblacion[i]) for i in idx_ord[:elitismo]]
        nueva.extend(elites)

        while len(nueva) < pop_size:
            padre1 = torneo_seleccion(poblacion, scores, k=tournament_k)
            padre2 = torneo_seleccion(poblacion, scores, k=tournament_k)
            hijo1, hijo2 = cruza_mejorada_con_reparacion(padre1, padre2, originales)
            p_mut = max(0.05, 0.5 - (gen / generations) * 0.45)
            if random.random() < p_mut:
                hijo1 = mutar_individuo_insert_gaps(hijo1, n_gaps=1, p_insert=0.6)
                hijo1 = reparar_integridad(hijo1, originales)
            if random.random() < p_mut:
                hijo2 = mutar_individuo_insert_gaps(hijo2, n_gaps=1, p_insert=0.6)
                hijo2 = reparar_integridad(hijo2, originales)
            hijo1, s1 = refinamiento_local(hijo1, originales, max_iters=12)
            hijo2, s2 = refinamiento_local(hijo2, originales, max_iters=12)
            nueva.append(hijo1)
            if len(nueva) < pop_size:
                nueva.append(hijo2)
        poblacion = [igualar_longitud_secuencias(ind) for ind in nueva[:pop_size]]
        poblacion = [reparar_integridad(ind, originales) for ind in poblacion]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        bests_per_gen.append(max(scores))
        if verbose:
            print(f"Gen {gen+1}/{generations} - best: {bests_per_gen[-1]}")
    return bests_per_gen, poblacion, scores

# ----- Múltiples ejecuciones -----
def run_multiple(experiments: int = 5, generations: int = 60, pop_size: int = 30, seed_base: int = 42):
    originales = get_sequences()
    all_runs = []
    for r in range(experiments):
        seed = seed_base + r
        random.seed(seed)
        bests, pobl, scores = run_ga_one(originales, generations=generations, pop_size=pop_size, initial_gaps=2, verbose=False)
        all_runs.append(bests)
    G = generations
    avg = [sum(run[g] for run in all_runs)/len(all_runs) for g in range(G)]
    med = [sorted([run[g] for run in all_runs])[len(all_runs)//2] for g in range(G)]
    return {
        'all_runs': all_runs,
        'avg': avg,
        'median': med
    }

# ----- Script principal -----
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t0 = time.time()
    EXPERIMENTS = 5
    GENERATIONS = 60
    POP_SIZE = 30

    resultados = run_multiple(experiments=EXPERIMENTS, generations=GENERATIONS, pop_size=POP_SIZE, seed_base=100)
    avg = resultados['avg']
    med = resultados['median']

    out_dir = "ga_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fitness_all_runs.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)

    plt.figure(figsize=(9,5))
    gens = list(range(1, GENERATIONS+1))
    for run in resultados['all_runs']:
        plt.plot(gens, run, color='gray', alpha=0.25)
    plt.plot(gens, avg, label='Promedio (mejor por gen)', linewidth=2)
    plt.plot(gens, med, label='Mediana', linestyle='--')
    plt.xlabel("Generación")
    plt.ylabel("Mejor fitness (BLOSUM62 sum pairwise)")
    plt.title("Evolución (GA mejorado con reparación de integridad) - FIXED")
    plt.legend(); plt.grid(True); plt.tight_layout()
    png_path = os.path.join(out_dir, "comparativa_promedio_fixed.png")
    plt.savefig(png_path, dpi=200); plt.close()

    t1 = time.time()
    print("Experimentos:", EXPERIMENTS)
    print("Generaciones:", GENERATIONS)
    print("Población:", POP_SIZE)
    print("Tiempo total (s):", round(t1 - t0, 2))
    print("Gráfica guardada en:", png_path)

    originales = get_sequences()
    random.seed(999)
    bests_final, pobl_final, scores_final = run_ga_one(originales, generations=GENERATIONS, pop_size=POP_SIZE, initial_gaps=2, verbose=False)
    ok = validar_poblacion_sin_gaps(pobl_final, originales)
    print("Validación de integridad (última run):", ok)
