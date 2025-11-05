

import random
import copy
import time
import math
import matplotlib.pyplot as plt
import pandas as pd

random.seed(42)

def get_sequences():
    seq1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDRWGKLVVLGAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
    seq2 = "MKTLLVAAAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQKELQKQLGQKAKEL"
    seq3 = "MAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKALCVFAIN"
    return [list(seq1), list(seq2), list(seq3)]

def score_column(a, b):
    if a == '-' or b == '-':
        return -4
    return 5 if a == b else -1

def evaluate_individuo(individuo):
    score = 0
    n_seqs = len(individuo)
    seq_len = len(individuo[0])
    for col in range(seq_len):
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                score += score_column(individuo[i][col], individuo[j][col])
    return score

def igualar_longitud_secuencias(individuo, gap='-'):
    max_len = max(len(fila) for fila in individuo)
    return [fila + [gap]*(max_len - len(fila)) for fila in individuo]

def validar_poblacion_sin_gaps(poblacion, originales):
    for individuo in poblacion:
        for seq, seq_orig in zip(individuo, originales):
            seq_sin_gaps = [a for a in seq if a != '-']
            seq_orig_sin_gaps = [a for a in seq_orig if a != '-']
            if seq_sin_gaps != seq_orig_sin_gaps:
                return False
    return True

def crear_poblacion_inicial(n=10):
    individuo_base = get_sequences()
    poblacion = [ [row[:] for row in individuo_base] for _ in range(n) ]
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

def eliminar_peores(poblacion, scores, porcentaje=0.5):
    idx_ordenados = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n_seleccionados = int(len(poblacion) * porcentaje)
    return [poblacion[i] for i in idx_ordenados[:n_seleccionados]], [scores[i] for i in idx_ordenados[:n_seleccionados]]

def mutar_individuo_simple(individuo, n_gaps, p):
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

def cruzar_individuos_doble_punto_simple(ind1, ind2):
    hijo1, hijo2 = [], []
    for seq1, seq2 in zip(ind1, ind2):
        aa_indices = [i for i, a in enumerate(seq1) if a != '-']
        if len(aa_indices) < 6:
            hijo1.append(seq1[:]); hijo2.append(seq2[:]); continue
        p1, p2 = sorted(random.sample(aa_indices, 2))
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
                    resultado.append(nueva[idx]); idx += 1
            return resultado
        hijo1.append(cruza(seq1, seq2))
        hijo2.append(cruza(seq2, seq1))
    hijo1 = mutar_individuo_simple(hijo1, 1, 0.8)
    hijo2 = mutar_individuo_simple(hijo2, 1, 0.8)
    return hijo1, hijo2

def cruzar_poblacion_doble_punto_simple(poblacion):
    nueva_poblacion = []
    n = len(poblacion); indices = list(range(n)); random.shuffle(indices)
    parejas = [(indices[i], indices[i+1]) for i in range(0, n-1, 2)]
    if n % 2 == 1:
        parejas.append((indices[-1], indices[0]))
    for idx1, idx2 in parejas:
        padre1 = poblacion[idx1]; padre2 = poblacion[idx2]
        hijo1, hijo2 = cruzar_individuos_doble_punto_simple(padre1, padre2)
        nueva_poblacion += [copy.deepcopy(padre1), copy.deepcopy(padre2), hijo1, hijo2]
    return nueva_poblacion[:2*n]

def run_original(pop_size=10, generations=50, num_gaps=1, seed=None):
    if seed is not None: random.seed(seed)
    poblacion = crear_poblacion_inicial(pop_size)
    poblacion = mutar_poblacion_v2(poblacion, num_gaps=num_gaps)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    scores = [evaluate_individuo(ind) for ind in poblacion]
    poblacion, scores = eliminar_peores(poblacion, scores)
    bests = []
    for gen in range(generations):
        poblacion = cruzar_poblacion_doble_punto_simple(poblacion)
        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluate_individuo(ind) for ind in poblacion]
        poblacion, scores = eliminar_peores(poblacion, scores)
        bests.append(max(scores))
    return bests, poblacion

# --- Improved implementation ---
def crear_poblacion_inicial_variada(n=10):
    base = get_sequences()
    poblacion = []
    for _ in range(n):
        individuo = [row[:] for row in base]
        for i, fila in enumerate(individuo):
            for _ in range(random.randint(0,2)):
                pos = random.randint(0, len(fila)); fila.insert(pos, '-')
        poblacion.append(individuo)
    return poblacion

def torneo_seleccion(poblacion, scores, k=3):
    selected = []
    n = len(poblacion)
    for _ in range(n):
        contenders = random.sample(range(n), k)
        best = max(contenders, key=lambda i: scores[i])
        selected.append(copy.deepcopy(poblacion[best]))
    return selected

def crossover_remap_preserve_gaps(parent1, parent2):
    child1, child2 = [], []
    for s1, s2 in zip(parent1, parent2):
        raw1 = [a for a in s1 if a != '-']; raw2 = [a for a in s2 if a != '-']
        if len(raw1) < 4 or len(raw2) < 4:
            child1.append(s1[:]); child2.append(s2[:]); continue
        cp = random.randint(1, min(len(raw1), len(raw2))-1)
        new1 = raw1[:cp] + raw2[cp:]; new2 = raw2[:cp] + raw1[cp:]
        pattern = []
        for a,b in zip(s1, s2):
            if a == '-' and b == '-': pattern.append('-')
            elif a == '-': pattern.append('-' if random.random()<0.5 else 'A')
            elif b == '-': pattern.append('-' if random.random()<0.5 else 'A')
            else: pattern.append('A')
        if pattern.count('A') < len(new1):
            res1=[];idx=0
            for ch in s1:
                if ch == '-': res1.append('-')
                else: res1.append(new1[idx]); idx+=1
            child1.append(res1)
        else:
            res1=[];idx=0
            for p in pattern:
                if p == '-': res1.append('-')
                else:
                    if idx < len(new1): res1.append(new1[idx]); idx+=1
                    else: res1.append('-')
            child1.append(res1)
        if pattern.count('A') < len(new2):
            res2=[];idx=0
            for ch in s2:
                if ch == '-': res2.append('-')
                else: res2.append(new2[idx]); idx+=1
            child2.append(res2)
        else:
            res2=[];idx=0
            for p in pattern:
                if p == '-': res2.append('-')
                else:
                    if idx < len(new2): res2.append(new2[idx]); idx+=1
                    else: res2.append('-')
            child2.append(res2)
    return child1, child2

def mutacion_mejorada(individuo, p_ins=0.2, p_del=0.1, max_ins=2):
    nuevo = []
    for seq in individuo:
        s = seq[:]
        if random.random() < p_del:
            gaps = [i for i,a in enumerate(s) if a == '-']
            if gaps: s.pop(random.choice(gaps))
        if random.random() < p_ins:
            for _ in range(random.randint(1,max_ins)):
                pos = random.randint(0, len(s)); s.insert(pos, '-')
        if random.random() < 0.05:
            gaps = [i for i,a in enumerate(s) if a == '-']
            if gaps:
                i = random.choice(gaps); j = i + random.choice([-1,1])
                if 0 <= j < len(s): s[i], s[j] = s[j], s[i]
        nuevo.append(s)
    return nuevo

def hill_climb_local(individuo, eval_fn, tries=20):
    best = individuo; best_score = eval_fn(igualar_longitud_secuencias(best))
    for _ in range(tries):
        cand = copy.deepcopy(best)
        seq_i = random.randrange(len(cand)); seq = cand[seq_i]
        gaps = [i for i,a in enumerate(seq) if a == '-']
        if not gaps:
            pos = random.randint(0, len(seq)); seq.insert(pos, '-')
            gaps2 = [i for i,a in enumerate(seq) if a == '-']
            if gaps2: seq.pop(random.choice(gaps2))
        else:
            gpos = random.choice(gaps); newpos = max(0, min(len(seq)-1, gpos + random.choice([-1,1])))
            seq.pop(gpos); seq.insert(newpos, '-')
        cand[seq_i] = seq
        cand_eq = igualar_longitud_secuencias(cand); sc = eval_fn(cand_eq)
        if sc > best_score:
            best = cand; best_score = sc
    return best, best_score

def run_improved(pop_size=10, generations=50, seed=None):
    if seed is not None: random.seed(seed)
    poblacion = crear_poblacion_inicial_variada(pop_size)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    scores = [evaluate_individuo(ind) for ind in poblacion]
    bests=[]
    for gen in range(generations):
        idx_orden = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        elite = [copy.deepcopy(poblacion[idx_orden[0]])]
        selected = torneo_seleccion(poblacion, scores, k=3)
        nueva=[]
        for i in range(0, len(selected), 2):
            a=selected[i]; b=selected[(i+1)%len(selected)]
            h1,h2 = crossover_remap_preserve_gaps(a,b)
            nueva += [h1,h2]
        p_ins = max(0.05, 0.3 * (1 - gen/generations)); p_del = 0.08
        nueva_mut = [mutacion_mejorada(ind, p_ins, p_del) for ind in nueva]
        for i in range(min(2, len(nueva_mut))):
            cand, sc = hill_climb_local(nueva_mut[i], evaluate_individuo, tries=25)
            nueva_mut[i] = cand
        poblacion = elite + nueva_mut[:pop_size-1]
        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluate_individuo(ind) for ind in poblacion]
        bests.append(max(scores))
    return bests, poblacion


def average_runs(func, runs=6, **kwargs):
    all_bests=[]
    for r in range(runs):
        b, _ = func(seed=100+r, **kwargs)
        all_bests.append(b)
    min_len = min(len(b) for b in all_bests)
    arr = [b[:min_len] for b in all_bests]
    df = pd.DataFrame(arr)
    return df.mean(axis=0).tolist(), df.std(axis=0).tolist(), df

gens = 50; runs = 6
orig_mean, orig_std, orig_df = average_runs(run_original, runs=runs, pop_size=10, generations=gens, num_gaps=1)
impr_mean, impr_std, impr_df = average_runs(run_improved, runs=runs, pop_size=10, generations=gens)

summary = pd.DataFrame({
    'generation': range(len(orig_mean)),
    'original_mean_best': orig_mean,
    'original_std': orig_std,
    'improved_mean_best': impr_mean,
    'improved_std': impr_std
})
summary.to_csv('fitness_summary.csv', index=False)

plt.figure(figsize=(9,5))
plt.plot(summary['generation'], summary['original_mean_best'], label='Original (mean best)')
plt.plot(summary['generation'], summary['improved_mean_best'], label='Improved (mean best)')
plt.xlabel('Generation'); plt.ylabel('Mean best fitness'); plt.title('Original vs Improved')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('compare_plot.png'); plt.show()

_, final_pop = run_improved(pop_size=10, generations=20, seed=12345)
print("Integrity:", validar_poblacion_sin_gaps(final_pop, get_sequences()))
