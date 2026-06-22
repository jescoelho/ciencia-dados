# =============================================================================
# ANÁLISE 05 — ÁRVORE DE DECISÃO PARA DEFAULT DE CARTÃO DE CRÉDITO
# =============================================================================
# Reproduz todas as etapas da análise em 05_arvore_decisao_default.md
#
# Dataset : UCI Default of Credit Card Clients (Yeh & Lien, 2009)
#           30.000 clientes de Taiwan
#           Período de observação: abril a setembro de 2005
#           Target: default em outubro de 2005
#
# Problema: classificação binária — prever inadimplência no mês seguinte
#           com base no histórico comportamental recente do cliente.
#
# Estrutura do script (segue CRISP-DM):
#   Fase 2 — Entendimento dos dados
#   Fase 3 — Preparação dos dados
#   Fase 4 — Modelagem
#   Fase 5 — Avaliação
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# Diretório do script — usado para construir caminhos absolutos.
# Isso garante que o script funcione independentemente de onde é chamado.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH  = SCRIPT_DIR / '../../../data/raw/UCI_Credit_Card.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, f1_score, confusion_matrix,
                              precision_score, recall_score)
from sklearn.calibration import calibration_curve

# Paleta de cores do projeto (fundo escuro, mesma do GitHub dark mode)
BG    = '#0d1117'   # fundo
BLUE  = '#58a6ff'   # cor principal
RED   = '#f78166'   # destaque / inadimplente
GREEN = '#3fb950'   # auxiliar / adimplente
GRAY  = '#8b949e'   # texto secundário


# =============================================================================
# FASE 2 — ENTENDIMENTO DOS DADOS
# =============================================================================
# Objetivo: mapear a estrutura bruta da base e identificar quais variáveis
# têm sinal discriminante para inadimplência, antes de qualquer transformação.
# =============================================================================

print("=" * 70)
print("FASE 2 — ENTENDIMENTO DOS DADOS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 Carregamento e inspeção inicial
# -----------------------------------------------------------------------------
# O dataset está no formato padrão UCI — uma linha por cliente, sem índice
# explícito além do ID. O strip() no nome das colunas remove espaços ocultos
# que causam KeyError silencioso em versões do arquivo baixadas com padding.
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n--- Estrutura bruta ---")
print(f"Shape: {df.shape}")           # esperado: (30000, 25)
print(df.dtypes.value_counts())       # float64 e int64 — sem strings
print(f"Nulos totais: {df.isnull().sum().sum()}")  # esperado: 0

# -----------------------------------------------------------------------------
# 2.2 Distribuição do target
# -----------------------------------------------------------------------------
# O primeiro passo em qualquer problema de classificação é verificar o
# balanceamento das classes. Desequilíbrio severo (< 5% de minoritária)
# exige estratégias mais agressivas (SMOTE, threshold customizado etc.).
# Aqui a classe minoritária representa 22% — desequilíbrio moderado.
tgt = 'default.payment.next.month'
y   = df[tgt]

print("\n--- Distribuição do target ---")
print(y.value_counts())
print(f"Taxa de default: {y.mean()*100:.2f}%")
# 77,9% adimplentes | 22,1% inadimplentes

# -----------------------------------------------------------------------------
# 2.3 Variáveis categóricas codificadas como inteiro
# -----------------------------------------------------------------------------
# EDUCATION e MARRIAGE foram originalmente categóricas mas estão como int.
# O dicionário oficial define categorias 1–4 para EDUCATION e 1–3 para MARRIAGE.
# Valores 0 aparecem em ambas — não documentados, precisam de tratamento.
print("\n--- EDUCATION ---")
print(df['EDUCATION'].value_counts().sort_index())
# 0: 14 registros  ← não documentado
# 1: pós-graduação | 2: graduação | 3: ensino médio | 4: outros | 5–6: desconhecido

print("\n--- MARRIAGE ---")
print(df['MARRIAGE'].value_counts().sort_index())
# 0: 54 registros  ← não documentado
# 1: casado | 2: solteiro | 3: outro

# -----------------------------------------------------------------------------
# 2.4 Status de pagamento — PAY_0 a PAY_6
# -----------------------------------------------------------------------------
# PAY_0 = setembro/05 (mais recente), PAY_6 = abril/05 (mais antigo).
# O dicionário define: -1 (pagou no prazo), 1–9 (meses de atraso).
# Valores 0 e -2 não estão documentados mas aparecem em todos os meses.
# A análise abaixo mostra que 0 e -2 têm taxas de default estáveis (~13–20%),
# indicando comportamentos reais de pagamento, não erros de entrada.
pay_status = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

print("\n--- Taxa de default por status de pagamento ---")
for col in pay_status:
    rates = df.groupby(col)[tgt].mean().round(3)
    print(f"\n{col}:\n{rates}")
# Padrão consistente: atraso >= 2 meses → taxa de default > 50% em todos os meses
# O degrau mais pronunciado é em PAY_0: PAY=1 (33,9%) → PAY=2 (69,1%)

# -----------------------------------------------------------------------------
# 2.5 Trajetórias temporais de pagamento
# -----------------------------------------------------------------------------
# Olhar cada mês isolado não captura a dinâmica: um cliente pode estar
# se recuperando (atrasado no passado, adimplente agora) ou em degradação
# (adimplente no passado, atrasado agora). A taxa de default difere muito
# entre essas trajetórias, o que justifica features que capturem essa dimensão.
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

def classifica_trajetoria(row):
    """Classifica cada cliente em uma das quatro trajetórias de pagamento."""
    recente, remoto = row['PAY_0'], row['PAY_6']
    if (row[pay_cols] <= 0).all():
        return 'Sempre adimplente'
    elif recente >= 2 and remoto <= 0:
        return 'Degradacao recente'
    elif recente >= 2 and remoto >= 2:
        return 'Atraso persistente'
    elif recente <= 0 and remoto >= 2:
        return 'Recuperacao'
    else:
        return 'Padrao misto'

def meses_consec_atraso(row):
    """Conta meses consecutivos de atraso (>= 2) a partir do mais recente."""
    consec = 0
    for col in pay_cols:
        if row[col] >= 2:
            consec += 1
        else:
            break
    return consec

df['trajetoria']   = df[pay_cols].apply(classifica_trajetoria, axis=1)
df['consec_atraso'] = df[pay_cols].apply(meses_consec_atraso, axis=1)

print("\n--- Taxa de default por trajetória ---")
traj = df.groupby('trajetoria')[tgt].agg(['mean', 'count']).rename(
    columns={'mean': 'taxa_default', 'count': 'n'})
traj['%base'] = (traj['n'] / len(df) * 100).round(1)
print(traj.sort_values('taxa_default', ascending=False))
# Atraso persistente: 76,6%  |  Degradação recente: 64,9%
# Recuperação: 27,3%  |  Padrão misto: 31,3%  |  Sempre adimplente: 11,7%

print("\n--- Taxa de default por meses consecutivos de atraso ---")
print(df.groupby('consec_atraso')[tgt].agg(['mean', 'count']))
# Um único mês consecutivo de atraso eleva a taxa de 16,6% para 65,7%

# -----------------------------------------------------------------------------
# 2.6 Limite de crédito — LIMIT_BAL
# -----------------------------------------------------------------------------
# LIMIT_BAL tem 81 valores únicos (múltiplos de NT$10.000) — política
# institucional de faixas de crédito. Dividido em quartis para ver o gradiente
# de risco. Quanto maior o limite, menor a inadimplência (relação monotônica).
df['lim_q'] = pd.qcut(df['LIMIT_BAL'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print("\n--- Taxa de default por quartil de LIMIT_BAL ---")
print(df.groupby('lim_q', observed=True)[tgt].mean().round(3))
# Q1=31,8%  Q2=24,7%  Q3=17,3%  Q4=14,0%

# -----------------------------------------------------------------------------
# 2.7 BILL_AMT e PAY_AMT — fatura e pagamentos
# -----------------------------------------------------------------------------
# BILL_AMT discrimina pouco: mediana similar entre classes em todos os meses.
# O sinal relevante está em PAY_AMT: inadimplentes pagam zero com ~10 p.p.
# de frequência maior que adimplentes. Zero em PAY_AMT não é ausência de fatura:
# 3.495 clientes tinham BILL_AMT1 > 0 e pagaram zero — desses, 39,8% defaultaram.
pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
meses        = ['set/05', 'ago/05', 'jul/05', 'jun/05', 'mai/05', 'abr/05']

print("\n--- Proporção de zeros em PAY_AMT por classe ---")
for col, mes in zip(pay_amt_cols, meses):
    z0 = df[df[tgt]==0][col].eq(0).mean() * 100
    z1 = df[df[tgt]==1][col].eq(0).mean() * 100
    print(f"{mes}  adimplente={z0:.1f}%  inadimplente={z1:.1f}%")

# Clientes com fatura > 0 e pagamento zero em setembro:
mask_zero_pag = (df['BILL_AMT1'] > 0) & (df['PAY_AMT1'] == 0)
print(f"\nClientes com fatura > 0 e pagamento zero (set/05): {mask_zero_pag.sum()}")
print(f"Taxa de default desse grupo: {df.loc[mask_zero_pag, tgt].mean():.3f}")

# -----------------------------------------------------------------------------
# 2.8 Outliers em variáveis contínuas
# -----------------------------------------------------------------------------
# Árvores de decisão podem criar splits espúrios se um único outlier extremo
# forçar um nó com n=1. Verificamos os extremos acima do P99 para ver se
# representam segmentos reais ou ruído. Resultado: extremos altos são
# sistematicamente mais seguros → representam clientes genuinamente diferentes,
# a árvore aprenderá a segmentá-los corretamente.
print("\n--- Outliers (acima do P99) ---")
for col in ['LIMIT_BAL'] + pay_amt_cols:
    p1, p99 = df[col].quantile([0.01, 0.99])
    dr_core = df.loc[(df[col] >= p1) & (df[col] <= p99), tgt].mean()
    dr_high = df.loc[df[col] > p99, tgt].mean()
    n_high  = (df[col] > p99).sum()
    print(f"{col:<12s}  p99={p99:>9,.0f}  n_acima={n_high}  "
          f"dr_core={dr_core:.3f}  dr_acima_p99={dr_high:.3f}")

# -----------------------------------------------------------------------------
# 2.9 Correlação ponto-bisserial com o target
# -----------------------------------------------------------------------------
# A correlação ponto-bisserial mede a associação entre uma variável contínua
# e um target binário. É calculada aqui sobre o dataset completo (30.000),
# antes de qualquer divisão treino-teste — o objetivo é mapear o sinal bruto
# de cada variável original, sem viés de seleção.
predictors = df.drop(columns=[tgt, 'ID', 'trajetoria', 'consec_atraso', 'lim_q'])
corrs = {}
for col in predictors.columns:
    r, p = pointbiserialr(predictors[col], y)
    corrs[col] = (r, p)

print("\n--- Correlação ponto-bisserial (preditores originais) ---")
for col in sorted(corrs, key=lambda c: abs(corrs[c][0]), reverse=True):
    r, p = corrs[col]
    sig = '' if p < 0.05 else '  [nao sig.]'
    print(f"{col:<20s}  r={r:+.4f}  p={p:.2e}{sig}")
# PAY_0 lidera (r=+0.32), seguido de PAY_2-6 e LIMIT_BAL (-0.15)
# Bloco BILL_AMT no fundo: r <= 0.023, três variáveis não-significativas
# Demográficas (SEX, MARRIAGE, EDUCATION, AGE): efeitos mínimos (r <= 0.04)

# -----------------------------------------------------------------------------
# 2.10 Variáveis demográficas
# -----------------------------------------------------------------------------
# Para variáveis categóricas usa-se qui-quadrado + V de Cramér.
# V < 0.10 = efeito negligenciável; V ≈ 0.30 = efeito moderado.
# Todos os testes atingem p < 0.05 por causa do n=30.000 — não efeito real.
print("\n--- Variáveis demográficas (V de Cramér) ---")
for var in ['SEX', 'MARRIAGE', 'EDUCATION']:
    ct = pd.crosstab(df[var], y)
    chi2, p, dof, _ = chi2_contingency(ct)
    v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
    print(f"{var:<12s}  V={v:.4f}  p={p:.2e}")
r_age, p_age = pointbiserialr(df['AGE'], y)
print(f"{'AGE':<12s}  r={r_age:+.4f}  p={p_age:.2e}")
# Maior efeito: EDUCATION V=0.074 — ainda negligenciável

# -----------------------------------------------------------------------------
# 2.11 Correlação entre preditores (colinearidade)
# -----------------------------------------------------------------------------
# Alta correlação entre dois preditores indica que carregam informação
# equivalente. O bloco BILL_AMT tem todos os 15 pares com r > 0.80 —
# colinearidade interna extrema, sem informação discriminante sobre o target.
# Isso justifica a remoção do bloco inteiro na Fase 3.
corr_m = df.drop(columns=[tgt, 'ID', 'trajetoria', 'consec_atraso', 'lim_q']).corr().abs()
upper  = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
high   = upper.stack().reset_index()
high.columns = ['var1', 'var2', 'r']

print("\n--- Pares com correlação > 0.70 ---")
print(high[high['r'] > 0.70].sort_values('r', ascending=False).to_string(index=False))
# Todos os 15 pares do bloco BILL_AMT acima de 0.80
# PAY_4–PAY_5 (0.82) e PAY_5–PAY_6 (0.82): colinearidade alta,
# mas sinal individual relevante (r ≈ 0.20) → mantidos


# =============================================================================
# FASE 3 — PREPARAÇÃO DOS DADOS
# =============================================================================
# Três decisões de limpeza (ID, recodificação, split) + feature engineering
# + seleção de features por análise de sinal e colinearidade.
# =============================================================================

print("\n" + "=" * 70)
print("FASE 3 — PREPARAÇÃO DOS DADOS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 Remoção de ID e colunas de EDA
# -----------------------------------------------------------------------------
# ID é identificador de linha — não tem relação causal com inadimplência.
# 'trajetoria' e 'consec_atraso' foram criadas para EDA e não são preditores
# diretos ('trajetoria' é string e seria codificada arbitrariamente).
df = df.drop(columns=['ID', 'trajetoria', 'consec_atraso', 'lim_q'])
print(f"\nShape após remoção de ID e colunas EDA: {df.shape}")  # (30000, 24)

# -----------------------------------------------------------------------------
# 3.2 Recodificação de categorias não documentadas
# -----------------------------------------------------------------------------
# EDUCATION=0 (14 registros) e MARRIAGE=0 (54 registros) não têm definição
# no dicionário. A decisão é agrupá-los na categoria "outros" já existente
# (EDUCATION=4, MARRIAGE=3) — não descartá-los, pois representam 0.2% da base
# e podem ter comportamento de default distinto.
# Os valores 0 e -2 em PAY_0-6 NÃO são recodificados: a Fase 2 mostrou
# taxas de default estáveis (12-20%), indicando comportamentos reais.
df['EDUCATION'] = df['EDUCATION'].replace(0, 4)
df['MARRIAGE']  = df['MARRIAGE'].replace(0, 3)
print("EDUCATION e MARRIAGE recodificados (0 → categoria 'outros')")

# -----------------------------------------------------------------------------
# 3.3 Divisão treino e teste
# -----------------------------------------------------------------------------
# Divisão 80/20 com estratificação pelo target.
# Estratificação garante que a proporção de inadimplentes (22%) seja
# preservada em ambas as partições — especialmente importante com desequilíbrio.
# Escalonamento NÃO é aplicado: árvores de decisão particionam o espaço
# por cortes binários e são invariantes a transformações monotônicas.
X = df.drop(columns=[tgt])
y = df[tgt]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTreino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
print(f"Taxa default treino: {y_train.mean():.4f} | teste: {y_test.mean():.4f}")
print(f"Features iniciais: {X_train.shape[1]}")  # 23 preditores originais

# -----------------------------------------------------------------------------
# 3.4 Feature engineering
# -----------------------------------------------------------------------------
# As quatro features abaixo foram derivadas após análise de sinal e colinearidade.
# São criadas nos dois conjuntos simultaneamente para evitar vazamento de dados.

pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
pay_amt_cols    = [f'PAY_AMT{i}' for i in range(1, 7)]

for Xp in [X_train, X_test]:
    # util_rate: taxa de utilização do crédito — razão entre fatura do mês mais
    # recente (BILL_AMT1 = setembro) e o limite total. Captura o grau de
    # comprometimento do limite. LIMIT_BAL mínimo é NT$10.000 — sem risco de
    # divisão por zero. Tem r=0.091 com o target, 4× maior que qualquer BILL_AMT.
    Xp['util_rate'] = Xp['BILL_AMT1'] / Xp['LIMIT_BAL']

    # n_meses_atraso: quantos dos 6 meses o cliente ficou em atraso (PAY >= 1).
    # Limiar >= 1 (e não >= 2) inclui PAY=1, cujos clientes têm 33,9% de default
    # em PAY_0 — risco intermediário real que a árvore pode refinar internamente.
    # Acabou sendo o preditor mais forte do modelo (r=0.401, importância 73,8%).
    Xp['n_meses_atraso'] = (Xp[pay_status_cols] >= 1).sum(axis=1)

    # total_pago: soma de todos os pagamentos no semestre. Captura o volume
    # absoluto de liquidação da dívida — distingue quem paga pouco de quem
    # paga muito, independentemente do status de atraso. r=-0.102 com target.
    Xp['total_pago'] = Xp[pay_amt_cols].sum(axis=1)

    # tendencia_pay: diferença entre o status mais recente (PAY_0) e o mais
    # antigo (PAY_6). Valores positivos indicam piora do comportamento ao longo
    # do semestre; negativos indicam recuperação. r=+0.131 com target.
    Xp['tendencia_pay'] = Xp['PAY_0'] - Xp['PAY_6']

# Candidatos avaliados e rejeitados (não incluídos acima):
# - pay_ratio_1 (PAY_AMT1/BILL_AMT1): p=0.31, não significativo
# - max_atraso (max dos PAY_0-6): r=0.766 com PAY_0, altamente redundante

print(f"\nCandidatos após feature engineering: {X_train.shape[1]}")  # 27

# -----------------------------------------------------------------------------
# 3.5 Análise de dimensionalidade — pós feature engineering
# -----------------------------------------------------------------------------
# Reavalia sinal e colinearidade sobre os 27 candidatos, no conjunto de treino.

# Correlação bisserial pós-FE
print("\n--- Correlação ponto-bisserial pós-FE (treino) ---")
corrs_fe = {}
for col in X_train.columns:
    r, p = pointbiserialr(X_train[col], y_train)
    corrs_fe[col] = (r, p)
for col in sorted(corrs_fe, key=lambda c: abs(corrs_fe[c][0]), reverse=True)[:12]:
    r, p = corrs_fe[col]
    print(f"  {col:<20s}  r={r:+.4f}")
# n_meses_atraso passa a liderar (r=0.401), acima de PAY_0 original (r=0.325)
# util_rate (r=0.091) supera qualquer BILL_AMT individual (r <= 0.023)

# -----------------------------------------------------------------------------
# 3.6 Remoção do bloco BILL_AMT
# -----------------------------------------------------------------------------
# Decisão fundamentada em dois critérios simultâneos:
# (1) Sinal quase nulo com o target (r <= 0.023; três variáveis não-significativas)
# (2) Colinearidade interna extrema (r entre 0.80 e 0.95 entre todos os pares)
# util_rate já captura a dimensão de utilização do crédito com 4× mais sinal
# e sem colinearidade relevante com os demais grupos de variáveis.
bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
X_train = X_train.drop(columns=bill_cols)
X_test  = X_test.drop(columns=bill_cols)

print(f"\nFeatures finais: {X_train.shape[1]}")  # 21 preditores
print("Preditores finais:")
print(X_train.columns.tolist())


# =============================================================================
# FASE 4 — MODELAGEM
# =============================================================================
# Árvore CART com poda combinada:
#   Pré-poda: min_samples_leaf=20 (evita folhas com poucos registros)
#   Pós-poda: ccp_alpha selecionado por validação cruzada
#   Ajuste de desequilíbrio: class_weight='balanced'
# =============================================================================

print("\n" + "=" * 70)
print("FASE 4 — MODELAGEM")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 Configuração base
# -----------------------------------------------------------------------------
# min_samples_leaf=20: impede folhas com menos de 20 registros, eliminando
#   partições estatisticamente instáveis sem restringir a profundidade da árvore.
#
# class_weight='balanced': compensa o desequilíbrio de 22% de inadimplentes.
#   O sklearn calcula: weight_k = n_total / (n_classes × n_k)
#   weight_0 = 24000 / (2 × 18679) ≈ 0.64
#   weight_1 = 24000 / (2 × 5321)  ≈ 2.26   → razão ~3.5× a favor da classe 1
#   Esse peso entra no cálculo do índice de Gini de Breiman em cada candidato
#   de corte, forçando splits que discriminem melhor inadimplentes.
#
# criterion='gini': índice de Gini de Breiman mede a probabilidade de
#   classificação incorreta num nó. Preferido à entropia por menor custo
#   computacional e desempenho empírico equivalente nesta escala.
#   NÃO confundir com o Gini de discriminação (2×AUC-1) da Fase 5.
dt_base = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)

# -----------------------------------------------------------------------------
# 4.2 Seleção do número de folds
# -----------------------------------------------------------------------------
# Avalia k=3, 5, 7, 10 em termos de média e desvio padrão do AUC em CV.
# k=5 é escolhido: média estável, variância baixa, ~1.061 inadimplentes por fold.
# k=7 tem média marginalmente maior (0.7718 vs 0.7713) mas o desvio padrão
# já começa a crescer; em k=10 quase dobra (0.0158) sem ganho adicional.
# k=5 segue a convenção da literatura e é adotado aqui.
print("\n--- Seleção do número de folds ---")
for k in [3, 5, 7, 10]:
    scores = cross_val_score(dt_base, X_train, y_train, cv=k, scoring='roc_auc')
    print(f"  k={k:2d}  mean={scores.mean():.4f}  std={scores.std():.4f}")

# -----------------------------------------------------------------------------
# 4.3 Pós-poda — caminho de ccp_alpha
# -----------------------------------------------------------------------------
# cost_complexity_pruning_path() cresce a árvore completa com a pré-poda
# definida e retorna todos os valores de ccp_alpha que produzem uma árvore
# distinta (um ponto de poda diferente). Para cada alpha, a árvore remove
# os ramos cujo custo de poda (redução de impureza / número de folhas
# removidas) é menor que alpha. Alpha=0 → árvore completa; alpha grande
# → árvore com apenas um nó.
#
# Avaliamos 60 alphas linearmente espaçados ao longo do caminho (não todos
# os ~500 valores, por custo computacional) e selecionamos o que maximiza
# AUC em CV de 5 folds.
path   = dt_base.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas  # ~502 alphas no caminho para msl=20

# Seleciona 60 alphas igualmente espaçados (exclui o último, que produziria
# uma árvore de nó único — degenera a AUC para ~0.50)
idx         = np.unique(np.round(np.linspace(0, len(alphas) - 2, 60)).astype(int))
alphas_eval = alphas[idx]

print("\n--- Busca de ccp_alpha ótimo (CV 5-fold, AUC-ROC) ---")
cv_scores = []
for a in alphas_eval:
    dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=20,
                                 class_weight='balanced', ccp_alpha=a,
                                 random_state=42)
    cv_scores.append(
        cross_val_score(dt, X_train, y_train, cv=5, scoring='roc_auc').mean()
    )

best_alpha = alphas_eval[np.argmax(cv_scores)]
print(f"  ccp_alpha ótimo: {best_alpha:.6f}  |  AUC CV: {max(cv_scores):.4f}")

# -----------------------------------------------------------------------------
# 4.4 Modelo final
# -----------------------------------------------------------------------------
# Treinado com os parâmetros ótimos sobre o conjunto completo de treino.
# A avaliação em ambas as partições confirma a generalização (gap treino-teste).
dt_final = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=20,
    class_weight='balanced',
    ccp_alpha=best_alpha,
    random_state=42
)
dt_final.fit(X_train, y_train)

auc_tr = roc_auc_score(y_train, dt_final.predict_proba(X_train)[:, 1])
auc_te = roc_auc_score(y_test,  dt_final.predict_proba(X_test)[:, 1])

print("\n--- Modelo final ---")
print(f"  Profundidade : {dt_final.get_depth()}")
print(f"  Folhas       : {dt_final.get_n_leaves()}")
print(f"  AUC treino   : {auc_tr:.4f}")
print(f"  AUC teste    : {auc_te:.4f}")
print(f"  Gap          : {auc_tr - auc_te:.4f}")
# Profundidade 5, 11 folhas — estrutura diretamente interpretável
# Gap 0.0086 → generalização sólida, sem overfitting

# -----------------------------------------------------------------------------
# 4.5 Otimização conjunta — min_samples_leaf × ccp_alpha
# -----------------------------------------------------------------------------
# A abordagem sequencial (fixar msl=20, depois buscar alpha) pode perder
# combinações em que uma pré-poda mais leve + pós-poda mais agressiva
# encontra um ponto de viés-variância superior. Testamos msl ∈ {5, 10, 20}
# com reotimização independente de ccp_alpha para cada valor.
# Resultado: reduzir msl aumenta variância sem reduzir viés — o modelo com
# msl=20 permanece a melhor combinação.
print("\n--- Busca conjunta min_samples_leaf × ccp_alpha ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for msl in [5, 10, 20]:
    dt_path = DecisionTreeClassifier(criterion='gini', min_samples_leaf=msl,
                                      class_weight='balanced', random_state=42)
    path_msl   = dt_path.cost_complexity_pruning_path(X_train, y_train)
    alphas_msl = path_msl.ccp_alphas
    idx_msl    = np.unique(
        np.round(np.linspace(0, len(alphas_msl) - 2, 60)).astype(int)
    )
    alphas_eval_msl = alphas_msl[idx_msl]

    best_a, best_cv = None, -1
    for a in alphas_eval_msl:
        dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=msl,
                                     class_weight='balanced', ccp_alpha=a,
                                     random_state=42)
        cv = cross_val_score(dt, X_train, y_train, cv=skf,
                              scoring='roc_auc').mean()
        if cv > best_cv:
            best_cv, best_a = cv, a

    dt_fit = DecisionTreeClassifier(criterion='gini', min_samples_leaf=msl,
                                     class_weight='balanced', ccp_alpha=best_a,
                                     random_state=42)
    dt_fit.fit(X_train, y_train)
    auc_tr_fit = roc_auc_score(y_train, dt_fit.predict_proba(X_train)[:, 1])
    auc_te_fit = roc_auc_score(y_test,  dt_fit.predict_proba(X_test)[:, 1])
    print(f"  msl={msl:2d}  alpha={best_a:.6f}  CV={best_cv:.4f}  "
          f"train={auc_tr_fit:.4f}  test={auc_te_fit:.4f}  "
          f"gap={auc_tr_fit - auc_te_fit:.4f}  "
          f"depth={dt_fit.get_depth()}  leaves={dt_fit.get_n_leaves()}")

# -----------------------------------------------------------------------------
# 4.6 Ajuste de class_weight
# -----------------------------------------------------------------------------
# 'balanced' define a razão ~3.5×. Testamos valores alternativos para verificar
# se há configuração que melhore recall sem comprometer demais a precision.
# 'balanced' empata em AUC e KS com {0:1, 1:3} e supera em F1.
print("\n--- Ajuste de class_weight ---")
weights  = ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}]
w_labels = ['balanced (~3.5×)', '{0:1, 1:2}', '{0:1, 1:3}', '{0:1, 1:4}']

for w, wl in zip(weights, w_labels):
    dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=20,
                                 class_weight=w, ccp_alpha=best_alpha,
                                 random_state=42)
    dt.fit(X_train, y_train)
    prob   = dt.predict_proba(X_test)[:, 1]
    pred   = dt.predict(X_test)
    auc    = roc_auc_score(y_test, prob)
    f1     = f1_score(y_test, pred)
    fpr_w, tpr_w, _ = roc_curve(y_test, prob)
    ks     = (tpr_w - fpr_w).max()
    auc_tr_w = roc_auc_score(y_train, dt.predict_proba(X_train)[:, 1])
    print(f"  {wl:<20s}  train={auc_tr_w:.4f}  test={auc:.4f}  "
          f"gap={auc_tr_w - auc:.4f}  F1={f1:.4f}  KS={ks:.4f}")

# -----------------------------------------------------------------------------
# 4.7 Otimização bayesiana com Optuna (F1-Score como objetivo)
# -----------------------------------------------------------------------------
# A busca conjunta é uma grade esparsa de dois parâmetros. O Optuna (TPE)
# explora simultaneamente seis parâmetros em 150 trials, sem grades predefinidas,
# otimizando F1-Score (alternativa ao AUC para verificar se há configuração
# que melhore o trade-off recall-precision).
# Resultado: F1 CV = 0.5399 em validação, mas no teste: AUC regride (-0.0052),
# gap aumenta (+0.0104). O ganho em CV não se transfere — overfitting na
# validação cruzada, não erro do Optuna. O baseline msl=20 permanece ótimo.
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'criterion':         trial.suggest_categorical('criterion',
                                     ['gini', 'entropy']),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
            'max_depth':         trial.suggest_int('max_depth', 3, 20),
            'ccp_alpha':         trial.suggest_float('ccp_alpha', 1e-5, 1e-2,
                                     log=True),
            'class_weight':      {0: 1, 1: trial.suggest_float(
                                     'class_weight_1', 1.0, 5.0)},
            'random_state': 42
        }
        dt = DecisionTreeClassifier(**params)
        skf_opt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(dt, X_train, y_train,
                               cv=skf_opt, scoring='f1').mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=150, show_progress_bar=False)

    print("\n--- Optuna (150 trials, objetivo: F1 CV) ---")
    print(f"  Melhor F1 CV : {study.best_value:.4f}")
    print(f"  Melhores parâmetros: {study.best_params}")

    # Avalia o modelo Optuna no teste e compara com baseline
    bp = study.best_params
    dt_opt = DecisionTreeClassifier(
        criterion=bp['criterion'],
        min_samples_leaf=bp['min_samples_leaf'],
        min_samples_split=bp['min_samples_split'],
        max_depth=bp['max_depth'],
        ccp_alpha=bp['ccp_alpha'],
        class_weight={0: 1, 1: bp['class_weight_1']},
        random_state=42
    )
    dt_opt.fit(X_train, y_train)
    prob_opt = dt_opt.predict_proba(X_test)[:, 1]
    pred_opt = dt_opt.predict(X_test)
    fpr_o, tpr_o, _ = roc_curve(y_test, prob_opt)

    print(f"\n  Optuna  → depth={dt_opt.get_depth()}  leaves={dt_opt.get_n_leaves()}")
    print(f"  AUC train={roc_auc_score(y_train, dt_opt.predict_proba(X_train)[:,1]):.4f}  "
          f"test={roc_auc_score(y_test, prob_opt):.4f}  "
          f"F1={f1_score(y_test, pred_opt):.4f}  "
          f"KS={(tpr_o - fpr_o).max():.4f}")
    print(f"  Baseline → AUC test={auc_te:.4f}  F1={f1_score(y_test, dt_final.predict(X_test)):.4f}")

except ImportError:
    print("\n  [Optuna não instalado — instale com: pip install optuna]")


# =============================================================================
# FASE 5 — AVALIAÇÃO
# =============================================================================
# Avalia o modelo final (msl=20, ccp_alpha=best_alpha, balanced) no teste.
# =============================================================================

print("\n" + "=" * 70)
print("FASE 5 — AVALIAÇÃO")
print("=" * 70)

y_prob = dt_final.predict_proba(X_test)[:, 1]
y_pred = dt_final.predict(X_test)

# -----------------------------------------------------------------------------
# 5.1 Discriminação — AUC e Average Precision
# -----------------------------------------------------------------------------
# AUC-ROC: área sob a curva ROC — probabilidade de que o modelo ranqueie um
#   inadimplente aleatório acima de um adimplente aleatório.
#   Valor de referência: 0.50 = aleatório, 1.0 = perfeito.
#
# Average Precision: área sob a curva Precision-Recall. Mais informativa
#   que AUC quando há desequilíbrio de classes, pois a baseline de um
#   classificador aleatório é a prevalência (22%), não 0.50.
auc = roc_auc_score(y_test, y_prob)
ap  = average_precision_score(y_test, y_prob)

print(f"\nAUC-ROC           : {auc:.4f}")
print(f"Average Precision : {ap:.4f}  (baseline aleatório: {y_test.mean():.2f})")

# -----------------------------------------------------------------------------
# 5.2 Matriz de confusão — threshold 0.5
# -----------------------------------------------------------------------------
# O threshold 0.5 é a convenção padrão do scikit-learn — não tem justificativa
# de negócio intrínseca. Com class_weight='balanced', as probabilidades de
# saída estão sistematicamente acima da taxa real de default (22%), então
# 0.5 pode não ser o corte ótimo. A seção 5.3 explora isso.
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n--- Matriz de confusão (threshold=0.5) ---")
print(f"  [[TN={tn}  FP={fp}]")
print(f"   [FN={fn}  TP={tp}]]")
print(f"\n  Acurácia      : {(tn+tp)/(tn+fp+fn+tp)*100:.1f}%")
print(f"  Recall (sens) : {tp/(tp+fn)*100:.1f}%  ← % inadimplentes capturados")
print(f"  Precision     : {tp/(tp+fp)*100:.1f}%  ← % sinalizados que são reais")
print(f"  F1-Score      : {f1_score(y_test, y_pred):.4f}")
print(f"  Especificidade: {tn/(tn+fp)*100:.1f}%  ← % adimplentes classificados corretamente")

# FN = 566: inadimplentes não detectados → perda direta não provisionada
# FP = 808: adimplentes acionados → custo operacional sem retorno

# -----------------------------------------------------------------------------
# 5.3 Otimização do threshold operacional
# -----------------------------------------------------------------------------
# A escolha do threshold é uma decisão operacional independente do modelo.
# O modelo entrega o score; o threshold define a ação de cobrança preventiva.
# Em crédito, o custo de FN (perda não detectada) costuma ser maior que o
# custo de FP (cliente bom acionado por engano). A análise abaixo mapeia
# o trade-off recall-precision em todos os thresholds.
print("\n--- Otimização do threshold ---")
thresholds = np.linspace(0.05, 0.95, 91)
rows = []
for t in thresholds:
    pred_t = (y_prob >= t).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, pred_t).ravel()
    rec  = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    f1_t = 2*rec*prec / (rec+prec) if (rec+prec) > 0 else 0
    fpr_t = fp_t / (fp_t + tn_t)
    rows.append((t, rec, prec, f1_t, fpr_t))

df_t  = pd.DataFrame(rows, columns=['t', 'recall', 'precision', 'f1', 'fpr'])
best_t = df_t.loc[df_t['f1'].idxmax()]
print(f"  Threshold ótimo por F1: {best_t.t:.3f}  "
      f"F1={best_t.f1:.4f}  recall={best_t.recall:.4f}  "
      f"precision={best_t.precision:.4f}")

# Tabela de thresholds operacionais relevantes
print("\n  Tabela de thresholds operacionais:")
print(f"  {'t':>6s}  {'Recall':>8s}  {'Precision':>10s}  {'F1':>8s}  {'FPR':>8s}")
for t_ref in [0.30, 0.38, 0.44, 0.50, best_t.t]:
    row = df_t.iloc[(df_t['t'] - t_ref).abs().argmin()]
    print(f"  {row.t:>6.2f}  {row.recall:>8.4f}  {row.precision:>10.4f}  "
          f"{row.f1:>8.4f}  {row.fpr:>8.4f}")

# Nota: a árvore tem apenas 11 folhas → ~11 probabilidades de saída distintas
# (uma por folha). Isso limita a granularidade da decisão: só ~7 thresholds
# produzem predições diferentes. O threshold 0.59 maximiza F1 mas o ganho
# sobre 0.50 é marginal (ΔF1 ≈ 0.009). Para cobrança preventiva, t=0.38
# (recall 73%) ou t=0.44 (recall 68%) são mais adequados que F1 ótimo.

# -----------------------------------------------------------------------------
# 5.4 KS, Gini e Calibração
# -----------------------------------------------------------------------------
fpr_roc, tpr_roc, _ = roc_curve(y_test, y_prob)
ks_stat = (tpr_roc - fpr_roc).max()
gini    = 2 * auc - 1

print(f"\nKS    : {ks_stat:.4f}  (> 0.40 = bom na escala de crédito)")
print(f"Gini  : {gini:.4f}  (= 2×AUC-1; linguagem padrão em comitês de risco)")
print(f"F1    : {f1_score(y_test, y_pred):.4f}")

# KS (Kolmogorov-Smirnov): mede a maior separação entre a CDF dos scores
#   de adimplentes e inadimplentes. Calculado aqui como max(TPR - FPR) ao
#   longo da curva ROC — equivalente à definição clássica de scorecard.
#   Escala de referência: < 0.20 fraco | 0.20–0.40 aceitável | > 0.40 bom.
#   KS=0.4105 → limiar inferior da faixa "bom".

# Calibração: com class_weight='balanced', o modelo treina como se a
#   prevalência fosse 50% (pesos iguais), mas a real é 22%. As probabilidades
#   de saída estão sistematicamente infladas — úteis para ranquear clientes,
#   mas NÃO para estimar probabilidade absoluta de default (ex: provisão,
#   capital regulatório). Para isso, calibração pós-treino é necessária
#   (Platt scaling ou regressão isotônica).
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
print("\n--- Calibração (prob prevista vs taxa real de default por bin) ---")
print(f"  {'Prob prevista':>14s}  {'Taxa real':>10s}  {'Desvio':>8s}")
for pt, pp in zip(prob_true, prob_pred):
    print(f"  {pp:>14.3f}  {pt:>10.3f}  {pp-pt:>+8.3f}")
# Desvios positivos: probabilidades previstas maiores que a taxa real → inflação

# -----------------------------------------------------------------------------
# 5.5 Importância das variáveis
# -----------------------------------------------------------------------------
# feature_importances_ no sklearn para árvores CART é a redução total de
# impureza de Gini acumulada por cada variável ao longo de todos os splits
# em que ela foi usada, normalizada para somar 1.
# Variáveis com importância zero não foram selecionadas por nenhum split
# que sobreviveu à poda — não necessariamente sem sinal, mas insuficientes
# para adicionar discriminação além do que as outras já oferecem.
importances_all = pd.Series(dt_final.feature_importances_, index=X_train.columns)
importances     = importances_all[importances_all > 0].sort_values(ascending=False)

print("\n--- Importância das variáveis (redução de impureza) ---")
print(importances.round(4).to_string())
# n_meses_atraso domina com ~73.8% — a feature engineered que condensa
# 6 meses de histórico em uma contagem é o sinal mais forte do modelo.
# Apenas 6 dos 21 preditores foram usados; as 15 variáveis com importância
# zero incluem todas as demográficas e a maior parte das PAY_AMT.

print(f"\nVariáveis utilizadas: {(importances_all > 0).sum()} de {len(X_train.columns)}")
print(f"Variáveis ignoradas : {(importances_all == 0).sum()}")

# -----------------------------------------------------------------------------
# 5.6 Resumo final de métricas
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMO FINAL — MODELO BASELINE (msl=20, ccp_alpha ótimo, balanced)")
print("=" * 70)
print(f"  Parâmetros    : min_samples_leaf=20  ccp_alpha={best_alpha:.6f}  "
      f"class_weight=balanced")
print(f"  Estrutura     : profundidade={dt_final.get_depth()}  "
      f"folhas={dt_final.get_n_leaves()}")
print(f"  AUC-ROC       : {auc:.4f}")
print(f"  Gini          : {gini:.4f}")
print(f"  KS            : {ks_stat:.4f}")
print(f"  F1 (t=0.50)   : {f1_score(y_test, y_pred):.4f}")
print(f"  AP            : {ap:.4f}")
print(f"  Gap tr-te     : {auc_tr - auc:.4f}")
print(f"  Recall (t=0.50): {recall_score(y_test, y_pred):.4f}")
print(f"  Recall (t=0.38): {df_t.iloc[(df_t['t']-0.38).abs().argmin()]['recall']:.4f}")
print()
print("  Nota: AUC 0.7685 é o teto empírico de um único CART neste dataset,")
print("  confirmado por busca conjunta de hiperparâmetros (msl x ccp_alpha),")
print("  otimização bayesiana Optuna (150 trials) e seleção de features.")
print("  Para superar esse teto é necessário mudar a classe de modelo.")
