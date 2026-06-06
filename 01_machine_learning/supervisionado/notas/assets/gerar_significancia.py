"""Gera o grafico de significancia estatistica para a nota de regressao linear multipla."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

BG     = "#0d1117"
EDGE   = "#30363d"
GRID   = "#21262d"
TEXT   = "#c9d1d9"
MUTED  = "#8b949e"
GREEN  = "#3fb950"
BLUE   = "#58a6ff"
ORANGE = "#f0883e"

def style_ax(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

np.random.seed(42)
n, p = 120, 5
X = np.random.randn(n, p)
beta_true = np.array([1.8, 0.10, -0.14, 0.06, -1.5])
y = X @ beta_true + 1.2 + np.random.randn(n) * 1.3

# OLS
X_aug = np.column_stack([np.ones(n), X])
beta_hat = np.linalg.lstsq(X_aug, y, rcond=None)[0]
y_hat = X_aug @ beta_hat
residuals = y - y_hat

dof = n - p - 1
s2 = np.sum(residuals**2) / dof
cov_b = s2 * np.linalg.inv(X_aug.T @ X_aug)
se = np.sqrt(np.diag(cov_b))[1:]
beta_coef = beta_hat[1:]
t_vals = beta_coef / se

t_crit = stats.t.ppf(0.975, dof)
ci_lo = beta_coef - t_crit * se
ci_hi = beta_coef + t_crit * se
significant = (ci_lo > 0) | (ci_hi < 0)

SS_tot = np.sum((y - np.mean(y))**2)
SS_res = np.sum(residuals**2)
SS_reg = SS_tot - SS_res
R2 = SS_reg / SS_tot
F_stat = (R2 / p) / ((1 - R2) / dof)
F_pval = 1 - stats.f.cdf(F_stat, p, dof)

# ── Figura ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.subplots_adjust(wspace=0.42)

# Painel 1 — gráfico de coeficientes (forest plot)
labels = [f"$x_{j+1}$" for j in range(p)]
y_pos  = np.arange(p)

x_min = min(ci_lo) - 0.5
x_max = max(ci_hi) + 0.9  # espaco para anotacoes a direita

for i, (b, lo, hi, sig, tv) in enumerate(zip(beta_coef, ci_lo, ci_hi, significant, t_vals)):
    color = GREEN if sig else MUTED
    ax1.plot([lo, hi], [i, i], color=color, lw=2.5, solid_capstyle="round", zorder=3)
    ax1.scatter([b], [i], color=color, s=55, zorder=4)
    # anotacao sempre a direita, fora das barras
    ax1.text(x_max - 0.05, i, f"t = {tv:.1f}", color=color, fontsize=8,
             va="center", ha="right")

ax1.axvline(0, color=EDGE, lw=1.5, ls="--", zorder=2)
ax1.set_xlim(x_min, x_max)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=11, color=TEXT)
ax1.set_xlabel(r"Valor do coeficiente $\hat{\beta}_j$ com IC 95%")
ax1.set_title("t-test por coeficiente\nIC que nao cruza zero = significativo (p < 0.05)",
              color=TEXT, fontsize=10, pad=10)
ax1.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.7)
ax1.grid(axis="y", visible=False)

patch_sig = mpatches.Patch(color=GREEN, label="Significativo")
patch_ns  = mpatches.Patch(color=MUTED,  label="Nao significativo")
ax1.legend(handles=[patch_sig, patch_ns], fontsize=8,
           facecolor="#161b22", edgecolor=EDGE, labelcolor=TEXT, loc="lower left")
style_ax(ax1)

# Painel 2 — decomposicao da variancia (F-test)
bar_h = 0.5
ax2.barh(0, SS_reg, height=bar_h, color=GREEN, alpha=0.85, label="Explicada (SSR)")
ax2.barh(0, SS_res, height=bar_h, left=SS_reg, color=MUTED, alpha=0.60, label="Residual (SSE)")

ax2.text(SS_reg * 0.5, 0, f"SSR\n{R2:.0%}", color=BG,
         fontsize=10, ha="center", va="center", fontweight="bold")
ax2.text(SS_reg + SS_res * 0.5, 0, f"SSE\n{1-R2:.0%}", color=BG,
         fontsize=9, ha="center", va="center", fontweight="bold")

p_label = "< 0.001" if F_pval < 0.001 else f"= {F_pval:.3f}"
ax2.text(0.5, -0.40, f"F({p}, {dof}) = {F_stat:.1f}     p {p_label}",
         transform=ax2.transAxes, ha="center", color=TEXT, fontsize=9)
ax2.text(0.5, -0.56, f"$R^2$ = {R2:.3f}",
         transform=ax2.transAxes, ha="center", color=GREEN, fontsize=9)

ax2.set_yticks([])
ax2.set_xlim(0, SS_tot * 1.04)
ax2.set_ylim(-0.55, 0.55)
ax2.set_xlabel("Variacao de $y$ (SST = SSR + SSE)")
ax2.set_title("F-test — decomposicao da variancia total\nquanto maior a fatia verde, mais o modelo supera o nulo",
              color=TEXT, fontsize=10, pad=10)
ax2.legend(fontsize=8, facecolor="#161b22", edgecolor=EDGE, labelcolor=TEXT, loc="upper right")
ax2.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.7)
ax2.grid(axis="y", visible=False)
style_ax(ax2)

plt.savefig("01_linear_significancia.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("ok 01_linear_significancia.png")
