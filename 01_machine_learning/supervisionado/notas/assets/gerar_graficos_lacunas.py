"""Gera os três gráficos das lacunas inseridas na nota de regressão logística."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
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
RED    = "#da3633"

def style_ax(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)


# ── Gráfico 1: Fronteira de decisão ──────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.subplots_adjust(wspace=0.38)

# Painel 1 — 1D: sigmoide com fronteira marcada
beta0, beta1 = -1.2, 1.8
x = np.linspace(-5, 5, 400)
p = 1 / (1 + np.exp(-(beta0 + beta1 * x)))
x_star = -beta0 / beta1

ax1.plot(x, p, color=GREEN, lw=2.5, zorder=3)
ax1.axhline(0.5, color=MUTED, lw=1, ls="--", alpha=0.6)
ax1.axvline(x_star, color=ORANGE, lw=2, ls="--", zorder=4)
ax1.fill_between(x, 0, 1, where=(x < x_star), alpha=0.10, color=BLUE)
ax1.fill_between(x, 0, 1, where=(x >= x_star), alpha=0.10, color=GREEN)
ax1.scatter([x_star], [0.5], color=ORANGE, s=70, zorder=5)

ax1.text(x_star - 0.2, 0.53, r"$x^* = -\beta_0/\beta_1$", color=ORANGE,
         fontsize=9, ha="right")
ax1.text(-3.2, 0.12, r"prever $\hat{y}=0$", color=BLUE, fontsize=9, alpha=0.9)
ax1.text(1.4, 0.82, r"prever $\hat{y}=1$", color=GREEN, fontsize=9, alpha=0.9)

ax1.set_xlabel("$x$")
ax1.set_ylabel(r"$\hat{p}(x)$")
ax1.set_xlim(-5, 5)
ax1.set_ylim(-0.05, 1.05)
ax1.set_title(r"Fronteira em 1D — ponto onde $\hat{p}=0.5$",
              color=ORANGE, fontsize=10, pad=10)
style_ax(ax1)

# Painel 2 — 2D: fronteira como hiperplano no espaço dos preditores
np.random.seed(42)
n = 180
X2 = np.random.randn(n, 2)
y2 = (1.0 * X2[:, 0] - 0.7 * X2[:, 1] - 0.2 + 0.7 * np.random.randn(n) > 0).astype(int)

clf2 = LogisticRegression(max_iter=500).fit(X2, y2)
b0, b1, b2 = clf2.intercept_[0], clf2.coef_[0, 0], clf2.coef_[0, 1]

xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 250), np.linspace(-3.5, 3.5, 250))
Z = clf2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

ax2.contourf(xx, yy, Z, levels=30, cmap="RdYlGn", alpha=0.28, vmin=0, vmax=1)
ax2.contour(xx, yy, Z, levels=[0.5], colors=[ORANGE], linewidths=2.2)

ax2.scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], color=BLUE,  s=22, alpha=0.75, label="$y=0$")
ax2.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], color=GREEN, s=22, alpha=0.75, label="$y=1$")

# anotação da equação da fronteira
ax2.annotate(r"$\mathbf{x}^\top\boldsymbol{\beta}=0$" + "\n" + r"$(\hat{p}=0.5)$",
             xy=(0.3, -(b0 + b1 * 0.3) / b2),
             xytext=(1.8, -2.4),
             color=ORANGE, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-3.5, 3.5)
ax2.set_title(r"Fronteira em 2D — hiperplano $\mathbf{x}^\top\boldsymbol{\beta}=0$",
              color=ORANGE, fontsize=10, pad=10)
ax2.legend(fontsize=9, facecolor="#161b22", edgecolor=EDGE, labelcolor=TEXT)
style_ax(ax2)

plt.savefig("01_logistica_fronteira.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("ok 01_logistica_fronteira.png")


# ── Gráfico 2: OvR vs Softmax ─────────────────────────────────────────────────
np.random.seed(0)
X3, y3 = make_blobs(n_samples=360, centers=3, cluster_std=1.1, random_state=7)
colors3 = [BLUE, GREEN, ORANGE]
labels3 = ["Classe A", "Classe B", "Classe C"]

clf_ovr  = LogisticRegression(multi_class="ovr",         max_iter=1000).fit(X3, y3)
clf_soft = LogisticRegression(multi_class="multinomial", max_iter=1000).fit(X3, y3)

x_min, x_max = X3[:, 0].min() - 1, X3[:, 0].max() + 1
y_min, y_max = X3[:, 1].min() - 1, X3[:, 1].max() + 1
xx3, yy3 = np.meshgrid(np.linspace(x_min, x_max, 300),
                        np.linspace(y_min, y_max, 300))
grid3 = np.c_[xx3.ravel(), yy3.ravel()]

# cores de fundo suaves por classe
bg_cmap = ListedColormap(["#0d2540", "#0d3020", "#3a1a00"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
fig.subplots_adjust(wspace=0.32)

for ax, clf, title, title_color in [
    (ax1, clf_ovr,  "One-vs-Rest (OvR)\n$k$ classificadores binários independentes", BLUE),
    (ax2, clf_soft, "Softmax (Multinomial)\notimização conjunta das $k$ classes",   GREEN),
]:
    Z3 = clf.predict(grid3).reshape(xx3.shape)
    ax.contourf(xx3, yy3, Z3, cmap=bg_cmap, alpha=0.40)
    ax.contour(xx3, yy3, Z3, colors=[EDGE], linewidths=1.8, alpha=0.85)

    for k, (c, lbl) in enumerate(zip(colors3, labels3)):
        mask = y3 == k
        ax.scatter(X3[mask, 0], X3[mask, 1], color=c, s=22, alpha=0.80, label=lbl)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, color=title_color, fontsize=10, pad=10)
    ax.legend(fontsize=9, facecolor="#161b22", edgecolor=EDGE, labelcolor=TEXT)
    style_ax(ax)

plt.savefig("01_logistica_ovr_softmax.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("ok 01_logistica_ovr_softmax.png")


# ── Gráfico 3: Pesos de classe ────────────────────────────────────────────────
np.random.seed(7)
n_maj, n_min = 270, 30
X_maj = np.random.randn(n_maj, 2) * np.array([1.0, 0.9]) + np.array([-1.5, 0.2])
X_min = np.random.randn(n_min, 2) * np.array([0.8, 0.8]) + np.array([1.8, -0.1])
X4 = np.vstack([X_maj, X_min])
y4 = np.array([0] * n_maj + [1] * n_min)

clf_unw = LogisticRegression(class_weight=None,       max_iter=1000).fit(X4, y4)
clf_w   = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X4, y4)

x4_min, x4_max = -5.2, 5.2
y4_min, y4_max = -4.0, 4.0
xx4, yy4 = np.meshgrid(np.linspace(x4_min, x4_max, 250),
                        np.linspace(y4_min, y4_max, 250))
grid4 = np.c_[xx4.ravel(), yy4.ravel()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
fig.subplots_adjust(wspace=0.32)

for ax, clf, title, title_color in [
    (ax1, clf_unw, f"Sem pesos — fronteira enviesada\npara a classe majoritária (n={n_maj})", RED),
    (ax2, clf_w,   f"Com class_weight='balanced' — fronteira\ncorrigida para a minoria (n={n_min})", GREEN),
]:
    Z4 = clf.predict_proba(grid4)[:, 1].reshape(xx4.shape)
    ax.contourf(xx4, yy4, Z4, levels=25, cmap="RdYlGn", alpha=0.22, vmin=0, vmax=1)
    ax.contour(xx4, yy4, Z4, levels=[0.5], colors=[ORANGE], linewidths=2.3)

    ax.scatter(X4[y4 == 0, 0], X4[y4 == 0, 1],
               color=BLUE, s=16, alpha=0.60, label=f"$y=0$ (n={n_maj})")
    ax.scatter(X4[y4 == 1, 0], X4[y4 == 1, 1],
               color="#ff7b72", s=50, alpha=0.90, marker="*",
               label=f"$y=1$ (n={n_min})")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(x4_min, x4_max)
    ax.set_ylim(y4_min, y4_max)
    ax.set_title(title, color=title_color, fontsize=10, pad=10)
    ax.legend(fontsize=9, facecolor="#161b22", edgecolor=EDGE, labelcolor=TEXT)
    style_ax(ax)

plt.savefig("01_logistica_pesos_classe.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("ok 01_logistica_pesos_classe.png")
