import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ASSETS = "01_machine_learning/supervisionado/notas/assets"

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Figura 1: árvore de decisão renderizada (max_depth=3) ─────────────────────
tree_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_viz.fit(X_tr, y_tr)

fig, ax = plt.subplots(figsize=(18, 7))
plot_tree(
    tree_viz,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    fontsize=9,
    ax=ax,
)
ax.set_title(
    "Estrutura da árvore de decisão (max_depth=3) — dataset Breast Cancer",
    fontsize=12,
    pad=12,
)
plt.tight_layout()
plt.savefig(f"{ASSETS}/04_arvores_estrutura.png", dpi=150, bbox_inches="tight")
plt.close()
print("04_arvores_estrutura.png gerado")

# ── Figura 2: curva viés-variância (AUC treino vs teste por profundidade) ──────
depths = list(range(1, 15))
auc_train, auc_test = [], []

for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_tr, y_tr)
    auc_train.append(roc_auc_score(y_tr, t.predict_proba(X_tr)[:, 1]))
    auc_test.append(roc_auc_score(y_te, t.predict_proba(X_te)[:, 1]))

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.axvspan(0.5, 2.5, alpha=0.07, color="#FF9800", zorder=0)
ax.axvspan(6.5, 14.5, alpha=0.07, color="#F44336", zorder=0)

ax.plot(depths, auc_train, marker="o", markersize=5, linewidth=2,
        color="#1565C0", label="Treino")
ax.plot(depths, auc_test,  marker="o", markersize=5, linewidth=2,
        color="#C62828", label="Teste")

ax.annotate("underfitting\n(alto viés)", xy=(1.5, 0.835), ha="center",
            fontsize=9, color="#E65100")
ax.annotate("overfitting\n(alta variância)", xy=(10.5, 0.835), ha="center",
            fontsize=9, color="#B71C1C")

best_depth = depths[np.argmax(auc_test)]
ax.axvline(best_depth, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.annotate(f"ótimo ≈ depth {best_depth}", xy=(best_depth + 0.15, 0.958),
            fontsize=8.5, color="gray")

ax.set_xlabel("Profundidade máxima (max_depth)", fontsize=11)
ax.set_ylabel("AUC", fontsize=11)
ax.set_xticks(depths)
ax.set_ylim(0.82, 1.01)
ax.legend(fontsize=10)
ax.set_title("Trade-off viés-variância por profundidade da árvore", fontsize=12, pad=10)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{ASSETS}/04_arvores_bias_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print("04_arvores_bias_variance.png gerado")
