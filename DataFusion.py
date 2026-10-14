"""
Multi-modal fusion strategies.
Includes clear graphical comparison of fusion methods.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)

# ------------------------------------------------------------
# 1. Generate complementary multi-modal data
# ------------------------------------------------------------
n = 1500
thickness = rng.uniform(0.5, 2.0, n)
density   = rng.uniform(1.0, 1.8, n)
composition = rng.uniform(0, 1, n)

spec = np.column_stack([
    0.5 + 0.4*composition + 0.1*thickness + rng.normal(0, 0.05, n),
    0.3 + 0.5*composition + rng.normal(0, 0.04, n),
    0.2 + 0.2*thickness + rng.normal(0, 0.03, n)
])
elec = np.column_stack([
    1.0 + 0.3*density - 0.1*thickness + rng.normal(0, 0.04, n),
    0.5 + 0.2*density + rng.normal(0, 0.03, n)
])
thermal = np.column_stack([
    40 + 8*thickness + 3*density + rng.normal(0, 1.5, n),
    3 + 1.5*thickness + rng.normal(0, 0.4, n)
])
process = np.column_stack([
    rng.uniform(30, 70, n),
    rng.uniform(2, 8, n),
    rng.uniform(10, 25, n)
])

y = np.column_stack([thickness, density, composition])
modalities = {
    "spectral": spec,
    "electrical": elec,
    "thermal": thermal,
    "process": process
}

# ------------------------------------------------------------
# 2. Fusion helpers
# ------------------------------------------------------------
def early_fusion(mods):
    return np.hstack(list(mods.values()))

def intermediate_fusion(mods):
    reduced = []
    for X in mods.values():
        n_comp = min(3, X.shape[1])
        reduced.append(PCA(n_components=n_comp).fit_transform(X))
    return np.hstack(reduced)

def attention_weights(mods, y):
    X_all = early_fusion(mods)
    pipe = make_pipeline(StandardScaler(),
                         MultiOutputRegressor(GradientBoostingRegressor(n_estimators=80, random_state=0)))
    pipe.fit(X_all, y)
    base_r2 = r2_score(y, pipe.predict(X_all), multioutput="raw_values").mean()

    weights = {}
    start = 0
    for name, X in mods.items():
        end = start + X.shape[1]
        X_perm = X_all.copy()
        X_perm[:, start:end] = rng.permutation(X_perm[:, start:end])
        perm_r2 = r2_score(y, pipe.predict(X_perm), multioutput="raw_values").mean()
        weights[name] = max(0.0, base_r2 - perm_r2)
        start = end
    total = sum(weights.values()) + 1e-8
    return {k: v / total for k, v in weights.items()}

# ------------------------------------------------------------
# 3. Evaluate strategies
# ------------------------------------------------------------
def eval_r2(X, y):
    pipe = make_pipeline(StandardScaler(),
                         MultiOutputRegressor(GradientBoostingRegressor(n_estimators=120, random_state=0)))
    scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    return scores.mean(), scores.std()

results = {}
results["Early fusion"] = eval_r2(early_fusion(modalities), y)
results["Intermediate (PCA)"] = eval_r2(intermediate_fusion(modalities), y)

# Single-modality baselines
for name, X in modalities.items():
    results[f"Only {name}"] = eval_r2(X, y)

weights = attention_weights(modalities, y)
print("Modality importance (attention-style):", {k: f"{v:.2f}" for k, v in weights.items()})

# Weighted fusion
weighted_feats = []
for name, X in modalities.items():
    weighted_feats.append(X * weights[name])
results["Attention-weighted"] = eval_r2(np.hstack(weighted_feats), y)

print("\n=== Cross-validated R² ===")
for name, (mean, std) in results.items():
    print(f"{name:25s}  {mean:.3f} ± {std:.3f}")

# ------------------------------------------------------------
# 4. Graphical outputs
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Bar chart of fusion performance
names = list(results.keys())
means = [results[n][0] for n in names]
stds  = [results[n][1] for n in names]
colors = ["C0" if "Only" not in n else "C1" for n in names]

axes[0, 0].barh(names, means, xerr=stds, color=colors, alpha=0.85)
axes[0, 0].set_xlabel("Mean R² (5-fold CV)")
axes[0, 0].set_title("Fusion Strategy Comparison")
axes[0, 0].axvline(0, color="k", lw=0.5)

# Modality importance
mod_names = list(weights.keys())
mod_vals  = list(weights.values())
axes[0, 1].bar(mod_names, mod_vals, color=["C2", "C3", "C4", "C5"])
axes[0, 1].set_ylabel("Relative Importance")
axes[0, 1].set_title("Attention-style Modality Weights")
axes[0, 1].set_ylim(0, 1)

# Example: predicted vs true for best fusion
X_best = early_fusion(modalities)
pipe = make_pipeline(StandardScaler(),
                     MultiOutputRegressor(GradientBoostingRegressor(n_estimators=150, random_state=0)))
pipe.fit(X_best, y)
y_pred = pipe.predict(X_best)

for i, name in enumerate(["Thickness", "Density", "Composition"]):
    axes[1, 0].scatter(y[:, i], y_pred[:, i], s=12, alpha=0.5, label=name)
axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1)
axes[1, 0].set_xlabel("True")
axes[1, 0].set_ylabel("Predicted")
axes[1, 0].set_title("Early Fusion: Predicted vs True")
axes[1, 0].legend()

# Residual histograms
residuals = y - y_pred
axes[1, 1].hist(residuals[:, 0], bins=30, alpha=0.7, label="Thickness")
axes[1, 1].hist(residuals[:, 1], bins=30, alpha=0.7, label="Density")
axes[1, 1].hist(residuals[:, 2], bins=30, alpha=0.7, label="Composition")
axes[1, 1].set_xlabel("Residual")
axes[1, 1].set_title("Residuals by Property")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("demo3_multimodal_fusion.png", dpi=140, bbox_inches="tight")
print("\nSaved: demo3_multimodal_fusion.png")
plt.show()