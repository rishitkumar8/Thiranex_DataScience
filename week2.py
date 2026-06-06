import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

sns.set_style("whitegrid")
os.makedirs("week2", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────

df = pd.read_csv("netflix_titles.csv")
print(f"Loaded: {df.shape}")

# ── Clean ──────────────────────────────────────────────────────────────────────

# Rescue misplaced duration values sitting in the rating column
misplaced = df["rating"].str.match(r"^\d+ min$", na=False)
needs_rescue = misplaced & df["duration"].isna()
df.loc[needs_rescue, "duration"] = df.loc[needs_rescue, "rating"]
df.loc[misplaced, "rating"] = pd.NA

df = df.dropna(subset=["show_id", "title", "type"])
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

# ── Target: Audience Category ──────────────────────────────────────────────────

RATING_MAP = {
    "TV-Y": "Kids",    "TV-Y7": "Kids",    "TV-Y7-FV": "Kids",
    "TV-G": "Kids",    "G": "Kids",
    "TV-PG": "Teen",   "PG": "Teen",       "PG-13": "Teen",    "TV-14": "Teen",
    "TV-MA": "Adult",  "R": "Adult",       "NC-17": "Adult",
}
df["audience"] = df["rating"].map(RATING_MAP)
df = df.dropna(subset=["audience"])
print(f"After audience mapping: {df.shape}")
print(df["audience"].value_counts())

# ── Feature Engineering ────────────────────────────────────────────────────────

df["is_movie"]       = (df["type"] == "Movie").astype(int)
df["duration_value"] = df["duration"].str.extract(r"(\d+)").astype(float)
df["is_minutes"]     = df["duration"].str.contains("min", na=False).astype(int)
df["release_year"]   = df["release_year"].fillna(df["release_year"].median())
df["num_genres"]     = df["listed_in"].fillna("").apply(lambda x: len(x.split(",")))
df["date_added"]     = pd.to_datetime(df["date_added"], errors="coerce")
df["month_added"]    = df["date_added"].dt.month.fillna(0).astype(int)
df["has_director"]   = (df["director"].fillna("Unknown") != "Unknown").astype(int)

top_countries = df["country"].value_counts().index[:10].tolist()
df["country_clean"] = df["country"].apply(lambda x: x if x in top_countries else "Other")
country_dummies = pd.get_dummies(df["country_clean"], prefix="country")

BASE_FEATURES = [
    "is_movie", "duration_value", "is_minutes",
    "release_year", "num_genres", "month_added", "has_director",
]
X = pd.concat([df[BASE_FEATURES].fillna(0), country_dummies], axis=1)
y = df["audience"]

print(f"\nFeature matrix: {X.shape}")

# ── Plot 0: Class Distribution ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
order = ["Kids", "Teen", "Adult"]
counts = y.value_counts().reindex(order)
bars = ax.bar(order, counts.values, color=["#2ecc71", "#3498db", "#e74c3c"],
              edgecolor="white", width=0.5)
ax.set_title("Audience Category Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Audience Category")
ax.set_ylabel("Number of Titles")
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 20,
            str(val), ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("week2/0_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week2/0_class_distribution.png")

# ── Train / Test Split ─────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Plot 0b: Correlation Heatmap ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
corr = X[BASE_FEATURES].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=ax, linewidths=0.5, square=True)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("week2/0b_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week2/0b_correlation_heatmap.png")

# ── Hyperparameter Tuning (Random Forest) ─────────────────────────────────────

print("\nRunning GridSearchCV for Random Forest (this may take a moment)...")
param_grid = {
    "n_estimators": [100, 200],
    "max_depth":    [10, 20, None],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best params: {best_params}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

# ── Train Models ───────────────────────────────────────────────────────────────

model_defs = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_sc, X_test_sc,
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=10, random_state=42),
        X_train, X_test,
    ),
    "Random Forest": (
        RandomForestClassifier(**best_params, random_state=42),
        X_train, X_test,
    ),
}

results = {}
for name, (model, X_tr, X_te) in model_defs.items():
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    acc   = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "acc": acc, "X_te": X_te}
    print(f"\n{name}  Accuracy = {acc:.4f}")
    print(classification_report(y_test, preds))

# ── Cross Validation ───────────────────────────────────────────────────────────

print("\n=== 5-Fold Cross Validation ===")
cv_results = {}
for name, (model, X_tr, _) in model_defs.items():
    X_cv = scaler.transform(X) if name == "Logistic Regression" else X
    scores = cross_val_score(model, X_cv, y, cv=5, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:22s}  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}  Scores: {np.round(scores, 4)}")

# ── Plot 1: Confusion Matrices ─────────────────────────────────────────────────

CLASSES = ["Adult", "Kids", "Teen"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices - Audience Category Prediction", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["preds"], labels=CLASSES)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES, linewidths=0.5)
    ax.set_title(f"{name}\nAcc: {res['acc']:.3f}", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("week2/1_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week2/1_confusion_matrices.png")

# ── Plot 2: ROC Curves (One-vs-Rest) ──────────────────────────────────────────

y_test_bin  = label_binarize(y_test, classes=CLASSES)
CLASS_COLORS = {"Adult": "#e74c3c", "Kids": "#2ecc71", "Teen": "#3498db"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("ROC Curves (One-vs-Rest) - Audience Category Prediction",
             fontsize=14, fontweight="bold")

macro_aucs = {}
for ax, (name, res) in zip(axes, results.items()):
    y_score  = res["model"].predict_proba(res["X_te"])
    auc_vals = []
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        auc_vals.append(roc_auc)
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls], lw=2,
                label=f"{cls} (AUC={roc_auc:.2f})")
    macro_aucs[name] = np.mean(auc_vals)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(name, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("week2/2_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week2/2_roc_curves.png")

# ── Plot 3: Feature Importance (% of total) ────────────────────────────────────

raw_importances = pd.Series(
    results["Random Forest"]["model"].feature_importances_, index=X.columns
).nlargest(15).sort_values()
importance_pct = (raw_importances / raw_importances.sum()) * 100

fig, ax = plt.subplots(figsize=(10, 7))
bars = importance_pct.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
ax.set_title("Top 15 Feature Importances - Random Forest (% of total)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance (%)")
ax.spines[["top", "right"]].set_visible(False)
for i, val in enumerate(importance_pct.values):
    ax.text(val + 0.2, i, f"{val:.1f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("week2/3_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week2/3_feature_importance.png")

# ── Plot 4: Model Comparison (Accuracy + CV + AUC) ────────────────────────────

summary = pd.DataFrame({
    "Model":      list(results.keys()),
    "Accuracy":   [res["acc"] for res in results.values()],
    "CV Mean":    [cv_results[n].mean() for n in results],
    "CV Std":     [cv_results[n].std()  for n in results],
    "Macro AUC":  [macro_aucs[n] for n in results],
})
print("\n=== Model Summary ===")
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
PALETTE = ["#e74c3c", "#f39c12", "#2ecc71"]

for ax, metric in zip(axes, ["Accuracy", "CV Mean", "Macro AUC"]):
    bars = ax.bar(summary["Model"], summary[metric], color=PALETTE,
                  edgecolor="white", width=0.5)
    if metric == "CV Mean":
        ax.errorbar(range(len(summary)), summary["CV Mean"],
                    yerr=summary["CV Std"], fmt="none", color="black",
                    capsize=6, lw=2)
    ax.set_ylim(0, 1.1)
    ax.set_title(metric, fontsize=12)
    ax.set_ylabel("Score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=10)
    for bar, val in zip(bars, summary[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("week2/4_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week2/4_model_comparison.png")

# ── Plot 5: Cross Validation Score Distribution ────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
cv_df = pd.DataFrame(cv_results)
cv_df.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.7)
ax.set_title("5-Fold Cross Validation Accuracy per Fold", fontsize=13, fontweight="bold")
ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy")
ax.set_xticklabels([f"Fold {i+1}" for i in range(5)], rotation=0)
ax.set_ylim(0, 1)
ax.legend(title="Model", loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("week2/5_cross_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week2/5_cross_validation.png")

# ── Business Insights ─────────────────────────────────────────────────────────

print("\n=== Business Insights ===")

avg_year = df.groupby("audience")["release_year"].mean().reindex(["Kids", "Teen", "Adult"])
print("\nAverage Release Year by Audience:")
for cat, yr in avg_year.items():
    print(f"  {cat:6s}: {yr:.0f}")

avg_duration = df.groupby("audience")["duration_value"].mean().reindex(["Kids", "Teen", "Adult"])
print("\nAverage Duration by Audience:")
for cat, dur in avg_duration.items():
    unit = "mins" if cat != "Kids" else "mins/seasons"
    print(f"  {cat:6s}: {dur:.1f} {unit}")

type_pct = df.groupby("audience")["type"].value_counts(normalize=True).mul(100).round(1)
print("\nContent Type Split (%) by Audience:")
print(type_pct.to_string())

genre_counts = df.groupby("audience")["num_genres"].mean().reindex(["Kids", "Teen", "Adult"])
print("\nAverage Number of Genres Listed:")
for cat, g in genre_counts.items():
    print(f"  {cat:6s}: {g:.2f}")

recent = df[df["release_year"] >= 2018]
recent_dist = recent["audience"].value_counts(normalize=True).mul(100).round(1)
print("\nAudience split in recent content (2018+):")
print(recent_dist.to_string())

print("\nAll outputs saved to week2/")
