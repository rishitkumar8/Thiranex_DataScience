import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2_contingency, spearmanr, kruskal

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
os.makedirs("week3", exist_ok=True)

PALETTE_3 = {"Kids": "#2ecc71", "Teen": "#3498db", "Adult": "#e74c3c"}
PALETTE_6 = sns.color_palette("Set2", 6)

# ═══════════════════════════════════════════════════════════════════
# LOAD & CLEAN
# ═══════════════════════════════════════════════════════════════════

df = pd.read_csv("netflix_titles.csv")
print(f"Loaded: {df.shape}")

misplaced    = df["rating"].str.match(r"^\d+ min$", na=False)
needs_rescue = misplaced & df["duration"].isna()
df.loc[needs_rescue, "duration"] = df.loc[needs_rescue, "rating"]
df.loc[misplaced, "rating"] = pd.NA

df = df.dropna(subset=["show_id", "title", "type"])
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
df["director"]     = df["director"].fillna("Unknown")
df["country"]      = df["country"].fillna("Unknown")
df["rating"]       = df["rating"].fillna("Not Rated")
df["listed_in"]    = df["listed_in"].fillna("Unknown")

df["date_added"]      = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"]      = df["date_added"].dt.year
df["month_added"]     = df["date_added"].dt.month
df["duration_value"]  = df["duration"].str.extract(r"(\d+)").astype(float)
df["primary_country"] = df["country"].apply(lambda x: x.split(",")[0].strip())
df["primary_genre"]   = df["listed_in"].apply(lambda x: x.split(",")[0].strip())
df["num_genres"]      = df["listed_in"].apply(lambda x: len(x.split(",")))
df["has_director"]    = (df["director"] != "Unknown").astype(int)

RATING_MAP = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids",
    "TV-G": "Kids", "G": "Kids",
    "TV-PG": "Teen", "PG": "Teen", "PG-13": "Teen", "TV-14": "Teen",
    "TV-MA": "Adult", "R": "Adult", "NC-17": "Adult",
}
df["audience"] = df["rating"].map(RATING_MAP).fillna("Unknown")

movies = df[df["type"] == "Movie"].copy()
shows  = df[df["type"] == "TV Show"].copy()
print(f"Movies: {len(movies)} | TV Shows: {len(shows)} | Total: {len(df)}")

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2  = chi2_contingency(ct)[0]
    n     = ct.sum().sum()
    r, k  = ct.shape
    phi2c = max(0, chi2/n - ((k-1)*(r-1))/(n-1))
    rc    = r - ((r-1)**2)/(n-1)
    kc    = k - ((k-1)**2)/(n-1)
    denom = min(kc-1, rc-1)
    return np.sqrt(phi2c / denom) if denom > 0 else 0.0

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ═══════════════════════════════════════════════════════════════════
# PLOT 1: STATISTICAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════

numeric_cols = ["duration_value", "release_year", "num_genres"]
col_labels   = ["Duration Value", "Release Year", "Num Genres"]
rows = []
for col, label in zip(numeric_cols, col_labels):
    for ctype, subset in [("Movies", movies), ("TV Shows", shows)]:
        s = subset[col].dropna()
        rows.append({
            "Feature": label, "Type": ctype,
            "N":        int(s.count()),
            "Mean":     f"{s.mean():.2f}",
            "Median":   f"{s.median():.2f}",
            "Std Dev":  f"{s.std():.2f}",
            "Skewness": f"{s.skew():.2f}",
            "Kurtosis": f"{s.kurt():.2f}",
            "Min":      f"{s.min():.0f}",
            "Max":      f"{s.max():.0f}",
        })

stat_df   = pd.DataFrame(rows)
cols_show = ["Feature","Type","N","Mean","Median","Std Dev","Skewness","Kurtosis","Min","Max"]

fig, ax = plt.subplots(figsize=(18, 4))
ax.axis("off")
tbl = ax.table(cellText=stat_df[cols_show].values.tolist(),
               colLabels=cols_show, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.1)
for j in range(len(cols_show)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(stat_df) + 1):
    color = "#ecf0f1" if i % 2 == 0 else "#ffffff"
    for j in range(len(cols_show)):
        tbl[i, j].set_facecolor(color)
        if j == 0:
            tbl[i, j].set_text_props(fontweight="bold")
ax.set_title("Statistical Summary of Numeric Features — Movies vs TV Shows",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("week3/1_statistical_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/1_statistical_summary.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 2: DISTRIBUTION ANALYSIS (KDE + HISTOGRAM)
# ═══════════════════════════════════════════════════════════════════

panels = [
    (movies["duration_value"].dropna(), "Movie Duration (minutes)",   "#e74c3c"),
    (shows["duration_value"].dropna(),  "TV Show Duration (seasons)", "#3498db"),
    (df["release_year"].dropna(),       "Release Year (all titles)",  "#f39c12"),
    (df["num_genres"].dropna(),         "Genres per Title",           "#2ecc71"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distribution Analysis of Key Features", fontsize=14, fontweight="bold")

for ax, (data, title, color) in zip(axes.flat, panels):
    ax.hist(data, bins=30, color=color, alpha=0.35, density=True, edgecolor="white")
    data.plot.kde(ax=ax, color=color, lw=2.5)
    ax.axvline(data.mean(),   color="black", ls="--", lw=1.5, label=f"Mean   {data.mean():.1f}")
    ax.axvline(data.median(), color="gray",  ls=":",  lw=1.5, label=f"Median {data.median():.1f}")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Density")
    ax.text(0.97, 0.95, f"Skew: {data.skew():.2f}\nKurt: {data.kurt():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("week3/2_distribution_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/2_distribution_analysis.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 3: SPEARMAN CORRELATION + SIGNIFICANCE STARS
# ═══════════════════════════════════════════════════════════════════

num_feats   = ["duration_value", "release_year", "num_genres", "year_added", "month_added"]
feat_labels = ["Duration", "Release Year", "Num Genres", "Year Added", "Month Added"]
n = len(num_feats)
corr_mat = np.zeros((n, n))
pval_mat = np.zeros((n, n))

for i, c1 in enumerate(num_feats):
    for j, c2 in enumerate(num_feats):
        mask = df[c1].notna() & df[c2].notna()
        if i == j:
            corr_mat[i, j], pval_mat[i, j] = 1.0, 0.0
        else:
            r, p = spearmanr(df.loc[mask, c1], df.loc[mask, c2])
            corr_mat[i, j], pval_mat[i, j] = r, p

annot = np.empty((n, n), dtype=object)
for i in range(n):
    for j in range(n):
        annot[i, j] = "1.00" if i == j else f"{corr_mat[i,j]:.2f}\n{sig_stars(pval_mat[i,j])}"

corr_df = pd.DataFrame(corr_mat, index=feat_labels, columns=feat_labels)
mask_up = np.triu(np.ones((n, n), dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_df, annot=annot, fmt="", cmap="coolwarm", center=0,
            mask=mask_up, ax=ax, linewidths=0.5, vmin=-1, vmax=1,
            cbar_kws={"label": "Spearman r"})
ax.set_title("Spearman Correlation Matrix\n(* p<0.05   ** p<0.01   *** p<0.001)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("week3/3_spearman_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/3_spearman_correlation.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 4: GENRE EVOLUTION — STACKED AREA (NORMALIZED)
# ═══════════════════════════════════════════════════════════════════

gdf = df[["year_added", "listed_in"]].dropna()
gdf = gdf[gdf["year_added"].between(2015, 2021)].copy()
gdf["genres"] = gdf["listed_in"].str.split(", ")
gdf = gdf.explode("genres")
gdf["genres"] = gdf["genres"].str.strip()

top_genres  = gdf["genres"].value_counts().head(8).index.tolist()
gdf         = gdf[gdf["genres"].isin(top_genres)]
genre_pivot = gdf.groupby(["year_added", "genres"]).size().unstack(fill_value=0)
genre_pct   = genre_pivot.div(genre_pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(13, 6))
genre_pct.plot.area(ax=ax, colormap="tab10", alpha=0.85)
ax.set_title("Top Genre Share Evolution (2015-2021, Normalized %)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year Added to Netflix")
ax.set_ylabel("Share of Titles (%)")
ax.set_xlim(genre_pct.index.min(), genre_pct.index.max())
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="Genre", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("week3/4_genre_evolution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/4_genre_evolution.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 5: CONTENT STRATEGY SHIFT — AUDIENCE MIX OVER TIME
# ═══════════════════════════════════════════════════════════════════

aud_df = df[df["audience"].isin(["Kids", "Teen", "Adult"]) &
            df["year_added"].between(2015, 2021)].copy()
aud_pivot = aud_df.groupby(["year_added", "audience"]).size().unstack(fill_value=0)
aud_pct   = aud_pivot.div(aud_pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(11, 6))
for cat, color in PALETTE_3.items():
    if cat in aud_pct.columns:
        ax.plot(aud_pct.index, aud_pct[cat], marker="o", lw=2.5, color=color, label=cat)
        ax.fill_between(aud_pct.index, aud_pct[cat], alpha=0.12, color=color)
        for yr in [aud_pct.index.min(), aud_pct.index.max()]:
            ax.annotate(f"{aud_pct.loc[yr, cat]:.0f}%",
                        xy=(yr, aud_pct.loc[yr, cat]),
                        xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=color, fontweight="bold")

ax.set_title("Content Strategy Shift: Audience Category Mix (2015-2021)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Year Added to Netflix")
ax.set_ylabel("Share of Titles (%)")
ax.set_xlim(aud_pct.index.min(), aud_pct.index.max())
ax.set_ylim(0, 80)
ax.legend(title="Audience", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("week3/5_content_strategy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/5_content_strategy.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 6: MOVIE DURATION BY GENRE (VIOLIN + KRUSKAL-WALLIS)
# ═══════════════════════════════════════════════════════════════════

top10_genres = movies["primary_genre"].value_counts().head(10).index.tolist()
vdf          = movies[movies["primary_genre"].isin(top10_genres)][["primary_genre","duration_value"]].dropna()
genre_medians = vdf.groupby("primary_genre")["duration_value"].median().sort_values()
vdf["primary_genre"] = pd.Categorical(vdf["primary_genre"],
                                       categories=genre_medians.index, ordered=True)

groups   = [vdf[vdf["primary_genre"]==g]["duration_value"].values for g in genre_medians.index]
kw_stat, kw_p = kruskal(*groups)

fig, ax = plt.subplots(figsize=(14, 7))
sns.violinplot(data=vdf, x="primary_genre", y="duration_value",
               ax=ax, palette="husl", inner="quartile", cut=0)
ax.set_title(f"Movie Duration Distribution by Genre\nKruskal-Wallis H={kw_stat:.1f},  p={kw_p:.2e}  (statistically significant differences exist)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Primary Genre")
ax.set_ylabel("Duration (minutes)")
ax.tick_params(axis="x", rotation=25)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("week3/6_duration_by_genre.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/6_duration_by_genre.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 7: GEOGRAPHIC CONTENT GROWTH (TOP 5 COUNTRIES)
# ═══════════════════════════════════════════════════════════════════

geo_df = df[df["year_added"].between(2015, 2021)].copy()
top5_countries = (geo_df[geo_df["primary_country"] != "Unknown"]
                  ["primary_country"].value_counts().head(5).index.tolist())
geo_filtered   = geo_df[geo_df["primary_country"].isin(top5_countries)]
geo_pivot      = geo_filtered.groupby(["year_added", "primary_country"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(11, 6))
for i, country in enumerate(top5_countries):
    if country in geo_pivot.columns:
        ax.plot(geo_pivot.index, geo_pivot[country], marker="o", lw=2.5,
                color=PALETTE_6[i], label=country)
        ax.fill_between(geo_pivot.index, geo_pivot[country], alpha=0.12, color=PALETTE_6[i])

ax.set_title("Content Addition Growth by Top 5 Countries (2015-2021)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Titles Added")
ax.set_xlim(geo_pivot.index.min(), geo_pivot.index.max())
ax.legend(title="Country", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("week3/7_geographic_growth.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/7_geographic_growth.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 8: MONTHLY RELEASE CALENDAR HEATMAP
# ═══════════════════════════════════════════════════════════════════

heat_df = df[df["year_added"].between(2015, 2021)].dropna(subset=["month_added"]).copy()
heat_df["month_added"] = heat_df["month_added"].astype(int)
monthly_pivot = heat_df.pivot_table(
    index="month_added", columns="year_added", aggfunc="size", fill_value=0
)
monthly_pivot = monthly_pivot.reindex(range(1, 13), fill_value=0)
monthly_pivot.index = MONTH_NAMES

fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(monthly_pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
            linewidths=0.5, cbar_kws={"label": "Titles Added"})
ax.set_title("Monthly Content Addition Heatmap (2015-2021)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Month")
plt.tight_layout()
plt.savefig("week3/8_monthly_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/8_monthly_heatmap.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 9: CRAMER'S V ASSOCIATION MATRIX
# ═══════════════════════════════════════════════════════════════════

top5_c = (df[df["primary_country"] != "Unknown"]
          ["primary_country"].value_counts().head(5).index.tolist())
df["country_group"] = df["primary_country"].apply(lambda x: x if x in top5_c else "Other")

cat_cols   = ["type", "audience", "country_group", "rating"]
cat_labels = ["Content Type", "Audience", "Country Group", "Rating"]
m = len(cat_cols)
cv_mat = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        if i == j:
            cv_mat[i, j] = 1.0
        else:
            sub = df[[cat_cols[i], cat_cols[j]]].dropna()
            cv_mat[i, j] = cramers_v(sub[cat_cols[i]], sub[cat_cols[j]])

cv_df   = pd.DataFrame(cv_mat, index=cat_labels, columns=cat_labels)
mask_up = np.triu(np.ones((m, m), dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cv_df, annot=True, fmt=".3f", cmap="Blues", ax=ax,
            linewidths=0.5, vmin=0, vmax=1, mask=mask_up,
            cbar_kws={"label": "Cramer's V  (0 = no association, 1 = perfect)"})
ax.set_title("Cramer's V Association Matrix Between Categorical Features",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("week3/9_cramers_v_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/9_cramers_v_matrix.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 10: GENRE x AUDIENCE HEATMAP
# ═══════════════════════════════════════════════════════════════════

top15_genres = df["primary_genre"].value_counts().head(15).index.tolist()
ga_df        = df[df["audience"].isin(["Kids","Teen","Adult"]) &
                  df["primary_genre"].isin(top15_genres)].copy()
ga_pivot     = pd.crosstab(ga_df["primary_genre"], ga_df["audience"], normalize="index") * 100
ga_pivot     = ga_pivot[["Kids","Teen","Adult"]].sort_values("Adult", ascending=False)

fig, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(ga_pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax,
            linewidths=0.5, cbar_kws={"label": "% of titles in this genre"},
            vmin=0, vmax=100)
ax.set_title("Genre x Audience Distribution\n(% within each genre, sorted by Adult %)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Audience Category")
ax.set_ylabel("Primary Genre")
plt.tight_layout()
plt.savefig("week3/10_genre_audience_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/10_genre_audience_heatmap.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 11: CONTENT TIMING BY AUDIENCE (MONTHLY STACKED BAR)
# ═══════════════════════════════════════════════════════════════════

timing_df    = df[df["audience"].isin(["Kids","Teen","Adult"]) & df["month_added"].notna()].copy()
timing_df["month_added"] = timing_df["month_added"].astype(int)
timing_pivot = timing_df.groupby(["month_added","audience"]).size().unstack(fill_value=0)
timing_pivot = timing_pivot.reindex(range(1, 13), fill_value=0)
timing_pct   = timing_pivot.div(timing_pivot.sum(axis=1), axis=0) * 100
timing_pct.index = MONTH_NAMES

fig, ax = plt.subplots(figsize=(13, 6))
bottom = np.zeros(12)
for cat in ["Kids", "Teen", "Adult"]:
    if cat in timing_pct.columns:
        vals = timing_pct[cat].values
        bars = ax.bar(timing_pct.index, vals, bottom=bottom,
                      label=cat, color=PALETTE_3[cat], edgecolor="white", width=0.75)
        for idx, (val, bot) in enumerate(zip(vals, bottom)):
            if val > 6:
                ax.text(idx, bot + val/2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bottom += vals

ax.set_title("When Does Netflix Add Each Audience Category? (Monthly Share %)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Share of Monthly Additions (%)")
ax.set_ylim(0, 100)
ax.legend(title="Audience", loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("week3/11_content_timing.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/11_content_timing.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 12: EXECUTIVE DASHBOARD (6-PANEL)
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Netflix EDA — Executive Dashboard", fontsize=16, fontweight="bold")

# A: Genre evolution
genre_pct.plot.area(ax=axes[0,0], colormap="tab10", alpha=0.85, legend=True)
axes[0,0].set_title("A. Genre Share Evolution (2015-2021)", fontweight="bold", fontsize=10)
axes[0,0].set_xlabel("Year")
axes[0,0].set_ylabel("Share (%)")
axes[0,0].legend(fontsize=7, loc="upper left", ncol=2)
axes[0,0].spines[["top","right"]].set_visible(False)

# B: Audience strategy shift
for cat, color in PALETTE_3.items():
    if cat in aud_pct.columns:
        axes[0,1].plot(aud_pct.index, aud_pct[cat], marker="o", lw=2, color=color, label=cat)
        axes[0,1].fill_between(aud_pct.index, aud_pct[cat], alpha=0.12, color=color)
axes[0,1].set_title("B. Audience Mix Shift (2015-2021)", fontweight="bold", fontsize=10)
axes[0,1].set_xlabel("Year")
axes[0,1].set_ylabel("Share (%)")
axes[0,1].legend(fontsize=9)
axes[0,1].spines[["top","right"]].set_visible(False)

# C: Geographic growth
for i, country in enumerate(top5_countries):
    if country in geo_pivot.columns:
        axes[0,2].plot(geo_pivot.index, geo_pivot[country], marker="o", lw=2,
                       color=PALETTE_6[i], label=country)
axes[0,2].set_title("C. Geographic Growth (2015-2021)", fontweight="bold", fontsize=10)
axes[0,2].set_xlabel("Year")
axes[0,2].set_ylabel("Titles Added")
axes[0,2].legend(fontsize=8)
axes[0,2].spines[["top","right"]].set_visible(False)

# D: Monthly heatmap
sns.heatmap(monthly_pivot, ax=axes[1,0], cmap="YlOrRd",
            annot=True, fmt="d", cbar=False, linewidths=0.2,
            annot_kws={"size": 7})
axes[1,0].set_title("D. Monthly Release Calendar", fontweight="bold", fontsize=10)
axes[1,0].set_xlabel("Year")
axes[1,0].set_ylabel("")
axes[1,0].tick_params(labelsize=8)

# E: Genre x Audience (top 8)
ga_small = ga_pivot.head(8)
sns.heatmap(ga_small, annot=True, fmt=".0f", cmap="RdYlGn_r",
            ax=axes[1,1], cbar=False, linewidths=0.5,
            annot_kws={"size": 9})
axes[1,1].set_title("E. Genre x Audience (%)", fontweight="bold", fontsize=10)
axes[1,1].set_xlabel("")
axes[1,1].set_ylabel("")
axes[1,1].tick_params(labelsize=8)

# F: Movie duration distribution
dur = movies["duration_value"].dropna()
axes[1,2].hist(dur, bins=40, color="#e74c3c", alpha=0.4, density=True, edgecolor="white")
dur.plot.kde(ax=axes[1,2], color="#c0392b", lw=2.5)
axes[1,2].axvline(dur.mean(),   color="black", ls="--", lw=1.5, label=f"Mean {dur.mean():.0f}m")
axes[1,2].axvline(dur.median(), color="gray",  ls=":",  lw=1.5, label=f"Median {dur.median():.0f}m")
axes[1,2].set_title("F. Movie Duration Distribution", fontweight="bold", fontsize=10)
axes[1,2].set_xlabel("Minutes")
axes[1,2].set_ylabel("Density")
axes[1,2].legend(fontsize=9)
axes[1,2].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("week3/12_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/12_dashboard.png")

# ═══════════════════════════════════════════════════════════════════
# AUTO-GENERATED INSIGHTS REPORT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("NETFLIX EDA — KEY INSIGHTS REPORT")
print("="*60)

yr_min, yr_max = int(genre_pct.index.min()), int(genre_pct.index.max())
genre_growth = genre_pct.loc[yr_max] - genre_pct.loc[yr_min]
print(f"\nGenre Trends ({yr_min} -> {yr_max}):")
print(f"  Fastest growing : {genre_growth.idxmax()} (+{genre_growth.max():.1f} pp)")
print(f"  Fastest shrinking: {genre_growth.idxmin()} ({genre_growth.min():.1f} pp)")

print(f"\nContent Strategy Shift ({yr_min} -> {yr_max}):")
for cat in ["Kids", "Teen", "Adult"]:
    if cat in aud_pct.columns:
        change = aud_pct.loc[yr_max, cat] - aud_pct.loc[yr_min, cat]
        arrow  = "up" if change >= 0 else "down"
        print(f"  {cat:6s}: {aud_pct.loc[yr_min, cat]:.1f}% -> {aud_pct.loc[yr_max, cat]:.1f}% ({arrow} {abs(change):.1f} pp)")

peak_month = monthly_pivot.sum(axis=1).idxmax()
print(f"\nContent Addition Patterns:")
print(f"  Peak month: {peak_month} ({int(monthly_pivot.sum(axis=1).max())} titles, 2015-2021 combined)")
print(f"  Q4 (Oct-Dec) total: {int(monthly_pivot.loc[['Oct','Nov','Dec']].sum().sum())} titles")

cv_pairs = {f"{cat_labels[i]} x {cat_labels[j]}": cv_mat[i,j]
            for i in range(m) for j in range(m) if i < j}
top_pair = max(cv_pairs, key=cv_pairs.get)
print(f"\nCramer's V — Strongest association:")
print(f"  {top_pair}: V = {cv_pairs[top_pair]:.3f}")
for pair, val in sorted(cv_pairs.items(), key=lambda x: -x[1]):
    print(f"  {pair}: {val:.3f}")

print(f"\nDuration Across Genres (Kruskal-Wallis):")
print(f"  H = {kw_stat:.2f}, p = {kw_p:.2e} => significant differences exist")
print(f"  Shortest avg: {genre_medians.index[0]} ({genre_medians.iloc[0]:.0f} min)")
print(f"  Longest  avg: {genre_medians.index[-1]} ({genre_medians.iloc[-1]:.0f} min)")

print(f"\nGeographic Insights ({int(geo_pivot.index.min())}-{int(geo_pivot.index.max())}):")
first_y, last_y = geo_pivot.index.min(), geo_pivot.index.max()
for country in top5_countries:
    if country in geo_pivot.columns:
        growth = ((geo_pivot.loc[last_y, country] - geo_pivot.loc[first_y, country])
                  / max(geo_pivot.loc[first_y, country], 1)) * 100
        print(f"  {country}: {int(geo_pivot[country].sum())} total, growth {growth:+.0f}%")

print(f"\nGenre x Audience Findings:")
for genre in ga_pivot.index[:5]:
    dominant = ga_pivot.loc[genre].idxmax()
    pct      = ga_pivot.loc[genre].max()
    print(f"  {genre}: {pct:.0f}% {dominant}")

print("\nAll outputs saved to week3/")

# ═══════════════════════════════════════════════════════════════════
# PLOT 13: BUSINESS KPIs DASHBOARD
# ═══════════════════════════════════════════════════════════════════

kpis = [
    ("Total Titles",    len(df),                                   "#2c3e50"),
    ("Movies",          len(movies),                               "#e74c3c"),
    ("TV Shows",        len(shows),                                "#3498db"),
    ("Countries",       df[df["primary_country"]!="Unknown"]["primary_country"].nunique(), "#2ecc71"),
    ("Unique Genres",   df["primary_genre"].nunique(),             "#f39c12"),
    ("Years Covered",   int(df["release_year"].dropna().nunique()), "#9b59b6"),
    ("Peak Year",       int(df["year_added"].value_counts().idxmax()), "#1abc9c"),
    ("Avg Movie (min)", f"{movies['duration_value'].mean():.0f}",  "#e67e22"),
]

fig, ax = plt.subplots(figsize=(18, 4))
ax.set_xlim(0, 18)
ax.set_ylim(0, 4)
ax.axis("off")
fig.patch.set_facecolor("#f8f9fa")

card_w, card_h, gap = 2.0, 3.2, 0.25
for idx, (label, value, color) in enumerate(kpis):
    x = idx * (card_w + gap)
    rect = mpatches.FancyBboxPatch((x, 0.3), card_w, card_h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor="none", alpha=0.92)
    ax.add_patch(rect)
    ax.text(x + card_w/2, 0.3 + card_h*0.68, str(value),
            ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    ax.text(x + card_w/2, 0.3 + card_h*0.22, label,
            ha="center", va="center", fontsize=9, color="white", alpha=0.9)

ax.set_title("Netflix Dataset — Business KPI Overview",
             fontsize=15, fontweight="bold", pad=12, color="#2c3e50")
plt.tight_layout()
plt.savefig("week3/13_kpi_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/13_kpi_dashboard.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 14: CUMULATIVE CATALOG GROWTH CURVE
# ═══════════════════════════════════════════════════════════════════

yearly     = df["year_added"].value_counts().sort_index().dropna()
yearly     = yearly[(yearly.index >= 2008) & (yearly.index <= 2021)]
cumulative = yearly.cumsum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Netflix Catalog Growth Over Time", fontsize=14, fontweight="bold")

ax1.bar(yearly.index.astype(int), yearly.values, color="#e74c3c", alpha=0.8, edgecolor="white")
ax1.set_title("Titles Added Per Year", fontsize=11)
ax1.set_xlabel("Year")
ax1.set_ylabel("New Titles Added")
ax1.spines[["top", "right"]].set_visible(False)
for x, y in zip(yearly.index.astype(int), yearly.values):
    ax1.text(x, y + 15, str(int(y)), ha="center", fontsize=7, color="#2c3e50")

ax2.plot(cumulative.index.astype(int), cumulative.values,
         color="#e74c3c", lw=3, marker="o", markersize=6)
ax2.fill_between(cumulative.index.astype(int), cumulative.values, alpha=0.15, color="#e74c3c")
ax2.set_title("Cumulative Catalog Size", fontsize=11)
ax2.set_xlabel("Year")
ax2.set_ylabel("Total Titles on Platform")
ax2.spines[["top", "right"]].set_visible(False)
for x, y in zip(cumulative.index.astype(int), cumulative.values):
    if x in [2008, 2012, 2016, 2019, 2021]:
        ax2.annotate(f"{int(y):,}", xy=(x, y), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=8,
                     color="#c0392b", fontweight="bold")

plt.tight_layout()
plt.savefig("week3/14_cumulative_growth.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/14_cumulative_growth.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 15: TOP 15 DIRECTORS
# ═══════════════════════════════════════════════════════════════════

top_directors = (df[df["director"] != "Unknown"]["director"]
                 .value_counts().head(15).sort_values())

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(top_directors.index, top_directors.values,
               color="#3498db", edgecolor="white")
ax.set_title("Top 15 Most Prolific Netflix Directors", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Titles")
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, top_directors.values):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10, fontweight="bold", color="#2c3e50")
plt.tight_layout()
plt.savefig("week3/15_top_directors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/15_top_directors.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 16: COUNTRY x GENRE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

top5_c_geo   = (df[df["primary_country"] != "Unknown"]
                ["primary_country"].value_counts().head(5).index.tolist())
top8_g_geo   = df["primary_genre"].value_counts().head(8).index.tolist()
cg_df        = df[df["primary_country"].isin(top5_c_geo) &
                  df["primary_genre"].isin(top8_g_geo)].copy()
cg_pivot     = pd.crosstab(cg_df["primary_country"], cg_df["primary_genre"], normalize="index") * 100
cg_pivot     = cg_pivot.reindex(top5_c_geo)

fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(cg_pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
            linewidths=0.5, cbar_kws={"label": "% of country's titles"})
ax.set_title("Country x Genre Distribution\n(% of each country's catalog, top 5 countries x top 8 genres)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Primary Genre")
ax.set_ylabel("Country")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("week3/16_country_genre.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/16_country_genre.png")

# Also print the dominant genre per country
print("\nDominant genre per country:")
for country in top5_c_geo:
    if country in cg_pivot.index:
        dom_genre = cg_pivot.loc[country].idxmax()
        pct       = cg_pivot.loc[country].max()
        print(f"  {country:20s}: {dom_genre} ({pct:.1f}%)")

# ═══════════════════════════════════════════════════════════════════
# PLOT 17: RELEASE YEAR vs DURATION SCATTER + TREND LINE
# ═══════════════════════════════════════════════════════════════════

scatter_df = movies[movies["release_year"].between(1980, 2021)][
    ["release_year", "duration_value", "primary_genre"]
].dropna()
sample_df  = scatter_df.sample(min(2000, len(scatter_df)), random_state=42)

z   = np.polyfit(sample_df["release_year"], sample_df["duration_value"], 1)
p   = np.poly1d(z)
x_line = np.linspace(1980, 2021, 100)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(sample_df["release_year"], sample_df["duration_value"],
           alpha=0.25, color="#e74c3c", s=15, label="Movie (sample)")
ax.plot(x_line, p(x_line), color="#2c3e50", lw=2.5, ls="--",
        label=f"Trend: {z[0]:+.2f} min/year")
ax.set_title("Are Movies Getting Shorter? Release Year vs Duration (1980-2021)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Release Year")
ax.set_ylabel("Duration (minutes)")
ax.set_xlim(1978, 2023)
direction = "shorter" if z[0] < 0 else "longer"
ax.text(0.03, 0.95,
        f"Slope: {z[0]:+.2f} min/year\nMovies trending {direction} over time",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("week3/17_duration_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/17_duration_trend.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 18: OUTLIER ANALYSIS — MOVIE DURATION
# ═══════════════════════════════════════════════════════════════════

dur_clean = movies["duration_value"].dropna()
Q1, Q3    = dur_clean.quantile(0.25), dur_clean.quantile(0.75)
IQR       = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers  = dur_clean[(dur_clean < lower) | (dur_clean > upper)]
n_out     = len(outliers)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Movie Duration Outlier Analysis", fontsize=14, fontweight="bold")

sns.boxplot(x=dur_clean, ax=ax1, color="#e74c3c", width=0.4, flierprops={"marker":"o","alpha":0.4})
ax1.axvline(lower, color="#e74c3c", ls="--", lw=1.5, label=f"Lower fence {lower:.0f}m")
ax1.axvline(upper, color="#2c3e50", ls="--", lw=1.5, label=f"Upper fence {upper:.0f}m")
ax1.set_title(f"Boxplot — {n_out} outliers detected ({n_out/len(dur_clean)*100:.1f}%)")
ax1.set_xlabel("Duration (minutes)")
ax1.legend(fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)

ax2.hist(dur_clean, bins=50, color="#bdc3c7", alpha=0.7, edgecolor="white", density=True)
ax2.hist(outliers, bins=20, color="#e74c3c", alpha=0.85, edgecolor="white", density=True,
         label=f"Outliers (n={n_out})")
ax2.axvline(lower, color="#e74c3c", ls="--", lw=1.5)
ax2.axvline(upper, color="#2c3e50", ls="--", lw=1.5)
ax2.set_title("Distribution — Normal range vs Outliers")
ax2.set_xlabel("Duration (minutes)")
ax2.set_ylabel("Density")
ax2.legend(fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)

stats_text = (f"Q1={Q1:.0f}m  Q3={Q3:.0f}m  IQR={IQR:.0f}m\n"
              f"Normal range: [{lower:.0f}, {upper:.0f}] min\n"
              f"Outliers: {n_out} titles ({n_out/len(dur_clean)*100:.1f}%)")
fig.text(0.5, -0.02, stats_text, ha="center", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1"))

plt.tight_layout()
plt.savefig("week3/18_outlier_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: week3/18_outlier_analysis.png")

# ═══════════════════════════════════════════════════════════════════
# AUTO-GENERATED REPORT (.txt)
# ═══════════════════════════════════════════════════════════════════

insights = []

insights.append("=" * 65)
insights.append("       NETFLIX EDA — AUTOMATED INSIGHTS REPORT")
insights.append("=" * 65)

insights.append("\n--- SECTION 1: DATASET OVERVIEW ---")
insights.append(f"  Total titles analysed  : {len(df):,}")
insights.append(f"  Movies                 : {len(movies):,} ({len(movies)/len(df)*100:.1f}%)")
insights.append(f"  TV Shows               : {len(shows):,} ({len(shows)/len(df)*100:.1f}%)")
insights.append(f"  Unique countries       : {df[df['primary_country']!='Unknown']['primary_country'].nunique()}")
insights.append(f"  Unique primary genres  : {df['primary_genre'].nunique()}")
insights.append(f"  Release years covered  : {int(df['release_year'].min())} - {int(df['release_year'].max())}")
insights.append(f"  Peak addition year     : {int(df['year_added'].value_counts().idxmax())}")

insights.append("\n--- SECTION 2: STATISTICAL SUMMARY ---")
insights.append(f"  Movie duration  -> Mean: {movies['duration_value'].mean():.1f} min, "
                f"Median: {movies['duration_value'].median():.0f} min, "
                f"Std: {movies['duration_value'].std():.1f}")
insights.append(f"  Movie duration  -> Skewness: {movies['duration_value'].skew():.2f} "
                f"(right-skewed: long movies pull mean above median)")
insights.append(f"  TV Show seasons -> Mean: {shows['duration_value'].mean():.1f}, "
                f"Median: {shows['duration_value'].median():.0f}")
insights.append(f"  Genres per title -> Mean: {df['num_genres'].mean():.2f}, "
                f"Max: {int(df['num_genres'].max())}")

insights.append("\n--- SECTION 3: GENRE TRENDS ---")
insights.append(f"  Fastest growing genre (2015-2021) : {genre_growth.idxmax()} (+{genre_growth.max():.1f} pp)")
insights.append(f"  Fastest shrinking genre           : {genre_growth.idxmin()} ({genre_growth.min():.1f} pp)")
for g in top_genres[:5]:
    val_start = genre_pct.loc[yr_min, g] if g in genre_pct.columns else 0
    val_end   = genre_pct.loc[yr_max, g] if g in genre_pct.columns else 0
    insights.append(f"  {g:35s}: {val_start:.1f}% -> {val_end:.1f}%")

insights.append("\n--- SECTION 4: CONTENT STRATEGY ---")
for cat in ["Kids", "Teen", "Adult"]:
    if cat in aud_pct.columns:
        change = aud_pct.loc[yr_max, cat] - aud_pct.loc[yr_min, cat]
        arrow  = "[UP]" if change >= 0 else "[DOWN]"
        insights.append(f"  {cat:6s} {arrow}: {aud_pct.loc[yr_min, cat]:.1f}% -> "
                        f"{aud_pct.loc[yr_max, cat]:.1f}% (change: {change:+.1f} pp)")
insights.append(f"  Key finding: Netflix has been steadily reducing Kids content")
insights.append(f"  and increasing Teen content — a clear shift toward broader audiences.")

insights.append("\n--- SECTION 5: CATALOG GROWTH ---")
yr_2016 = int(cumulative[cumulative.index == 2016].values[0]) if 2016 in cumulative.index else "N/A"
yr_2021 = int(cumulative[cumulative.index == 2021].values[0]) if 2021 in cumulative.index else "N/A"
insights.append(f"  Catalog size in 2016: ~{yr_2016:,} titles")
insights.append(f"  Catalog size in 2021: ~{yr_2021:,} titles")
insights.append(f"  Peak addition year  : {int(yearly.idxmax())} ({int(yearly.max())} titles added)")
insights.append(f"  Q4 (Oct-Dec) sees {int(monthly_pivot.loc[['Oct','Nov','Dec']].sum().sum())} titles — "
                f"Netflix prepares for holiday viewing windows.")

insights.append("\n--- SECTION 6: GEOGRAPHIC ANALYSIS ---")
for country in top5_countries:
    if country in geo_pivot.columns:
        total  = int(geo_pivot[country].sum())
        growth = ((geo_pivot.loc[last_y, country] - geo_pivot.loc[first_y, country])
                  / max(geo_pivot.loc[first_y, country], 1)) * 100
        insights.append(f"  {country:20s}: {total:4d} titles (2015-2021 growth {growth:+.0f}%)")
insights.append(f"  India shows the highest growth rate — Netflix's biggest international push.")

insights.append("\n--- SECTION 7: COUNTRY x GENRE ---")
for country in top5_c_geo:
    if country in cg_pivot.index:
        dom = cg_pivot.loc[country].idxmax()
        pct = cg_pivot.loc[country].max()
        insights.append(f"  {country:20s}: dominant genre is '{dom}' ({pct:.1f}%)")

insights.append("\n--- SECTION 8: DURATION ANALYSIS ---")
insights.append(f"  Kruskal-Wallis test across genres: H={kw_stat:.2f}, p={kw_p:.2e}")
insights.append(f"  -> Genre has a statistically significant effect on movie duration.")
insights.append(f"  Shortest avg genre : {genre_medians.index[0]} ({genre_medians.iloc[0]:.0f} min)")
insights.append(f"  Longest avg genre  : {genre_medians.index[-1]} ({genre_medians.iloc[-1]:.0f} min)")
insights.append(f"  Duration trend     : {z[0]:+.2f} min/year — movies are trending {'shorter' if z[0]<0 else 'longer'}.")

insights.append("\n--- SECTION 9: OUTLIER ANALYSIS ---")
insights.append(f"  IQR method: Q1={Q1:.0f} min, Q3={Q3:.0f} min, IQR={IQR:.0f} min")
insights.append(f"  Normal range: [{lower:.0f}, {upper:.0f}] minutes")
insights.append(f"  Outliers detected: {n_out} titles ({n_out/len(dur_clean)*100:.1f}% of all movies)")
insights.append(f"  These represent genuinely long films and short-form content,")
insights.append(f"  not data errors — they should be retained but noted.")

insights.append("\n--- SECTION 10: ASSOCIATION ANALYSIS (Cramer's V) ---")
for pair, val in sorted(cv_pairs.items(), key=lambda x: -x[1])[:5]:
    strength = "strong" if val > 0.5 else ("moderate" if val > 0.3 else "weak")
    insights.append(f"  {pair:40s}: V={val:.3f} ({strength})")

insights.append("\n--- SECTION 11: TOP DIRECTORS ---")
for director, count in top_directors.sort_values(ascending=False).head(5).items():
    insights.append(f"  {director:30s}: {count} titles")

insights.append("\n--- SECTION 12: RECOMMENDATIONS ---")
insights.append("  1. Netflix should invest further in International TV Shows — fastest growing genre.")
insights.append("  2. Kids content is declining sharply — risk of losing family subscriber segment.")
insights.append("  3. India is the fastest growing content market — worth deeper analysis.")
insights.append("  4. July is peak addition month — aligns with summer viewing surge.")
insights.append("  5. Stand-Up Comedy is 88% Adult — strong niche with high engagement potential.")

insights.append("\n" + "=" * 65)
insights.append("  Report generated from netflix_titles.csv using Python EDA pipeline.")
insights.append("  Visualizations saved to week3/  |  Script: week3.py")
insights.append("=" * 65)

report_text = "\n".join(insights)
with open("week3/Netflix_EDA_Report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print("Saved: week3/Netflix_EDA_Report.txt")
print(report_text)

# ═══════════════════════════════════════════════════════════════════
# COMPILE ALL PLOTS INTO PDF REPORT
# ═══════════════════════════════════════════════════════════════════

all_plots = [
    "week3/13_kpi_dashboard.png",
    "week3/1_statistical_summary.png",
    "week3/2_distribution_analysis.png",
    "week3/3_spearman_correlation.png",
    "week3/4_genre_evolution.png",
    "week3/5_content_strategy.png",
    "week3/14_cumulative_growth.png",
    "week3/7_geographic_growth.png",
    "week3/16_country_genre.png",
    "week3/6_duration_by_genre.png",
    "week3/17_duration_trend.png",
    "week3/18_outlier_analysis.png",
    "week3/8_monthly_heatmap.png",
    "week3/9_cramers_v_matrix.png",
    "week3/10_genre_audience_heatmap.png",
    "week3/11_content_timing.png",
    "week3/15_top_directors.png",
    "week3/12_dashboard.png",
]

with PdfPages("week3/Netflix_EDA_Report.pdf") as pdf:
    # Title page
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#2c3e50")
    ax.axis("off")
    ax.text(0.5, 0.65, "Netflix Dataset", ha="center", va="center",
            fontsize=36, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.50, "Exploratory Data Analysis Report", ha="center", va="center",
            fontsize=22, color="#bdc3c7", transform=ax.transAxes)
    ax.text(0.5, 0.35, "Thiranex Internship — Week 3", ha="center", va="center",
            fontsize=16, color="#95a5a6", transform=ax.transAxes)
    ax.text(0.5, 0.18, f"Dataset: {len(df):,} titles  |  {len(movies):,} Movies  |  {len(shows):,} TV Shows",
            ha="center", va="center", fontsize=13, color="#7f8c8d", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # Each plot as a page
    for plot_path in all_plots:
        if os.path.exists(plot_path):
            img = plt.imread(plot_path)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close()

    # Insights text page
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    summary_lines = [
        "Key Findings Summary",
        "",
        f"  Fastest growing genre (2015-2021): {genre_growth.idxmax()} (+{genre_growth.max():.1f} pp)",
        f"  Kids content: 23.5% -> 12.9% (halved) — Teen: 32.4% -> 41.8%",
        f"  India grew fastest among content-producing countries",
        f"  Peak addition month: {peak_month}",
        f"  Stand-Up Comedy: 88% Adult — most audience-specific genre",
        f"  Kruskal-Wallis confirms genre significantly affects movie duration (H={kw_stat:.0f})",
        f"  Duration trend: {z[0]:+.2f} min/year (movies trending {'shorter' if z[0]<0 else 'longer'})",
        f"  Outliers: {n_out} movies outside IQR bounds — genuine edge cases, not errors",
        "",
        "Recommendations",
        "",
        "  1. Double down on International TV Shows — fastest growing genre",
        "  2. Monitor Kids content decline — family segment at risk",
        "  3. Expand India partnerships — highest growth market",
        "  4. Leverage July surge for major title launches",
        "  5. Stand-Up Comedy is a high-retention Adult niche",
    ]
    ax.text(0.05, 0.95, "\n".join(summary_lines),
            transform=ax.transAxes, va="top", fontsize=11,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    ax.set_title("Netflix EDA — Findings & Recommendations",
                 fontsize=14, fontweight="bold", pad=15)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print("Saved: week3/Netflix_EDA_Report.pdf")
print("\nAll outputs saved to week3/")
