import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ── Load ───────────────────────────────────────────────────────────────────────

df = pd.read_csv("netflix_titles.csv")
print("Dataset loaded  →  shape:", df.shape)

print("\nFirst 5 rows")
print(df.head())
print("\nDataset info")
print(df.info())
print("\nMissing values before cleaning")
print(df.isnull().sum())

df_original = df.copy()

# ── Clean ──────────────────────────────────────────────────────────────────────

# Some rows have duration strings like "74 min" in the rating column.
# Rescue the value into duration (when duration is empty), then clear rating.
misplaced = df["rating"].str.match(r"^\d+ min$", na=False)
print(f"\nMisplaced rating entries (e.g. '74 min'): {misplaced.sum()}")
if misplaced.sum():
    print(df.loc[misplaced, ["title", "rating", "duration"]])

needs_rescue = misplaced & df["duration"].isna()
df.loc[needs_rescue, "duration"] = df.loc[needs_rescue, "rating"]
df.loc[misplaced, "rating"] = pd.NA

before = len(df)
df = df.dropna(subset=["show_id", "title", "type"])
print(f"Dropped {before - len(df)} rows missing show_id / title / type")

df["director"]    = df["director"].fillna("Unknown")
df["cast"]        = df["cast"].fillna("Not Available")
df["country"]     = df["country"].fillna("Unknown")
df["rating"]      = df["rating"].fillna("Not Rated")
df["listed_in"]   = df["listed_in"].fillna("Unknown")
df["description"] = df["description"].fillna("No Description")

dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
df.drop_duplicates(inplace=True)

# ── Feature engineering ────────────────────────────────────────────────────────

df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
df = df.dropna(subset=["date_added"])

df["year_added"]  = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month_name()

df["duration_num"] = pd.to_numeric(
    df["duration"].str.extract(r"(\d+)")[0], errors="coerce"
)

# Movies → minutes; TV Shows → seasons  (kept separate to avoid mixing units)
df["duration_min"]     = df.loc[df["type"] == "Movie",   "duration_num"]
df["duration_seasons"] = df.loc[df["type"] == "TV Show", "duration_num"]

countries = (
    df["country"].str.split(", ").explode().str.strip()
)
countries = countries[countries != "Unknown"]

genres = (
    df[df["listed_in"] != "Unknown"]["listed_in"]
    .str.split(", ").explode()
)

yearly_counts = df["year_added"].value_counts().sort_index()
yoy_growth    = yearly_counts.pct_change() * 100

# ── Save cleaned file ──────────────────────────────────────────────────────────

df.to_csv("cleaned_netflix_titles.csv", index=False)
print("\nCleaned dataset saved to cleaned_netflix_titles.csv")

# ── EDA summary ────────────────────────────────────────────────────────────────

print("\n── Content type distribution ──")
print(df["type"].value_counts())
print("\n── Top 10 countries ──")
print(countries.value_counts().head(10))
print("\n── Top ratings ──")
print(df["rating"].value_counts().head(10))
print("\n── Content added per year ──")
print(yearly_counts)
print("\n── Year-over-year growth (%) ──")
print(yoy_growth.round(1))

# ── Outlier handling ───────────────────────────────────────────────────────────

movies   = df[df["type"] == "Movie"]
tv_shows = df[df["type"] == "TV Show"]

Q1 = movies["duration_min"].quantile(0.25)
Q3 = movies["duration_min"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = movies[
    (movies["duration_min"] < lower_bound) |
    (movies["duration_min"] > upper_bound)
]

print("\n── Outlier analysis ──")
print(f"Q1: {Q1}  |  Q3: {Q3}  |  IQR: {IQR}")
print(f"Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
print(f"Outliers found: {len(outliers)}")

# Cap rather than drop — real films that happen to be very short/long still matter
df["duration_min"] = df["duration_min"].clip(lower=lower_bound, upper=upper_bound)

movies   = df[df["type"] == "Movie"]
tv_shows = df[df["type"] == "TV Show"]

print(f"Outliers capped to [{lower_bound:.1f}, {upper_bound:.1f}] min")
print(movies["duration_min"].describe().round(1))

# ── Visualizations ─────────────────────────────────────────────────────────────

# 1. Movies vs TV Shows
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="type", hue="type", palette="Set2", legend=False)
plt.title("Movies vs TV Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("1_movies_vs_tvshows.png")
plt.show()

# 2. Top 10 countries
plt.figure(figsize=(12, 6))
countries.value_counts().head(10).plot(kind="bar", color="steelblue")
plt.title("Top 10 Countries Producing Netflix Content")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("2_top_countries.png")
plt.show()

# 3. Rating distribution
plt.figure(figsize=(12, 6))
sns.countplot(
    y="rating", data=df,
    order=df["rating"].value_counts().index,
    hue="rating", palette="muted", legend=False
)
plt.title("Content Ratings Distribution")
plt.tight_layout()
plt.savefig("3_rating_distribution.png")
plt.show()

# 4. Content added per year
plt.figure(figsize=(12, 6))
yearly_counts.plot(marker="o", color="tomato")
plt.title("Netflix Content Added Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig("4_content_added_per_year.png")
plt.show()

# 5. Movie duration histogram
plt.figure(figsize=(12, 6))
sns.histplot(movies["duration_min"], bins=30, kde=True, color="mediumseagreen")
plt.title("Movie Duration Distribution (Minutes, Outliers Capped)")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("5_movie_duration_distribution.png")
plt.show()

# 6. Top genres
plt.figure(figsize=(12, 6))
genres.value_counts().head(10).plot(kind="bar", color="mediumpurple")
plt.title("Top 10 Netflix Genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("6_top_genres.png")
plt.show()

# 7. Monthly additions
plt.figure(figsize=(12, 6))
sns.countplot(
    y="month_added", data=df,
    order=df["month_added"].value_counts().index,
    hue="month_added", palette="coolwarm", legend=False
)
plt.title("Monthly Content Additions")
plt.tight_layout()
plt.savefig("7_monthly_additions.png")
plt.show()

# 8. Missing values before vs after
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.heatmap(df_original.isnull(), cbar=False, yticklabels=False, ax=axes[0])
axes[0].set_title("Missing Values BEFORE Cleaning")
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=axes[1])
axes[1].set_title("Missing Values AFTER Cleaning")
plt.suptitle("Missing Value Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("8_missing_values_comparison.png")
plt.show()

# 9. Movie duration boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x=movies["duration_min"])
plt.title("Movie Duration Boxplot (Outliers Capped via IQR)")
plt.tight_layout()
plt.savefig("9_movie_duration_outliers.png")
plt.show()

# 10. Movies vs TV Shows by top 10 countries
top_country_list = countries.value_counts().head(10).index.tolist()
df["primary_country"] = df["country"].str.split(", ").str[0].str.strip()
country_type_df = df[df["primary_country"].isin(top_country_list)]

plt.figure(figsize=(14, 7))
sns.countplot(
    data=country_type_df,
    y="primary_country", hue="type",
    order=top_country_list, palette="Set1"
)
plt.title("Movies vs TV Shows by Top 10 Countries")
plt.xlabel("Count")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("10_content_type_by_country.png")
plt.show()

# 11. Year-over-year growth
plt.figure(figsize=(12, 6))
yoy_growth.dropna().plot(kind="bar", color="darkorange")
plt.title("Year-over-Year Growth in Netflix Content (%)")
plt.xlabel("Year")
plt.ylabel("Growth Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.savefig("11_yoy_growth.png")
plt.show()

# 12. TV show season distribution
tv_season_counts = (
    tv_shows["duration_seasons"].dropna().astype(int)
    .value_counts().sort_index()
)
plt.figure(figsize=(12, 6))
tv_season_counts.plot(kind="bar", color="steelblue")
plt.title("TV Show Season Count Distribution")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of Shows")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("12_tvshow_seasons.png")
plt.show()

# ── Dashboard ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(22, 16))
fig.suptitle("Netflix Content Analysis Dashboard", fontsize=20, fontweight="bold")

df["type"].value_counts().plot(
    kind="bar", ax=axes[0][0], color=["#E50914", "#221F1F"]
)
axes[0][0].set_title("Movies vs TV Shows")
axes[0][0].tick_params(axis="x", rotation=0)

countries.value_counts().head(10).plot(
    kind="bar", ax=axes[0][1], color="steelblue"
)
axes[0][1].set_title("Top 10 Countries")
axes[0][1].tick_params(axis="x", rotation=45)

df["rating"].value_counts().head(5).plot(
    kind="bar", ax=axes[0][2], color="mediumpurple"
)
axes[0][2].set_title("Top 5 Ratings")
axes[0][2].tick_params(axis="x", rotation=45)

yearly_counts.plot(ax=axes[1][0], marker="o", color="tomato")
axes[1][0].set_title("Content Added Per Year")

axes[1][1].hist(movies["duration_min"].dropna(), bins=30,
                color="mediumseagreen", edgecolor="white")
axes[1][1].set_title("Movie Duration (Capped)")

genres.value_counts().head(10).plot(
    kind="bar", ax=axes[1][2], color="darkorange"
)
axes[1][2].set_title("Top 10 Genres")
axes[1][2].tick_params(axis="x", rotation=45)

df["month_added"].value_counts().plot(
    kind="bar", ax=axes[2][0], color="coral"
)
axes[2][0].set_title("Monthly Content Additions")
axes[2][0].tick_params(axis="x", rotation=45)

yoy_growth.dropna().plot(kind="bar", ax=axes[2][1], color="darkorange")
axes[2][1].axhline(y=0, color="black", linewidth=0.8, linestyle="--")
axes[2][1].set_title("YoY Growth (%)")
axes[2][1].tick_params(axis="x", rotation=45)

axes[2][2].boxplot(movies["duration_min"].dropna(), vert=False)
axes[2][2].set_title("Movie Duration (After Capping)")
axes[2][2].set_xlabel("Minutes")

plt.tight_layout()
plt.savefig("13_netflix_dashboard.png", bbox_inches="tight", dpi=150)
plt.show()
print("Dashboard saved → 13_netflix_dashboard.png")

# ── Key insights ───────────────────────────────────────────────────────────────

total        = len(df)
movie_count  = (df["type"] == "Movie").sum()
tv_count     = (df["type"] == "TV Show").sum()
movie_pct    = round(movie_count / total * 100, 1)
tv_pct       = round(tv_count    / total * 100, 1)
top_country  = countries.value_counts().index[0]
top_genre    = genres.mode()[0]
top_rating   = df["rating"].mode()[0]
peak_year    = int(yearly_counts.idxmax())
avg_duration = round(movies["duration_min"].mean(), 1)

print("\n── Key insights ──")
print(f"Total titles               : {total}")
print(f"Movies                     : {movie_count} ({movie_pct}%)")
print(f"TV Shows                   : {tv_count} ({tv_pct}%)")
print(f"Most common rating         : {top_rating}")
print(f"Most prolific country      : {top_country}")
print(f"Most common genre          : {top_genre}")
print(f"Peak content year          : {peak_year}")
print(f"Avg movie duration (capped): {avg_duration} min")
print(f"Outliers detected & capped : {len(outliers)}")

print(f"""
── Data story ──

Netflix's catalog of {total} titles leans heavily toward Movies ({movie_pct}%),
with TV Shows making up the remaining {tv_pct}%.

{top_country} dominates content production by a significant margin, and the most
common genre is '{top_genre}', confirming Netflix's focus on broad-appeal
international dramas and comedies.

Most content is rated '{top_rating}', indicating the platform primarily targets
mature audiences rather than families.

Content additions peaked in {peak_year} — Netflix's most aggressive phase of
global catalog expansion. Growth has been uneven year-to-year, with some years
showing sharp jumps.

The average movie runs {avg_duration} minutes after capping. {len(outliers)} movies
fell outside the normal duration range and were capped using IQR-based
Winsorization — preserving the rows while reducing their influence on statistics.

Monthly patterns show certain months receive significantly more titles, likely
tied to Netflix's strategic release calendar around awards seasons and holiday
viewing windows.
""")

print("Done!")
