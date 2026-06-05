# Netflix Data Cleaning & Visualization
### Thiranex Internship — Week 1

A complete data cleaning and exploratory analysis pipeline on the Netflix titles dataset, producing 13 visualizations and a summary dashboard.

---

## What this project does

| Step | Description |
|------|-------------|
| Load | Read `netflix_titles.csv` and snapshot it for before/after comparison |
| Clean | Fix misplaced rating values, drop rows missing essential fields, fill non-critical nulls, remove duplicates |
| Engineer | Parse dates, extract year/month, split duration by content type, explode multi-value country and genre columns |
| Analyse | Distribution breakdowns, outlier detection via IQR, Winsorization capping |
| Visualize | 12 individual charts + 1 combined dashboard |
| Report | Auto-generated key insights printed to console |

---

## Setup

```bash
pip install pandas matplotlib seaborn
```

Place `netflix_titles.csv` in the same folder as `week1.py`, then run:

```bash
python week1.py
```

All output images are saved to the working directory automatically.

---

## Visualizations

### 1 · Movies vs TV Shows
![Movies vs TV Shows](1_movies_vs_tvshows.png)
Simple count of content type — shows Netflix's movie-heavy catalog split.

---

### 2 · Top 10 Countries
![Top 10 Countries](2_top_countries.png)
Countries with the most titles, after splitting multi-country entries so each country counts individually.

---

### 3 · Rating Distribution
![Rating Distribution](3_rating_distribution.png)
Horizontal bar chart of all content ratings, ordered by frequency. Highlights which audience segments Netflix targets most.

---

### 4 · Content Added Per Year
![Content Per Year](4_content_added_per_year.png)
Line plot showing how Netflix's catalog grew year over year — peak year is clearly visible.

---

### 5 · Movie Duration Distribution
![Movie Duration](5_movie_duration_distribution.png)
Histogram + KDE of movie runtimes in minutes, after IQR-based outlier capping.

---

### 6 · Top 10 Genres
![Top Genres](6_top_genres.png)
Most frequent genre tags across the catalog. Multi-genre titles are counted once per genre.

---

### 7 · Monthly Content Additions
![Monthly Additions](7_monthly_additions.png)
Which months Netflix adds the most content — useful for spotting seasonal release patterns.

---

### 8 · Missing Values Before vs After Cleaning
![Missing Values](8_missing_values_comparison.png)
Side-by-side heatmap showing data quality improvement. Bright streaks on the left disappear on the right.

---

### 9 · Movie Duration Boxplot
![Duration Boxplot](9_movie_duration_outliers.png)
Boxplot of movie durations after Winsorization — the distribution is now compact without losing any rows.

---

### 10 · Content Type by Country
![Content by Country](10_content_type_by_country.png)
Grouped bar chart showing the Movie / TV Show split within each of the top 10 producing countries.

---

### 11 · Year-over-Year Growth Rate
![YoY Growth](11_yoy_growth.png)
Percentage growth in new titles added each year. The dashed zero line makes growth vs decline easy to read.

---

### 12 · TV Show Season Distribution
![TV Seasons](12_tvshow_seasons.png)
How many seasons most Netflix TV shows have — reveals whether the platform favours short-run or long-run series.

---

### 13 · Combined Dashboard
![Dashboard](13_netflix_dashboard.png)
All key charts in one 3×3 grid for a quick executive summary view.

---

## Data Cleaning Details

### Misplaced rating values
Some rows had duration strings like `"74 min"` in the `rating` column. The script detects these with a regex, rescues the value into `duration` if that field was empty, then marks the rating as `NaN` so it gets filled with `"Not Rated"`.

### Essential field drops
Rows missing `show_id`, `title`, or `type` are dropped — they cannot be meaningfully identified or categorised.

### Non-critical fills

| Column | Fill value |
|--------|-----------|
| `director` | `"Unknown"` |
| `cast` | `"Not Available"` |
| `country` | `"Unknown"` |
| `rating` | `"Not Rated"` |
| `listed_in` | `"Unknown"` |
| `description` | `"No Description"` |

### Outlier handling
Movie durations are capped using IQR Winsorization (not removed). Rows with genuinely unusual runtimes are real films — dropping them would introduce bias. Capping limits their influence on statistics while keeping the data complete.

---

## Output files

| File | Description |
|------|-------------|
| `cleaned_netflix_titles.csv` | Cleaned dataset ready for further analysis |
| `1_movies_vs_tvshows.png` … `12_tvshow_seasons.png` | Individual charts |
| `13_netflix_dashboard.png` | Combined 3×3 dashboard (150 dpi) |

---

## Sample console output

```
Dataset loaded  →  shape: (8807, 12)
Misplaced rating entries (e.g. '74 min'): 3
Dropped 3 rows missing show_id / title / type
Duplicate rows: 0
Cleaned dataset saved to cleaned_netflix_titles.csv

── Key insights ──
Total titles               : 6169
Movies                     : 4265 (69.1%)
TV Shows                   : 1904 (30.9%)
Most common rating         : TV-MA
Most prolific country      : United States
Most common genre          : Dramas
Peak content year          : 2019
Avg movie duration (capped): 99.8 min
Outliers detected & capped : 108
```

*(Numbers vary slightly depending on dataset version.)*

---

## Project structure

```
week1.py                      ← main script
netflix_titles.csv            ← source data (not committed)
cleaned_netflix_titles.csv    ← output: cleaned data
1_movies_vs_tvshows.png
2_top_countries.png
3_rating_distribution.png
4_content_added_per_year.png
5_movie_duration_distribution.png
6_top_genres.png
7_monthly_additions.png
8_missing_values_comparison.png
9_movie_duration_outliers.png
10_content_type_by_country.png
11_yoy_growth.png
12_tvshow_seasons.png
13_netflix_dashboard.png
```
