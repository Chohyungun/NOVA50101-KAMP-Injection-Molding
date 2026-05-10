"""Run EDA script — generates all figures and tables (notebook simulation)."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

from data import load_raw, generate_splits, get_fold
from utils import set_seed, setup_korean_font
from pathlib import Path
from scipy import stats as scipy_stats
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import seaborn as sns

set_seed(42)
setup_korean_font()
PROJECT_ROOT = Path('.')
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
TABLES_DIR  = PROJECT_ROOT / 'results' / 'tables'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

# ── Load all 8 CSVs ────────────────────────────────────────────────────────
DATASET_NAMES = [
    'labeled_data', 'moldset_labeled', 'unlabeled_data',
    'supervised_label_cn7', 'moldset_labeled_cn7', 'moldset_unlabeled_cn7',
    'moldset_labeled_rg3', 'moldset_unlabeled_rg3',
]
dfs = {n: load_raw(n) for n in DATASET_NAMES}

rows = []
for name, df in dfs.items():
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    rows.append({
        'dataset': name, 'rows': df.shape[0], 'cols': df.shape[1],
        'memory_MB': round(mem_mb, 2),
        'max_null_pct': round(df.isnull().mean().max() * 100, 1),
        'has_label': 'PassOrFail' in df.columns,
    })
pd.DataFrame(rows).to_csv(TABLES_DIR / 'eda_dataset_summary.csv', index=False)
print(f"[1] Dataset summary saved. Total rows: {sum(r['rows'] for r in rows):,}")

# ── labeled_data setup ─────────────────────────────────────────────────────
df_ld = dfs['labeled_data'].copy()
df_ld['PassOrFail'] = df_ld['PassOrFail'].astype(int)
num_cols = df_ld.select_dtypes('number').columns.tolist()
stds = df_ld[num_cols].std()
zero_var = stds[stds == 0].index.tolist()
DROP_COLS = zero_var + ['Barrel_Temperature_7']
FEAT_COLS = [c for c in num_cols if c not in DROP_COLS and c != 'PassOrFail']

pd.DataFrame({
    'variable': num_cols, 'mean': df_ld[num_cols].mean().values,
    'std': stds.values, 'n_unique': df_ld[num_cols].nunique().values,
    'drop_flag': [c in DROP_COLS for c in num_cols],
}).to_csv(TABLES_DIR / 'eda_variable_variance.csv', index=False)
print(f"[2] Variance table saved. zero_var={len(zero_var)}, effective={len(FEAT_COLS)}")

# ── Class distribution ─────────────────────────────────────────────────────
df_ld['date'] = pd.to_datetime(df_ld['PART_FACT_PLAN_DATE'], errors='coerce').dt.date
n_total = len(df_ld)
n_fail  = int(df_ld['PassOrFail'].sum())
n_pass  = n_total - n_fail
fail_pct = n_fail / n_total * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].pie(
    [n_pass, n_fail],
    labels=[f'양품 ({100-fail_pct:.2f}%)', f'불량 ({fail_pct:.2f}%)'],
    colors=['#4C72B0', '#DD8452'], startangle=90,
    explode=[0, 0.08], autopct='%1.2f%%',
)
axes[0].set_title('PassOrFail 분포 (labeled_data)', fontsize=12)

daily = df_ld.groupby('date')['PassOrFail'].sum().reset_index()
daily.columns = ['date', 'fail_cnt']
daily['date'] = pd.to_datetime(daily['date'])
axes[1].bar(daily['date'], daily['fail_cnt'], color='#DD8452', alpha=0.7)
axes[1].set_title('일별 불량 건수 시계열')
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'eda_class_distribution.png', bbox_inches='tight')
plt.close()
daily.to_csv(TABLES_DIR / 'eda_daily_fail.csv', index=False)
print(f"[3] Class distribution saved. fail%={fail_pct:.2f}%")

# ── Histograms + KDE ───────────────────────────────────────────────────────
pass_df = df_ld[df_ld['PassOrFail'] == 0]
fail_df = df_ld[df_ld['PassOrFail'] == 1]
ncols = 4
nrows = (len(FEAT_COLS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(FEAT_COLS):
    ax = axes[i]
    ax.hist(pass_df[col].dropna(), bins=40, color='#4C72B0', alpha=0.5,
            density=True, label='양품')
    ax.hist(fail_df[col].dropna(), bins=40, color='#DD8452', alpha=0.6,
            density=True, label='불량')
    try:
        pass_df[col].dropna().plot.kde(ax=ax, color='#4C72B0', linewidth=1.5)
        if len(fail_df[col].dropna()) > 1:
            fail_df[col].dropna().plot.kde(ax=ax, color='#DD8452', linewidth=1.5)
    except Exception:
        pass
    ax.set_title(col, fontsize=8)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)

for j in range(len(FEAT_COLS), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('유효 변수 히스토그램 + KDE (양품 vs 불량)', y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'eda_histograms.png', bbox_inches='tight')
plt.close()
print("[4] Histograms saved.")

# ── Correlation heatmap ────────────────────────────────────────────────────
corr = df_ld[FEAT_COLS].corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 5}, linewidths=0.3)
ax.set_title('변수 간 피어슨 상관행렬', fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'eda_correlation_heatmap.png', bbox_inches='tight')
plt.close()

THRESHOLD = 0.95
pairs = [
    {'var1': c1, 'var2': c2, 'pearson_r': round(corr.loc[c1, c2], 4)}
    for i, c1 in enumerate(FEAT_COLS)
    for j, c2 in enumerate(FEAT_COLS)
    if j > i and abs(corr.loc[c1, c2]) > THRESHOLD
]
hc_df = pd.DataFrame(pairs)
hc_df.to_csv(TABLES_DIR / 'eda_high_corr_pairs.csv', index=False)
print(f"[5] Correlation heatmap saved. high_corr_pairs={len(pairs)}")

# ── Time series ────────────────────────────────────────────────────────────
df_ld['datetime'] = pd.to_datetime(df_ld['PART_FACT_PLAN_DATE'], errors='coerce')
df_ld = df_ld.sort_values('datetime')
TS_COLS = [
    'Injection_Time', 'Max_Injection_Pressure', 'Cycle_Time',
    'Barrel_Temperature_1', 'Mold_Temperature_3',
]
dm = df_ld.groupby('datetime')[TS_COLS].mean()
fig, axes = plt.subplots(len(TS_COLS), 1, figsize=(14, 12), sharex=True)
for ax, col in zip(axes, TS_COLS):
    ax.plot(dm.index, dm[col], linewidth=1.2, color='#4C72B0')
    ax.set_ylabel(col, fontsize=8)
    ax.tick_params(labelsize=8)
plt.suptitle('일별 평균 공정 변수 시계열', fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'eda_time_series.png', bbox_inches='tight')
plt.close()
print("[6] Time series saved.")

# ── CN7 vs RG3 ─────────────────────────────────────────────────────────────
cn7 = dfs['moldset_labeled_cn7'].copy()
rg3 = dfs['moldset_labeled_rg3'].copy()
common = sorted(
    set(cn7.select_dtypes('number').columns) &
    set(rg3.select_dtypes('number').columns) - {'PassOrFail'}
)
ks_rows = []
for col in common:
    a, b = cn7[col].dropna().values, rg3[col].dropna().values
    if len(a) > 0 and len(b) > 0:
        stat, pval = scipy_stats.ks_2samp(a, b)
        ks_rows.append({'variable': col, 'ks_stat': round(stat, 4), 'p_value': round(pval, 4)})
ks_df = pd.DataFrame(ks_rows)
ks_df.to_csv(TABLES_DIR / 'eda_cn7_rg3_ks_test.csv', index=False)
sig = (ks_df['p_value'] < 0.05).sum()
print(f"[7] KS test saved. sig={sig}/{len(ks_df)}")

CMPF = [c for c in ['Injection_Time', 'Cycle_Time', 'Max_Injection_Pressure',
                     'Barrel_Temperature_1', 'Mold_Temperature_3'] if c in common]
fig, axes = plt.subplots(1, len(CMPF), figsize=(4 * len(CMPF), 4))
if len(CMPF) == 1:
    axes = [axes]
for ax, col in zip(axes, CMPF):
    cn7[col].dropna().plot.kde(ax=ax, label='CN7', color='#4C72B0', linewidth=2)
    rg3[col].dropna().plot.kde(ax=ax, label='RG3', color='#DD8452', linewidth=2)
    ax.set_title(col, fontsize=9)
    ax.legend(fontsize=8)
plt.suptitle('CN7 vs RG3 분포 비교', fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'eda_cn7_vs_rg3.png', bbox_inches='tight')
plt.close()
print("[7b] CN7 vs RG3 plot saved.")

# ── Guidebook stats recomputation ──────────────────────────────────────────
our_stats = df_ld[FEAT_COLS].agg(['count', 'mean', 'std', 'min', 'median', 'max']).T
our_stats.columns = ['count', 'mean', 'std', 'min', 'median', 'max']
our_stats = our_stats.round(4)
our_stats.to_csv(TABLES_DIR / 'eda_statistics_vs_guidebook.csv')
print("[8] Guidebook stats table saved.")

# ── 5-fold split generation ─────────────────────────────────────────────────
X = df_ld[FEAT_COLS].values
y = df_ld['PassOrFail'].values
generate_splits(X, y)
diffs = [abs(get_fold(i, X, y)[3].mean() - y.mean()) * 100 for i in range(5)]
print(f"[9] 5-fold splits OK. max_drift={max(diffs):.4f}pp")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EDA COMPLETE")
print(f"  유효 변수: {len(FEAT_COLS)}개")
print(f"  불량률: {fail_pct:.2f}%")
print(f"  강한 상관 짝 (|r|>0.95): {len(pairs)}개")
print(f"  CN7 vs RG3 KS 유의: {sig}/{len(ks_df)}개")
print(f"  Fold 최대 편차: {max(diffs):.4f}pp")
print("="*60)

fig_files = list(FIGURES_DIR.glob('eda_*.png'))
tbl_files = list(TABLES_DIR.glob('eda_*.csv'))
print(f"\n  Figures ({len(fig_files)}): {[f.name for f in sorted(fig_files)]}")
print(f"  Tables  ({len(tbl_files)}): {[f.name for f in sorted(tbl_files)]}")
