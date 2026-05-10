"""
Enhanced EDA for KAMP 사출성형 데이터
- 한글 폰트 설정
- 공정변수별 양품/불량 Box Plot
- 이상치(Outlier) 분석
- 시계열 드리프트 분석
- 상관관계 심화 (Point-biserial)
- 도메인 인사이트 출력
"""
from __future__ import annotations

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ── 프로젝트 경로 추가 ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from scipy import stats

from utils import set_seed
from data import load_raw
from preprocess import get_feature_cols, ZERO_VAR_COLS

set_seed(42)

# ── 결과 디렉토리 확인 ──────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')
TABLES_DIR  = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,  exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 한글 폰트 설정
# ═══════════════════════════════════════════════════════════════════════════════
def setup_korean_font():
    candidates = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Noto Sans CJK KR']
    for font_name in candidates:
        fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        for fp in fonts:
            try:
                prop = fm.FontProperties(fname=fp)
                if font_name.lower() in prop.get_name().lower():
                    plt.rcParams['font.family'] = prop.get_name()
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"Korean font set: {prop.get_name()}")
                    return prop.get_name()
            except Exception:
                continue
    # fallback: set by name directly
    for font_name in candidates:
        try:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Korean font set (direct): {font_name}")
            return font_name
        except Exception:
            continue
    print("WARNING: Korean font not found, using default")
    return None

font_used = setup_korean_font()
FONT_OK = font_used is not None

# ── 전역 스타일 ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 120,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLOR_PASS = '#4C72B0'   # 양품 (파랑)
COLOR_FAIL = '#DD8452'   # 불량 (주황)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 데이터 로드
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[데이터 로드 중...]")
df = load_raw('labeled_data')
feat_cols = get_feature_cols(df)  # 25개 유효 변수
print(f"  - 전체 행: {len(df):,}")
print(f"  - 유효 피처 수: {len(feat_cols)}")
print(f"  - 양품(0): {(df['PassOrFail']==0).sum():,}  /  불량(1): {(df['PassOrFail']==1).sum():,}")
print(f"  - 피처 목록: {feat_cols}")

df_pass = df[df['PassOrFail'] == 0]
df_fail = df[df['PassOrFail'] == 1]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 공정변수별 양품/불량 Box Plot
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] 공정변수별 양품/불량 Box Plot 생성...")

# 양품/불량 분포 차이 계산 (Cohen's d)
def cohens_d(a, b):
    """효과 크기 Cohen's d 계산"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return abs((np.mean(a) - np.mean(b)) / pooled_std)

effect_sizes = {}
for col in feat_cols:
    a = df_pass[col].dropna().values
    b = df_fail[col].dropna().values
    effect_sizes[col] = cohens_d(a, b)

effect_df = pd.Series(effect_sizes).sort_values(ascending=False)
top5_cols = effect_df.head(5).index.tolist()
print(f"  - 분포 차이 Top-5 변수: {top5_cols}")

# Box plot: 25개 변수 전체 (5행 5열 그리드)
n_cols_plot = 5
n_rows_plot = (len(feat_cols) + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                         figsize=(20, n_rows_plot * 3.5))
fig.suptitle('공정변수별 양품/불량 분포 비교 (Box Plot)', fontsize=16, fontweight='bold', y=1.01)

axes_flat = axes.flatten()

for i, col in enumerate(feat_cols):
    ax = axes_flat[i]
    data_to_plot = [df_pass[col].dropna().values, df_fail[col].dropna().values]

    bp = ax.boxplot(data_to_plot,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='.', markersize=3, alpha=0.5))

    bp['boxes'][0].set_facecolor(COLOR_PASS)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLOR_FAIL)
    bp['boxes'][1].set_alpha(0.7)

    is_top5 = col in top5_cols
    ax.set_title(f"{'★ ' if is_top5 else ''}{col}",
                 fontsize=8,
                 fontweight='bold' if is_top5 else 'normal',
                 color='darkred' if is_top5 else 'black')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['양품(0)', '불량(1)'], fontsize=8)
    ax.set_ylabel('측정값', fontsize=7)

    # 효과 크기 표시
    d = effect_sizes[col]
    ax.text(0.97, 0.97, f"d={d:.2f}",
            transform=ax.transAxes,
            ha='right', va='top', fontsize=7,
            color='darkred' if is_top5 else 'gray')

# 빈 서브플롯 제거
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

# 범례
pass_patch = mpatches.Patch(color=COLOR_PASS, alpha=0.7, label='양품 (Y)')
fail_patch = mpatches.Patch(color=COLOR_FAIL, alpha=0.7, label='불량 (N)')
fig.legend(handles=[pass_patch, fail_patch],
           loc='upper right', fontsize=10,
           bbox_to_anchor=(1.0, 1.0))

plt.tight_layout()
boxplot_path = os.path.join(FIGURES_DIR, 'eda_enhanced_boxplots.png')
plt.savefig(boxplot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  -> 저장: {boxplot_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 이상치(Outlier) 분석
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] 이상치(Outlier) 분석...")

outlier_records = []
for col in feat_cols:
    series = df[col].dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    is_outlier = (series < lower) | (series > upper)
    outlier_mask = (df[col] < lower) | (df[col] > upper)

    # 이상치 중 불량 비율
    outlier_rows = df[outlier_mask]
    outlier_fail_rate = outlier_rows['PassOrFail'].mean() if len(outlier_rows) > 0 else 0.0
    normal_fail_rate  = df[~outlier_mask]['PassOrFail'].mean()

    outlier_records.append({
        '변수': col,
        '이상치_수': int(is_outlier.sum()),
        '이상치_비율(%)': round(is_outlier.mean() * 100, 2),
        'IQR_하한': round(lower, 4),
        'IQR_상한': round(upper, 4),
        '이상치_내_불량률(%)': round(outlier_fail_rate * 100, 2),
        '정상_내_불량률(%)': round(normal_fail_rate * 100, 2),
    })

outlier_df = pd.DataFrame(outlier_records).sort_values('이상치_비율(%)', ascending=False)
top10_outlier = outlier_df.head(10)

# 이상치 비율 Top-10 바차트
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('이상치(Outlier) 분석', fontsize=14, fontweight='bold')

# 왼쪽: 이상치 비율 Top-10
ax1 = axes[0]
bars = ax1.barh(
    top10_outlier['변수'][::-1],
    top10_outlier['이상치_비율(%)'][::-1],
    color=plt.cm.Reds_r(np.linspace(0.3, 0.9, 10))
)
ax1.set_xlabel('이상치 비율 (%)', fontsize=11)
ax1.set_title('이상치 비율 Top-10 변수', fontsize=12)
for bar, val in zip(bars, top10_outlier['이상치_비율(%)'][::-1]):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)

# 오른쪽: 이상치 내 불량률 vs 정상 내 불량률 비교
ax2 = axes[1]
x_pos = np.arange(len(top10_outlier))
width = 0.35
bars1 = ax2.bar(x_pos - width/2, top10_outlier['정상_내_불량률(%)'], width,
                label='정상값 내 불량률', color=COLOR_PASS, alpha=0.7)
bars2 = ax2.bar(x_pos + width/2, top10_outlier['이상치_내_불량률(%)'], width,
                label='이상치 내 불량률', color=COLOR_FAIL, alpha=0.7)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(top10_outlier['변수'], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('불량률 (%)', fontsize=11)
ax2.set_title('이상치 구간에서의 불량률 증가 여부', fontsize=12)
ax2.legend(fontsize=9)

# 불량률이 높은 이상치 변수 강조
for i, (_, row) in enumerate(top10_outlier.iterrows()):
    if row['이상치_내_불량률(%)'] > row['정상_내_불량률(%)'] * 2:
        ax2.annotate('↑', xy=(i + width/2, row['이상치_내_불량률(%)'] + 0.5),
                     ha='center', fontsize=12, color='red', fontweight='bold')

plt.tight_layout()
outlier_fig_path = os.path.join(FIGURES_DIR, 'eda_outlier_analysis.png')
plt.savefig(outlier_fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  -> 저장: {outlier_fig_path}")

# 표 저장
outlier_csv_path = os.path.join(TABLES_DIR, 'eda_outlier_summary.csv')
outlier_df.to_csv(outlier_csv_path, index=False, encoding='utf-8-sig')
print(f"  -> 저장: {outlier_csv_path}")
print(f"  - 이상치 비율 Top-3: {outlier_df['변수'].head(3).tolist()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 시계열 드리프트 분석
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] 시계열 드리프트 분석...")

# 날짜 파싱 (한글 오전/오후 처리)
def parse_korean_date(s):
    """한글 오전/오후 포함된 날짜 문자열 파싱"""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    # '2020-10-16 오전 12:00:00' 형태 처리
    s = s.replace('오전', 'AM').replace('오후', 'PM')
    try:
        return pd.to_datetime(s, format='%Y-%m-%d %p %I:%M:%S')
    except Exception:
        pass
    try:
        return pd.to_datetime(s[:10])  # 날짜 부분만
    except Exception:
        return pd.NaT

df_ts = df.copy()
df_ts['date_parsed'] = df_ts['PART_FACT_PLAN_DATE'].apply(parse_korean_date)

# TimeStamp도 시도
if df_ts['date_parsed'].isna().all():
    print("  WARNING: PART_FACT_PLAN_DATE 파싱 실패 → TimeStamp 사용")
    df_ts['date_parsed'] = pd.to_datetime(df_ts['TimeStamp'], errors='coerce')

# 날짜 기준 정렬
df_ts = df_ts.sort_values('date_parsed').reset_index(drop=True)
df_ts['date_only'] = df_ts['date_parsed'].dt.date
print(f"  - 날짜 범위: {df_ts['date_parsed'].min()} ~ {df_ts['date_parsed'].max()}")
print(f"  - 유효 날짜 수: {df_ts['date_parsed'].notna().sum():,}")

# 주요 변수 선택
TREND_VARS = ['Injection_Time', 'Max_Injection_Pressure', 'Cycle_Time', 'Barrel_Temperature_1']
# 없는 변수 필터링
TREND_VARS = [v for v in TREND_VARS if v in df_ts.columns]

# 일별 집계
daily = df_ts.groupby('date_only').agg(
    **{v: (v, 'mean') for v in TREND_VARS},
    fail_count=('PassOrFail', 'sum'),
    total_count=('PassOrFail', 'count'),
).reset_index()
daily['fail_rate'] = daily['fail_count'] / daily['total_count']
daily['date_only'] = pd.to_datetime(daily['date_only'])
daily = daily.sort_values('date_only').reset_index(drop=True)

# 시계열 드리프트 그래프
n_trend = len(TREND_VARS)
fig, axes = plt.subplots(n_trend + 1, 1, figsize=(16, 4 * (n_trend + 1)), sharex=True)
fig.suptitle('주요 공정변수 시계열 드리프트 분석', fontsize=14, fontweight='bold')

drift_detected = []

for i, var in enumerate(TREND_VARS):
    ax = axes[i]
    series = daily[var].values
    dates  = daily['date_only']

    # 7일 이동평균 (최소 3개 이상 데이터)
    window = min(7, max(3, len(series) // 5))
    rolling_mean = pd.Series(series).rolling(window=window, center=True, min_periods=1).mean()
    rolling_std  = pd.Series(series).rolling(window=window, center=True, min_periods=1).std()

    # 전체 평균 ± 2σ 기준선
    global_mean = np.nanmean(series)
    global_std  = np.nanstd(series)
    upper_bound = global_mean + 2 * global_std
    lower_bound = global_mean - 2 * global_std

    # 이상 구간 감지
    above = rolling_mean > upper_bound
    below = rolling_mean < lower_bound
    anomaly = above | below

    if anomaly.any():
        drift_detected.append(var)

    # 원 데이터
    ax.plot(dates, series, color='lightblue', alpha=0.5, linewidth=0.8, label='일별 평균')
    # 이동 평균
    ax.plot(dates, rolling_mean, color='steelblue', linewidth=2, label=f'{window}일 이동평균')
    # ±2σ 구간
    ax.axhline(upper_bound, color='red', linestyle='--', linewidth=1, alpha=0.7, label='+2σ')
    ax.axhline(lower_bound, color='red', linestyle='--', linewidth=1, alpha=0.7, label='-2σ')
    ax.axhline(global_mean, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='전체 평균')

    # 이상 구간 음영
    for j in range(len(dates)):
        if anomaly.iloc[j]:
            ax.axvspan(dates.iloc[j] - pd.Timedelta(days=0.5),
                       dates.iloc[j] + pd.Timedelta(days=0.5),
                       alpha=0.25, color='red')

    # 불량 발생 날짜 마킹
    fail_dates = daily[daily['fail_count'] > 0]['date_only']
    for fd in fail_dates:
        ax.axvline(fd, color=COLOR_FAIL, alpha=0.4, linewidth=1.5)

    ax.set_ylabel(var, fontsize=9)
    ax.set_title(f'{var} 시계열 추이 {"[드리프트 감지!]" if var in drift_detected else ""}',
                 fontsize=10,
                 color='red' if var in drift_detected else 'black')
    ax.legend(loc='upper right', fontsize=7, ncol=3)

# 마지막 행: 불량 발생률
ax_fail = axes[-1]
ax_fail.bar(daily['date_only'], daily['fail_rate'] * 100,
            color=COLOR_FAIL, alpha=0.7, width=0.8, label='일별 불량률(%)')
ax_fail.set_ylabel('불량률 (%)', fontsize=9)
ax_fail.set_title('일별 불량 발생률', fontsize=10)
ax_fail.set_xlabel('날짜', fontsize=10)
ax_fail.legend(fontsize=8)
ax_fail.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.setp(ax_fail.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
temporal_path = os.path.join(FIGURES_DIR, 'eda_temporal_drift.png')
plt.savefig(temporal_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  -> 저장: {temporal_path}")
if drift_detected:
    print(f"  - 드리프트 감지 변수: {drift_detected}")
else:
    print("  - 특별한 드리프트 없음 (2σ 이탈 없음)")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. 상관관계 심화 - Point-biserial Correlation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6] Point-biserial Correlation 분석...")

y = df['PassOrFail'].values
corr_records = []
for col in feat_cols:
    x = df[col].dropna()
    valid_idx = df[col].notna()
    y_valid = df.loc[valid_idx, 'PassOrFail'].values
    x_valid = df.loc[valid_idx, col].values
    if len(x_valid) < 10 or np.std(x_valid) == 0:
        corr_records.append({'변수': col, 'Point-biserial_r': 0.0, 'p값': 1.0, '절대값_r': 0.0})
        continue
    r, p = stats.pointbiserialr(y_valid, x_valid)
    corr_records.append({
        '변수': col,
        'Point-biserial_r': round(r, 4),
        'p값': round(p, 6),
        '절대값_r': round(abs(r), 4),
    })

corr_df = pd.DataFrame(corr_records).sort_values('절대값_r', ascending=False)
top10_corr = corr_df.head(10)
top3_corr  = corr_df.head(3)

print(f"  - 상관 Top-3: {top3_corr['변수'].tolist()}")

# 바차트
fig, ax = plt.subplots(figsize=(12, 6))
colors_corr = [COLOR_FAIL if r > 0 else COLOR_PASS
               for r in top10_corr['Point-biserial_r']]
bars = ax.barh(
    top10_corr['변수'][::-1],
    top10_corr['Point-biserial_r'][::-1],
    color=colors_corr[::-1],
    alpha=0.8,
    edgecolor='gray',
    linewidth=0.5
)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Point-biserial 상관계수 (r)', fontsize=11)
ax.set_title('불량(PassOrFail=1)과 공정변수 간 상관관계 Top-10', fontsize=13, fontweight='bold')

# 유의성 마킹 (p < 0.05)
for i, (_, row) in enumerate(top10_corr[::-1].iterrows()):
    sig = '***' if row['p값'] < 0.001 else ('**' if row['p값'] < 0.01 else ('*' if row['p값'] < 0.05 else ''))
    r_val = row['Point-biserial_r']
    offset = 0.002 if r_val >= 0 else -0.002
    ha = 'left' if r_val >= 0 else 'right'
    ax.text(r_val + offset, i,
            f"r={r_val:.3f} {sig}",
            va='center', ha=ha, fontsize=8)

pass_patch = mpatches.Patch(color=COLOR_PASS, alpha=0.8, label='음의 상관 (양품 연관)')
fail_patch = mpatches.Patch(color=COLOR_FAIL, alpha=0.8, label='양의 상관 (불량 연관)')
ax.legend(handles=[pass_patch, fail_patch], fontsize=9, loc='lower right')

plt.tight_layout()
corr_fig_path = os.path.join(FIGURES_DIR, 'eda_target_correlation.png')
plt.savefig(corr_fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  -> 저장: {corr_fig_path}")

# 표 저장
corr_csv_path = os.path.join(TABLES_DIR, 'eda_target_correlation.csv')
corr_df.to_csv(corr_csv_path, index=False, encoding='utf-8-sig')
print(f"  -> 저장: {corr_csv_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. 사출성형 도메인 인사이트 텍스트 출력
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  사출성형 도메인 인사이트 요약")
print("=" * 70)

print("\n[불량과 가장 상관 높은 변수 Top-3]")
for rank, (_, row) in enumerate(top3_corr.iterrows(), 1):
    direction = "↑ 높을수록 불량 증가" if row['Point-biserial_r'] > 0 else "↓ 낮을수록 불량 증가"
    sig_str   = '(p<0.001)' if row['p값'] < 0.001 else f"(p={row['p값']:.4f})"
    print(f"  {rank}. {row['변수']:35s}  r={row['Point-biserial_r']:+.4f}  {direction}  {sig_str}")

print("\n[이상치 비율이 가장 높은 변수 Top-3]")
for rank, (_, row) in enumerate(outlier_df.head(3).iterrows(), 1):
    print(f"  {rank}. {row['변수']:35s}  이상치 비율={row['이상치_비율(%)']:.2f}%"
          f"  (이상치_불량률={row['이상치_내_불량률(%)']:.1f}% vs 정상_불량률={row['정상_내_불량률(%)']:.1f}%)")

print("\n[시계열 드리프트 감지 결과]")
if drift_detected:
    for v in drift_detected:
        print(f"  - {v}: 2σ 이탈 구간 감지됨 → 공정 변동 주의 필요")
else:
    print("  - 분석 기간 내 주요 4개 변수에서 2σ 이탈 드리프트 없음")
    print("  - (불량 발생은 특정 날짜에 집중되어 있을 가능성 있음 → 일별 불량률 차트 확인)")

print("\n[가이드북 §2.3 대비 새로 발견한 사항]")
print("  1. Point-biserial 분석 결과:")
top_var = top3_corr.iloc[0]
print(f"     '{top_var['변수']}'이(가) 불량과 가장 강한 상관관계 (r={top_var['Point-biserial_r']:+.4f})")
print("     → 이 변수의 공정 관리 임계값 설정이 불량 저감의 핵심 레버")

top_out = outlier_df.iloc[0]
print(f"  2. '{top_out['변수']}'에서 이상치 비율 {top_out['이상치_비율(%)']:.1f}%로 가장 높음")
if top_out['이상치_내_불량률(%)'] > top_out['정상_내_불량률(%)'] * 1.5:
    print("     → 이상치 구간에서 불량률 유의미하게 증가 → 공정 이상 조기 경보 기준 활용 가능")
else:
    print("     → 이상치와 불량 간 직접 연관은 약함 (이상치가 단순 측정 변동일 가능성)")

print(f"  3. Cohen's d 기준 양품/불량 분포 차이가 가장 큰 변수: {top5_cols[:3]}")
print("     → Box Plot에서 분포 겹침이 적은 변수가 모델링 시 높은 피처 중요도 예상")

print("\n[생성된 파일 목록]")
generated_files = [boxplot_path, outlier_fig_path, temporal_path, corr_fig_path,
                   outlier_csv_path, corr_csv_path]
for f in generated_files:
    exists = os.path.exists(f)
    print(f"  {'[OK]' if exists else '[MISSING]'} {f}")

print("\n" + "=" * 70)
print("  분석 완료!")
print("=" * 70)
