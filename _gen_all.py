#!/usr/bin/env python3
"""Fix NB08 + generate 9 method-specific EDA notebooks."""
import json, uuid
from pathlib import Path

ROOT   = Path(r'E:\IAAI_Term_project')
NB_DIR = ROOT / 'notebooks'

def uid(): return uuid.uuid4().hex[:8]
def cc(s): return {"cell_type":"code","id":uid(),"metadata":{},"outputs":[],"execution_count":None,"source":s}
def mc(s): return {"cell_type":"markdown","id":uid(),"metadata":{},"source":s}
def get_nb(p): return json.loads(p.read_text(encoding='utf-8'))
def save_nb(nb,p): p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
def find_i(cells, cid): return next(i for i,c in enumerate(cells) if c['id']==cid)
def make_nb(cells):
    return {"nbformat":4,"nbformat_minor":5,
            "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                        "language_info":{"name":"python","version":"3.11.0"}},
            "cells":cells}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Fix NB08 — inline display + 3D visualizations
# ══════════════════════════════════════════════════════════════════════════════
print("=== Fixing NB08 ===")
nb08 = get_nb(NB_DIR / '08_pseudo_labeling.ipynb')
c8   = nb08['cells']

# 1a) Edit setup cell: remove Agg, add Axes3D
for cell in c8:
    if cell['id'] == '17a81e93':
        src = cell['source'] if isinstance(cell['source'],str) else ''.join(cell['source'])
        src = src.replace("import matplotlib\nmatplotlib.use('Agg')\n","")
        src = src.replace("import matplotlib.pyplot as plt\n",
                          "import matplotlib.pyplot as plt\nfrom mpl_toolkits.mplot3d import Axes3D\n")
        cell['source'] = src
        print("  [OK] cell 17a81e93: removed Agg, added Axes3D")
        break

# 1b) 3D PCA source
PCA3D = r"""# ── [Step 1] PCA 3D 시각화 ──────────────────────────────────────────────────
print("PCA 3D 시각화...")
fig = plt.figure(figsize=(14, 10))
ax3 = fig.add_subplot(111, projection='3d')
s3 = max(1, len(X_unlab_pca)//3000)
ax3.scatter(X_unlab_pca[::s3,0], X_unlab_pca[::s3,1], X_unlab_pca[::s3,2],
            alpha=0.07, s=5, c='#90CAF9', label=f'Unlabeled(1/{s3})')
ax3.scatter(X_lab_pca[mask_normal,0], X_lab_pca[mask_normal,1], X_lab_pca[mask_normal,2],
            alpha=0.5, s=20, c='#4CAF50', label=f'양품({mask_normal.sum()})', zorder=3)
ax3.scatter(X_lab_pca[mask_defect,0], X_lab_pca[mask_defect,1], X_lab_pca[mask_defect,2],
            alpha=0.95, s=150, c='#F44336', marker='*', label=f'불량({mask_defect.sum()})', zorder=4)
ax3.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)'); ax3.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)')
ax3.set_zlabel(f'PC3 ({evr[2]*100:.1f}%)')
ax3.set_title('PCA 3D — Unlabeled(배경)+Labeled(전경)', fontsize=13); ax3.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR/'NB08_fig1b_pca3d.png', dpi=150, bbox_inches='tight')
plt.show(); print("PCA 3D 저장 완료")
print("불량 3D 범위:")
for nm, col in [('PC1',0),('PC2',1),('PC3',2)]:
    d_mn = X_lab_pca[mask_defect,col].min(); d_mx = X_lab_pca[mask_defect,col].max()
    n_mn = X_lab_pca[mask_normal,col].min(); n_mx = X_lab_pca[mask_normal,col].max()
    print(f"  {nm}: 불량[{d_mn:.2f},{d_mx:.2f}]  양품[{n_mn:.2f},{n_mx:.2f}]")"""

# 1c) 3D UMAP source
UMAP3D = r"""# ── [Step 1] UMAP 3D 시각화 ─────────────────────────────────────────────────
if HAS_UMAP:
    from time import time as _t
    print("UMAP 3D fitting (n_components=3)...")
    t0 = _t()
    r3d = umap_lib.UMAP(n_components=3, random_state=SEED, n_neighbors=15, min_dist=0.1, verbose=False)
    X_u3 = r3d.fit_transform(np.vstack([X_unlab_sc[idx_u], X_lab_sc]))
    U3_unlab = X_u3[:N_UMAP]; U3_lab = X_u3[N_UMAP:]
    print(f"  done in {_t()-t0:.1f}s")
    fig = plt.figure(figsize=(14,10))
    ax3 = fig.add_subplot(111, projection='3d')
    s3 = max(1, N_UMAP//2000)
    ax3.scatter(U3_unlab[::s3,0], U3_unlab[::s3,1], U3_unlab[::s3,2], alpha=0.07, s=5, c='#90CAF9')
    ax3.scatter(U3_lab[mask_normal,0], U3_lab[mask_normal,1], U3_lab[mask_normal,2],
                alpha=0.5, s=20, c='#4CAF50', label=f'양품({mask_normal.sum()})', zorder=3)
    ax3.scatter(U3_lab[mask_defect,0], U3_lab[mask_defect,1], U3_lab[mask_defect,2],
                alpha=0.95, s=150, c='#F44336', marker='*', label=f'불량({mask_defect.sum()})', zorder=4)
    ax3.set_title('UMAP 3D — Unlabeled(배경)+Labeled(전경)', fontsize=13); ax3.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR/'NB08_fig2b_umap3d.png', dpi=150, bbox_inches='tight')
    plt.show(); print("UMAP 3D 저장 완료")
    for nm, col in [('UMAP1',0),('UMAP2',1),('UMAP3',2)]:
        print(f"  {nm} 불량: [{U3_lab[mask_defect,col].min():.2f},{U3_lab[mask_defect,col].max():.2f}]")
else:
    U3_unlab = U3_lab = None; print("UMAP not available")"""

# Insert in reverse order (last → first) to preserve indices
i051 = find_i(c8,'051d859d'); c8.insert(i051+1, cc(UMAP3D))
i34c = find_i(c8,'34c4dd7e'); c8.insert(i34c+1, cc(PCA3D))
i2bd = find_i(c8,'2bd03c39'); c8.insert(i2bd+1, cc('%matplotlib inline'))
save_nb(nb08, NB_DIR/'08_pseudo_labeling.ipynb')
print("  [OK] NB08 saved with inline + 3D viz\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Common cell templates for method notebooks
# ══════════════════════════════════════════════════════════════════════════════

T_SETUP = """\
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp
from time import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.cluster import MiniBatchKMeans

try:
    import umap as umap_lib; HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from pyod.models.hbos import HBOS
    from pyod.models.copod import COPOD
    from pyod.models.knn import KNN as PyODKNN
    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False

SEED = 42
np.random.seed(SEED)
BASE = Path(r'E:\\\\IAAI_Term_project')
FIG_DIR = BASE / 'results/figures'
TAB_DIR = BASE / 'results/tables'
FIG_DIR.mkdir(exist_ok=True, parents=True)
TAB_DIR.mkdir(exist_ok=True, parents=True)

def setup_korean_font():
    import matplotlib.font_manager as fm
    for f in fm.fontManager.ttflist:
        if 'Malgun' in f.name:
            plt.rcParams['font.family'] = f.name; break
    plt.rcParams['axes.unicode_minus'] = False

sns.set_theme(style='whitegrid')
setup_korean_font()
print(f"Setup OK | pyod:{HAS_PYOD} | umap:{HAS_UMAP}")"""

T_DATA = """\
print("Loading data...")
lab   = pd.read_csv(BASE / 'data/raw/labeled_data.csv')
lab['label'] = (lab['PassOrFail'] == 'N').astype(int)
unlab = pd.read_csv(BASE / 'data/raw/unlabeled_data.csv')

ZERO_VAR = ['Mold_Temperature_1','Mold_Temperature_2','Mold_Temperature_5',
    'Mold_Temperature_6','Mold_Temperature_7','Mold_Temperature_8',
    'Mold_Temperature_9','Mold_Temperature_10','Mold_Temperature_11',
    'Mold_Temperature_12','Barrel_Temperature_7']
META = {'_id','Unnamed: 0','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL',
        'PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason','ERR_FACT_QTY','label'}

eq_vals = set(lab['EQUIP_NAME'].unique())
cn7rg3  = [p for p in lab['PART_NAME'].unique() if 'CN7' in p or 'RG3' in p]
unf     = unlab[unlab['EQUIP_NAME'].isin(eq_vals) & unlab['PART_NAME'].isin(cn7rg3)].reset_index(drop=True)

fcols = [c for c in lab.columns if c not in META and c not in ZERO_VAR
         and lab[c].dtype in ['float64','int64','float32','int32']]
X_lab = lab[fcols].values.astype(float); y_lab = lab['label'].values
mask_n = y_lab==0; mask_d = y_lab==1
X_unl  = unf[[c for c in fcols if c in unf.columns]].fillna(0).values.astype(float)
DR     = y_lab.mean()

sc = StandardScaler(); Xu = sc.fit_transform(X_unl); Xl = sc.transform(X_lab)
print(f"Labeled  : {len(X_lab):,} (불량 {mask_d.sum()}, 양품 {mask_n.sum()})")
print(f"Unlabeled: {len(X_unl):,} (CN7/RG3 필터) | DR={DR*100:.4f}%")
print(f"Features : {fcols}")"""

T_PCA = """\
pca = PCA(n_components=3, random_state=SEED)
pca.fit(Xu); Xu_p = pca.transform(Xu); Xl_p = pca.transform(Xl)
evr = pca.explained_variance_ratio_
print(f"PCA: PC1={evr[0]*100:.1f}%  PC2={evr[1]*100:.1f}%  PC3={evr[2]*100:.1f}%  cum={evr[:3].sum()*100:.1f}%")

# ── 2D ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(18,5))
fig.suptitle(f'{NB_NAME} — PCA 2D (데이터 구조)', fontsize=13)
for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
    s=max(1,len(Xu_p)//4000)
    ax.scatter(Xu_p[::s,i],Xu_p[::s,j],alpha=0.08,s=5,c='#90CAF9',label='Unlabeled')
    ax.scatter(Xl_p[mask_n,i],Xl_p[mask_n,j],alpha=0.4,s=15,c='#4CAF50',label=f'양품({mask_n.sum()})',zorder=3)
    ax.scatter(Xl_p[mask_d,i],Xl_p[mask_d,j],alpha=0.9,s=80,c='#F44336',marker='*',label=f'불량({mask_d.sum()})',zorder=4)
    ax.set_xlabel(f'PC{i+1}({evr[i]*100:.0f}%)'); ax.set_ylabel(f'PC{j+1}({evr[j]*100:.0f}%)')
    if i==0 and j==1: ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_pca2d.png',dpi=150,bbox_inches='tight')
plt.show(); print("PCA 2D 저장 완료")

# ── 3D ──────────────────────────────────────────────────────────────────────
fig=plt.figure(figsize=(13,10)); ax3=fig.add_subplot(111,projection='3d')
s=max(1,len(Xu_p)//3000)
ax3.scatter(Xu_p[::s,0],Xu_p[::s,1],Xu_p[::s,2],alpha=0.07,s=5,c='#90CAF9')
ax3.scatter(Xl_p[mask_n,0],Xl_p[mask_n,1],Xl_p[mask_n,2],alpha=0.5,s=20,c='#4CAF50',label=f'양품({mask_n.sum()})',zorder=3)
ax3.scatter(Xl_p[mask_d,0],Xl_p[mask_d,1],Xl_p[mask_d,2],alpha=0.95,s=120,c='#F44336',marker='*',label=f'불량({mask_d.sum()})',zorder=4)
ax3.set_xlabel(f'PC1({evr[0]*100:.0f}%)'); ax3.set_ylabel(f'PC2({evr[1]*100:.0f}%)'); ax3.set_zlabel(f'PC3({evr[2]*100:.0f}%)')
ax3.set_title(f'{NB_NAME} — PCA 3D',fontsize=13); ax3.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_pca3d.png',dpi=150,bbox_inches='tight')
plt.show(); print("PCA 3D 저장 완료")

# ── 불량 PCA 위치 분석 ───────────────────────────────────────────────────────
print("\\n▶ 불량의 PCA 공간 위치:")
for nm,col in [('PC1',0),('PC2',1),('PC3',2)]:
    d_r = [Xl_p[mask_d,col].min(), Xl_p[mask_d,col].max()]
    n_r = [Xl_p[mask_n,col].min(), Xl_p[mask_n,col].max()]
    overlap = np.sum((Xl_p[mask_d,col]>=n_r[0])&(Xl_p[mask_d,col]<=n_r[1]))
    print(f"  {nm}: 불량[{d_r[0]:.2f},{d_r[1]:.2f}]  양품[{n_r[0]:.2f},{n_r[1]:.2f}]  불량 양품범위내: {overlap}/{mask_d.sum()}")"""

T_UMAP = """\
N_U = 6000
rng_u = np.random.RandomState(SEED)
idx_u = rng_u.choice(len(Xu), N_U, replace=False)
Xcomb = np.vstack([Xu[idx_u], Xl])

if HAS_UMAP:
    # ── UMAP 2D ─────────────────────────────────────────────────────────────
    print("UMAP 2D fitting..."); t0=time()
    r2 = umap_lib.UMAP(n_components=2,random_state=SEED,n_neighbors=15,min_dist=0.1,verbose=False)
    U2 = r2.fit_transform(Xcomb); U2u=U2[:N_U]; U2l=U2[N_U:]
    print(f"  done in {time()-t0:.1f}s")
    fig,ax=plt.subplots(figsize=(10,8))
    ax.scatter(U2u[:,0],U2u[:,1],alpha=0.1,s=6,c='#90CAF9',label=f'Unlabeled({N_U})')
    ax.scatter(U2l[mask_n,0],U2l[mask_n,1],alpha=0.6,s=20,c='#4CAF50',label=f'양품({mask_n.sum()})',zorder=3)
    ax.scatter(U2l[mask_d,0],U2l[mask_d,1],alpha=0.95,s=100,c='#F44336',marker='*',label=f'불량({mask_d.sum()})',zorder=4)
    ax.set_title(f'{NB_NAME} — UMAP 2D'); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f'{NB_PRE}_fig_umap2d.png',dpi=150,bbox_inches='tight')
    plt.show(); print("UMAP 2D 저장 완료")

    # ── UMAP 3D ─────────────────────────────────────────────────────────────
    print("UMAP 3D fitting..."); t0=time()
    r3 = umap_lib.UMAP(n_components=3,random_state=SEED,n_neighbors=15,min_dist=0.1,verbose=False)
    U3 = r3.fit_transform(Xcomb); U3u=U3[:N_U]; U3l=U3[N_U:]
    print(f"  done in {time()-t0:.1f}s")
    fig=plt.figure(figsize=(13,10)); ax3=fig.add_subplot(111,projection='3d')
    s=max(1,N_U//2000)
    ax3.scatter(U3u[::s,0],U3u[::s,1],U3u[::s,2],alpha=0.07,s=5,c='#90CAF9')
    ax3.scatter(U3l[mask_n,0],U3l[mask_n,1],U3l[mask_n,2],alpha=0.5,s=20,c='#4CAF50',label=f'양품({mask_n.sum()})',zorder=3)
    ax3.scatter(U3l[mask_d,0],U3l[mask_d,1],U3l[mask_d,2],alpha=0.95,s=120,c='#F44336',marker='*',label=f'불량({mask_d.sum()})',zorder=4)
    ax3.set_title(f'{NB_NAME} — UMAP 3D',fontsize=13); ax3.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f'{NB_PRE}_fig_umap3d.png',dpi=150,bbox_inches='tight')
    plt.show(); print("UMAP 3D 저장 완료")
    # 불량 분산 분석
    dc = U2l[mask_d].mean(axis=0); nc = U2l[mask_n].mean(axis=0)
    d_sp = np.sqrt(((U2l[mask_d]-dc)**2).sum(axis=1)).mean()
    n_sp = np.sqrt(((U2l[mask_n]-nc)**2).sum(axis=1)).mean()
    print(f"\\n▶ UMAP 2D 군집 퍼짐: 불량={d_sp:.3f}  양품={n_sp:.3f}  비율={d_sp/n_sp:.2f}x")
    print(f"  → {'불량이 산재 — 단일 클러스터 없음' if d_sp/n_sp>1.2 else '불량이 비교적 군집'}")
else:
    U2l = U3l = U2u = U3u = None; print("UMAP not available")"""

T_SCORE_VIZ = """\
# 정규화
ns = (sc_lab - sc_lab.min())/(sc_lab.max()-sc_lab.min()+1e-9)

# ── 이상 점수 PCA 2D overlay ─────────────────────────────────────────────────
fig,axes=plt.subplots(1,3,figsize=(18,5))
fig.suptitle(f'{NB_NAME} — 이상 점수 PCA 2D overlay',fontsize=13)
for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
    s=max(1,len(Xu_p)//3000)
    ax.scatter(Xu_p[::s,i],Xu_p[::s,j],alpha=0.05,s=5,c='#BDBDBD')
    sc_=ax.scatter(Xl_p[:,i],Xl_p[:,j],c=ns,cmap='RdYlGn_r',alpha=0.8,s=30,zorder=3,vmin=0,vmax=1)
    ax.scatter(Xl_p[mask_d,i],Xl_p[mask_d,j],alpha=0.95,s=130,c='black',marker='*',zorder=5,label='실제 불량')
    ax.set_xlabel(f'PC{i+1}'); ax.set_ylabel(f'PC{j+1}')
    if i==0 and j==1: plt.colorbar(sc_,ax=ax,label='이상점수'); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_score_pca2d.png',dpi=150,bbox_inches='tight')
plt.show(); print("이상점수 PCA 2D 저장 완료")

# ── 이상 점수 PCA 3D overlay ─────────────────────────────────────────────────
fig=plt.figure(figsize=(13,10)); ax3=fig.add_subplot(111,projection='3d')
s=max(1,len(Xu_p)//2000)
ax3.scatter(Xu_p[::s,0],Xu_p[::s,1],Xu_p[::s,2],alpha=0.05,s=5,c='#BDBDBD')
p3=ax3.scatter(Xl_p[:,0],Xl_p[:,1],Xl_p[:,2],c=ns,cmap='RdYlGn_r',alpha=0.8,s=30,zorder=3,vmin=0,vmax=1)
ax3.scatter(Xl_p[mask_d,0],Xl_p[mask_d,1],Xl_p[mask_d,2],alpha=0.95,s=150,c='black',marker='*',zorder=5,label='실제 불량')
plt.colorbar(p3,ax=ax3,label='이상점수',shrink=0.5)
ax3.set_title(f'{NB_NAME} — 이상점수 PCA 3D',fontsize=13); ax3.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_score_pca3d.png',dpi=150,bbox_inches='tight')
plt.show(); print("이상점수 PCA 3D 저장 완료")

# ── 이상 점수 UMAP overlay ──────────────────────────────────────────────────
if HAS_UMAP and U2l is not None:
    fig,ax=plt.subplots(figsize=(10,8))
    ax.scatter(U2u[:,0],U2u[:,1],alpha=0.05,s=5,c='#BDBDBD')
    sc_=ax.scatter(U2l[:,0],U2l[:,1],c=ns,cmap='RdYlGn_r',alpha=0.8,s=30,zorder=3,vmin=0,vmax=1)
    ax.scatter(U2l[mask_d,0],U2l[mask_d,1],alpha=0.95,s=120,c='black',marker='*',zorder=5,label='실제 불량')
    plt.colorbar(sc_,ax=ax,label='이상점수')
    ax.set_title(f'{NB_NAME} — 이상점수 UMAP 2D',fontsize=13); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f'{NB_PRE}_fig_score_umap2d.png',dpi=150,bbox_inches='tight')
    plt.show(); print("이상점수 UMAP 2D 저장 완료")

    fig=plt.figure(figsize=(13,10)); ax3=fig.add_subplot(111,projection='3d')
    s=max(1,N_U//2000)
    ax3.scatter(U3u[::s,0],U3u[::s,1],U3u[::s,2],alpha=0.05,s=5,c='#BDBDBD')
    p3=ax3.scatter(U3l[:,0],U3l[:,1],U3l[:,2],c=ns,cmap='RdYlGn_r',alpha=0.8,s=30,zorder=3,vmin=0,vmax=1)
    ax3.scatter(U3l[mask_d,0],U3l[mask_d,1],U3l[mask_d,2],alpha=0.95,s=150,c='black',marker='*',zorder=5,label='실제 불량')
    plt.colorbar(p3,ax=ax3,label='이상점수',shrink=0.5)
    ax3.set_title(f'{NB_NAME} — 이상점수 UMAP 3D',fontsize=13); ax3.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f'{NB_PRE}_fig_score_umap3d.png',dpi=150,bbox_inches='tight')
    plt.show(); print("이상점수 UMAP 3D 저장 완료")"""

T_DIST = """\
roc = roc_auc_score(y_lab, sc_lab); pr = average_precision_score(y_lab, sc_lab)
ks, kp = ks_2samp(sc_lab[mask_n], sc_lab[mask_d])
print("="*60)
print(f"{NB_NAME} — 성능 지표")
print("="*60)
print(f"  ROC-AUC : {roc:.4f}")
print(f"  PR-AUC  : {pr:.4f}")
print(f"  KS 통계 : {ks:.4f}  p={kp:.4f}  {'유의(p<0.05)' if kp<0.05 else '비유의'}")
print(f"  분리 등급: {'★★★ 우수(KS>0.4)' if ks>0.4 else '★★ 양호(KS>0.25)' if ks>0.25 else '★ 약함(KS>0.1)' if ks>0.1 else '✗ 분리불가'}")
print()
print(f"  양품({mask_n.sum()}) 점수: mean={sc_lab[mask_n].mean():.4f}  median={np.median(sc_lab[mask_n]):.4f}  std={sc_lab[mask_n].std():.4f}")
print(f"  불량({mask_d.sum()}) 점수: mean={sc_lab[mask_d].mean():.4f}  median={np.median(sc_lab[mask_d]):.4f}  std={sc_lab[mask_d].std():.4f}")
mn_q = np.percentile(sc_lab[mask_n],[25,50,75]); md_q = np.percentile(sc_lab[mask_d],[25,50,75])
print(f"  양품 Q1/Q2/Q3: {mn_q[0]:.4f} / {mn_q[1]:.4f} / {mn_q[2]:.4f}")
print(f"  불량 Q1/Q2/Q3: {md_q[0]:.4f} / {md_q[1]:.4f} / {md_q[2]:.4f}")

fig,axes=plt.subplots(1,2,figsize=(14,5))
fig.suptitle(f'{NB_NAME} — 이상 점수 분포 (ROC={roc:.4f}, KS={ks:.4f})',fontsize=13)
axes[0].hist(sc_lab[mask_n],bins=50,density=True,color='#4CAF50',alpha=0.6,label=f'양품({mask_n.sum()})')
axes[0].hist(sc_lab[mask_d],bins=20,density=True,color='#F44336',alpha=0.8,label=f'불량({mask_d.sum()})')
axes[0].set_xlabel('이상 점수'); axes[0].set_ylabel('밀도'); axes[0].legend()
axes[1].boxplot([sc_lab[mask_n],sc_lab[mask_d]],labels=['양품','불량'],patch_artist=True,
                boxprops={'facecolor':'#E3F2FD'},medianprops={'color':'red','linewidth':2})
axes[1].set_ylabel('이상 점수'); axes[1].set_title(f'Boxplot (KS p={kp:.4f})')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_dist.png',dpi=150,bbox_inches='tight')
plt.show(); print("분포 저장 완료")"""

T_FEAT = """\
print("="*60); print("피처별 이상 점수 상관 분석"); print("="*60)
corrs = [(fcols[i], np.corrcoef(X_lab[:,i], sc_lab)[0,1]) for i in range(len(fcols))]
corr_df = pd.DataFrame(corrs, columns=['Feature','Corr'])
corr_df = corr_df.reindex(corr_df['Corr'].abs().sort_values(ascending=False).index).reset_index(drop=True)
print(f"\\n  {'순위':<4} {'피처':<35} {'Pearson r':>10}  {'방향'}")
print("  "+"-"*55)
for rank,row in corr_df.iterrows():
    if abs(row['Corr'])<0.03: break
    print(f"  {rank+1:<4} {row['Feature']:<35} {row['Corr']:>+10.4f}  {'이상점수↑=불량증가' if row['Corr']>0 else '이상점수↑=불량감소'}")

top15=corr_df.head(15)
fig,ax=plt.subplots(figsize=(10,7))
colors=['#F44336' if v>0 else '#2196F3' for v in top15['Corr']]
ax.barh(top15['Feature'][::-1],top15['Corr'][::-1],color=colors[::-1],edgecolor='white')
ax.axvline(0,color='black',lw=0.8)
ax.set_xlabel('이상 점수와의 Pearson 상관계수')
ax.set_title(f'{NB_NAME} — 피처별 이상 점수 상관도(Top15)')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_feat_corr.png',dpi=150,bbox_inches='tight')
plt.show(); print("피처 상관 저장 완료")"""

T_TOP = """\
n_top = max(int(DR*len(y_lab)),10)
top_idx = np.argsort(sc_lab)[-n_top:][::-1]
hit = y_lab[top_idx].sum()
print("="*60); print(f"상위 {n_top}개 이상 샘플 분석 (DR={DR*100:.2f}%)"); print("="*60)
print(f"\\n▶ 상위 {n_top}개 중 실제 불량: {hit}개")
print(f"  Precision@{n_top}: {hit/n_top:.4f}")
print(f"  Recall   @{n_top}: {hit/mask_d.sum():.4f} ({hit}/{mask_d.sum()}개 탐지)")

comp=pd.DataFrame({
    f'Top{n_top}이상 평균': X_lab[top_idx].mean(axis=0),
    '불량 평균'           : X_lab[mask_d].mean(axis=0),
    '양품 평균'           : X_lab[mask_n].mean(axis=0),
},index=fcols)
comp['방향일치'] = ((comp[f'Top{n_top}이상 평균']-comp['양품 평균'])*(comp['불량 평균']-comp['양품 평균'])>0)
n_match = comp['방향일치'].sum()
print(f"\\n▶ 방향일치도: {n_match}/25 피처 = {n_match/25*100:.0f}% (50%=무작위, 100%=완전일치)")

print(f"\\n▶ 피처별 비교 (|차이|>50% 피처):")
print(f"  {'피처':<32} {'Top이상평균':>10} {'불량평균':>10} {'양품평균':>10}  {'일치'}")
print("  "+"-"*70)
for feat in fcols:
    tv=comp.loc[feat,f'Top{n_top}이상 평균']; dv=comp.loc[feat,'불량 평균']; nv=comp.loc[feat,'양품 평균']
    diff_pct = abs(tv-dv)/(abs(dv)+1e-9)*100
    if diff_pct>50:
        m='✓' if comp.loc[feat,'방향일치'] else '✗'
        print(f"  {feat:<32} {tv:10.2f} {dv:10.2f} {nv:10.2f}   {m}")

# Top 이상 점수 vs 불량/양품 분포 (top-8 피처)
top8=corr_df.head(8)['Feature'].tolist()
fig,axes=plt.subplots(2,4,figsize=(18,8))
fig.suptitle(f'{NB_NAME} — Top-8 피처 분포: 상위이상 vs Labeled',fontsize=12)
for ax,feat in zip(axes.flatten(),top8):
    fi=fcols.index(feat)
    ax.hist(X_lab[mask_n,fi],bins=30,alpha=0.4,color='#4CAF50',density=True,label='양품')
    ax.hist(X_lab[mask_d,fi],bins=10,alpha=0.7,color='#F44336',density=True,label='불량')
    ax.hist(X_lab[top_idx,fi],bins=20,alpha=0.35,color='#FF9800',density=True,label=f'Top{n_top}이상')
    ax.set_title(feat,fontsize=8); ax.tick_params(labelsize=7)
    if feat==top8[0]: ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_top_anomaly.png',dpi=150,bbox_inches='tight')
plt.show(); print("상위 이상 분포 저장 완료")"""

T_ROC_PR = """\
from sklearn.metrics import roc_curve, precision_recall_curve
fpr,tpr,_ = roc_curve(y_lab,sc_lab)
prec_c,rec_c,_ = precision_recall_curve(y_lab,sc_lab)
roc=roc_auc_score(y_lab,sc_lab); pr=average_precision_score(y_lab,sc_lab)
fig,axes=plt.subplots(1,2,figsize=(14,5))
fig.suptitle(f'{NB_NAME} — ROC / PR 곡선',fontsize=13)
axes[0].plot(fpr,tpr,'#2196F3',lw=2,label=f'ROC (AUC={roc:.4f})')
axes[0].plot([0,1],[0,1],'--',color='gray',lw=1)
axes[0].fill_between(fpr,tpr,alpha=0.12,color='#2196F3')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve'); axes[0].legend(fontsize=10)
axes[1].plot(rec_c,prec_c,'#F44336',lw=2,label=f'PR (AP={pr:.4f})')
axes[1].axhline(DR,color='gray',ls='--',lw=1,label=f'Baseline DR={DR*100:.2f}%')
axes[1].fill_between(rec_c,prec_c,alpha=0.12,color='#F44336')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve'); axes[1].legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_roc_pr.png',dpi=150,bbox_inches='tight')
plt.show(); print("ROC/PR 곡선 저장 완료")"""

T_VIOLIN = """\
fig,axes=plt.subplots(1,2,figsize=(13,6))
fig.suptitle(f'{NB_NAME} — 이상 점수 심층 분포',fontsize=13)
vp=axes[0].violinplot([sc_lab[mask_n],sc_lab[mask_d]],showmedians=True,showextrema=True)
for pc,clr in zip(vp['bodies'],['#4CAF50','#F44336']):
    pc.set_facecolor(clr); pc.set_alpha(0.65)
vp['cmedians'].set_colors(['#388E3C','#C62828']); vp['cmedians'].set_linewidth(2)
axes[0].set_xticks([1,2]); axes[0].set_xticklabels([f'양품\\n(n={mask_n.sum()})',f'불량\\n(n={mask_d.sum()})'])
axes[0].set_ylabel('이상 점수'); axes[0].set_title('Violin Plot')
for pct in [25,50,75]:
    axes[0].axhline(np.percentile(sc_lab[mask_n],pct),color='#4CAF50',ls=':',lw=0.8)
for data,lbl,clr in [(sc_lab[mask_n],'양품','#4CAF50'),(sc_lab[mask_d],'불량','#F44336')]:
    x=np.sort(data); y=np.arange(1,len(data)+1)/len(data)
    axes[1].plot(x,y,color=clr,lw=2,label=lbl)
axes[1].set_xlabel('이상 점수'); axes[1].set_ylabel('누적 비율')
axes[1].set_title('Score ECDF (누적분포)'); axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_violin.png',dpi=150,bbox_inches='tight')
plt.show()"""

T_THRESHOLD = """\
thrs=np.percentile(sc_lab,np.arange(50,100,2))
res=[]
for thr in thrs:
    pred=(sc_lab>=thr).astype(int)
    tp=((pred==1)&(y_lab==1)).sum(); fp=((pred==1)&(y_lab==0)).sum()
    fn=((pred==0)&(y_lab==1)).sum()
    pr_t=tp/(tp+fp+1e-9); rc_t=tp/(tp+fn+1e-9)
    f1=2*pr_t*rc_t/(pr_t+rc_t+1e-9)
    res.append({'thr':thr,'Flagged':pred.sum(),'Prec':pr_t,'Rec':rc_t,'F1':f1,'TP':tp})
res_df=pd.DataFrame(res)
fig,axes=plt.subplots(1,2,figsize=(14,5))
fig.suptitle(f'{NB_NAME} — 임계값 분석',fontsize=13)
axes[0].plot(res_df['Rec'],res_df['Prec'],'-o',ms=3,color='#673AB7',lw=1.5)
hp=res_df[res_df['Prec']>=0.99]
if len(hp):
    r=hp.iloc[-1]
    axes[0].plot(r['Rec'],r['Prec'],'r*',ms=14,zorder=5,label=f"Prec≥0.99→Rec={r['Rec']:.3f}(TP={int(r['TP'])})")
axes[0].axhline(0.99,color='red',ls='--',lw=0.8,alpha=0.7)
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Trade-off (임계값 이동)'); axes[0].legend(fontsize=9)
axes[1].plot(res_df['Flagged'],res_df['Rec'],'-o',ms=3,color='#4CAF50',label='Recall')
axes[1].plot(res_df['Flagged'],res_df['Prec'],'-o',ms=3,color='#F44336',label='Precision')
axes[1].plot(res_df['Flagged'],res_df['F1'],'-o',ms=3,color='#2196F3',label='F1')
axes[1].set_xlabel('경보 발령 수 (Flagged)'); axes[1].set_title('경보 수 vs 지표'); axes[1].legend()
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_threshold.png',dpi=150,bbox_inches='tight')
plt.show()
best_f1_row=res_df.loc[res_df['F1'].idxmax()]
print(f"\\n▶ 최적 F1 임계값: Flagged={int(best_f1_row['Flagged'])}  Prec={best_f1_row['Prec']:.4f}  Rec={best_f1_row['Rec']:.4f}  F1={best_f1_row['F1']:.4f}")"""

T_PARALLEL = """\
top6f=corr_df.head(6)['Feature'].tolist(); top6f=[f for f in top6f if f in fcols][:6]
n_top_p=max(int(DR*len(y_lab))*3,30)
top_idx_p=np.argsort(sc_lab)[-n_top_p:]
rng_p=np.random.RandomState(SEED)
norm_samp=rng_p.choice(np.where(mask_n)[0],min(80,mask_n.sum()),replace=False)
rows_p=[]
for i in list(top_idx_p)+list(np.where(mask_d)[0])+list(norm_samp):
    grp='실제불량' if y_lab[i]==1 else ('상위이상' if i in top_idx_p else '양품샘플')
    rows_p.append({f:X_lab[i,fcols.index(f)] for f in top6f}|{'그룹':grp})
par_df=pd.DataFrame(rows_p)
for f in top6f:
    mn,mx=par_df[f].min(),par_df[f].max()
    par_df[f]=(par_df[f]-mn)/(mx-mn+1e-9)
fig,ax=plt.subplots(figsize=(14,6))
colors_p={'실제불량':'#F44336','상위이상':'#FF9800','양품샘플':'#4CAF50'}
for grp,gdf in par_df.groupby('그룹'):
    for _,row in gdf.iterrows():
        ax.plot(top6f,row[top6f].values,color=colors_p[grp],alpha=0.22,lw=1.2)
for grp,clr in colors_p.items():
    gdf=par_df[par_df['그룹']==grp]
    if len(gdf): ax.plot(top6f,gdf[top6f].mean().values,color=clr,lw=3,
                          label=f'{grp}(평균, n={len(gdf)})')
ax.set_title(f'{NB_NAME} — 병렬 좌표계 (정규화, Top-6 피처)',fontsize=12)
ax.legend(fontsize=9); ax.tick_params(axis='x',labelsize=8); ax.set_ylabel('정규화 값(0~1)')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_parallel.png',dpi=150,bbox_inches='tight')
plt.show()
print("병렬 좌표계 저장 완료")"""

T_CORR_HEAT = """\
top10f=corr_df.head(10)['Feature'].tolist()
top10idx_c=[fcols.index(f) for f in top10f]
thr80=np.percentile(sc_lab,80); hi_sc=sc_lab>=thr80
heat=pd.DataFrame({
    '양품':X_lab[mask_n][:,top10idx_c].mean(axis=0),
    '불량':X_lab[mask_d][:,top10idx_c].mean(axis=0),
    '상위20%이상':X_lab[hi_sc][:,top10idx_c].mean(axis=0) if hi_sc.sum()>0 else np.zeros(len(top10f)),
},index=top10f)
heat_z=heat.apply(lambda r:(r-r.mean())/(r.std()+1e-9),axis=1)
q_groups={}
for pct_lo,pct_hi,lbl in [(0,25,'Q1(하위25%)'),(25,50,'Q2'),(50,75,'Q3'),(75,100,'Q4(상위25%)')]:
    lo_=np.percentile(sc_lab,pct_lo); hi_=np.percentile(sc_lab,pct_hi)
    mq=(sc_lab>=lo_)&(sc_lab<hi_) if pct_hi<100 else sc_lab>=lo_
    if mq.sum()>0: q_groups[lbl]=X_lab[mq][:,top10idx_c].mean(axis=0)
q_df=pd.DataFrame(q_groups,index=top10f)
q_z=q_df.apply(lambda r:(r-r.mean())/(r.std()+1e-9),axis=1)
fig,axes=plt.subplots(1,2,figsize=(14,7))
sns.heatmap(heat_z,annot=True,fmt='.2f',cmap='RdBu_r',center=0,ax=axes[0],linewidths=0.5,
            cbar_kws={'label':'z-score'})
axes[0].set_title(f'{NB_NAME} — 피처 평균 히트맵 (양품/불량/상위이상)',fontsize=11); axes[0].set_ylabel('')
sns.heatmap(q_z,annot=True,fmt='.2f',cmap='RdYlGn_r',center=0,ax=axes[1],linewidths=0.5,
            cbar_kws={'label':'z-score'})
axes[1].set_title('이상점수 분위별 피처 평균 (Q1=정상~Q4=이상)',fontsize=11); axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_heatmap.png',dpi=150,bbox_inches='tight')
plt.show()
print("히트맵 저장 완료")"""

# ══════════════════════════════════════════════════════════════════════════════
# Method-specific extra visualizations
# ══════════════════════════════════════════════════════════════════════════════

EXTRA_COPOD = """\
top4f=[f for f in corr_df.head(4)['Feature'].tolist() if f in fcols][:4]
fig,axes=plt.subplots(1,len(top4f),figsize=(16,4))
fig.suptitle('COPOD — 주요 피처 ECDF (Copula 한계 분포 입력)',fontsize=12)
for ax,feat in zip(axes if len(top4f)>1 else [axes],top4f):
    fi=fcols.index(feat)
    for data,lbl,clr,lw_ in [(X_lab[mask_n,fi].astype(float),'양품','#4CAF50',2),
                               (X_lab[mask_d,fi].astype(float),'불량','#F44336',2),
                               (X_unl[:,fi].astype(float),'Unlabeled','#90CAF9',1)]:
        x=np.sort(data); y=np.arange(1,len(x)+1)/len(x)
        ax.plot(x,y,color=clr,lw=lw_,alpha=0.9 if lw_>1 else 0.5,label=lbl)
    ks_s,_=ks_2samp(X_lab[mask_n,fi],X_lab[mask_d,fi])
    ax.set_title(f'{feat}\\nKS={ks_s:.3f}',fontsize=8); ax.legend(fontsize=6)
    ax.set_xlabel('값'); ax.set_ylabel('ECDF')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_ecdf.png',dpi=150,bbox_inches='tight')
plt.show()
print("COPOD ECDF 저장 완료")"""

EXTRA_IF = """\
try:
    depths=np.zeros(len(fcols))
    for tree in clf.estimators_:
        for f in tree.tree_.feature:
            if f>=0 and f<len(fcols): depths[f]+=1
    depths/=max(len(clf.estimators_),1)
    fi_df=pd.DataFrame({'Feature':fcols,'SplitFreq':depths}).sort_values('SplitFreq',ascending=False)
    top5_names=set(corr_df.head(5)['Feature'].tolist())
    cols_fi=['#F44336' if r['Feature'] in top5_names else '#2196F3' for _,r in fi_df.iterrows()]
    feat_corrs=[abs(np.corrcoef(X_lab[:,fcols.index(f)],sc_lab)[0,1]) for f in fcols]

    fig,axes=plt.subplots(1,2,figsize=(16,6))
    axes[0].barh(fi_df['Feature'][::-1],fi_df['SplitFreq'][::-1],color=cols_fi[::-1],edgecolor='white')
    axes[0].set_xlabel('평균 분기 횟수/트리')
    axes[0].set_title('IF 피처 분기 빈도 (빨강=이상점수 상위5)')

    depths_ord=[depths[fcols.index(f)] for f in fcols]
    axes[1].scatter(depths_ord,feat_corrs,alpha=0.7,s=60,color='#673AB7')
    th_d=np.percentile(depths_ord,75); th_c=np.percentile(feat_corrs,75)
    for i,f in enumerate(fcols):
        if depths_ord[i]>th_d or feat_corrs[i]>th_c:
            axes[1].annotate(f,xy=(depths_ord[i],feat_corrs[i]),fontsize=6,alpha=0.8)
    axes[1].set_xlabel('분기 빈도(IF)'); axes[1].set_ylabel('|이상점수 상관계수|')
    axes[1].set_title('분기빈도 vs 이상점수 상관도')
    plt.tight_layout()
    plt.savefig(FIG_DIR/f'{NB_PRE}_fig_feat_split.png',dpi=150,bbox_inches='tight')
    plt.show(); print("IF 피처 분기 빈도 저장 완료")
except Exception as e:
    print(f"분기 빈도 실패: {e}")"""

EXTRA_GMM = """\
from matplotlib.patches import Ellipse as MEll
means_p2=pca.transform(clf.means_)
fig,axes=plt.subplots(1,2,figsize=(16,7))
s_=max(1,len(Xu_p)//3000)
axes[0].scatter(Xu_p[::s_,0],Xu_p[::s_,1],alpha=0.05,s=5,c='#90CAF9')
axes[0].scatter(Xl_p[mask_n,0],Xl_p[mask_n,1],alpha=0.5,s=15,c='#4CAF50',label='양품',zorder=3)
axes[0].scatter(Xl_p[mask_d,0],Xl_p[mask_d,1],alpha=0.95,s=100,c='#F44336',marker='*',label='불량',zorder=4)
try:
    for k in range(best_k):
        cov_k=(clf.covariances_[k] if clf.covariance_type=='full'
               else np.diag(clf.covariances_[k]) if clf.covariance_type=='diag'
               else clf.covariances_[k]*np.eye(Xl.shape[1]))
        cov2=pca.components_[:2]@cov_k@pca.components_[:2].T
        ev,evec=np.linalg.eigh(cov2)
        angle=np.degrees(np.arctan2(evec[1,-1],evec[0,-1]))
        w,h=2*np.sqrt(np.abs(ev))
        ell=MEll(xy=means_p2[k,:2],width=w,height=h,angle=angle,
                  alpha=0.13,color=plt.cm.tab20(k/best_k),lw=1.5,fill=True)
        axes[0].add_patch(ell)
        axes[0].plot(*means_p2[k,:2],'x',ms=7,color=plt.cm.tab20(k/best_k))
except Exception as e:
    print(f"타원 실패: {e}")
axes[0].set_title(f'GMM k={best_k} — 가우시안 타원 (PCA 2D)',fontsize=11); axes[0].legend(fontsize=8)
proba_d=clf.predict_proba(Xl[mask_d])
im=axes[1].imshow(proba_d,aspect='auto',cmap='YlOrRd',vmin=0,vmax=1)
axes[1].set_xlabel(f'GMM 컴포넌트'); axes[1].set_ylabel(f'불량 샘플')
axes[1].set_title('불량 71개의 각 컴포넌트 소속 확률')
plt.colorbar(im,ax=axes[1],label='책임 확률')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_ellipses.png',dpi=150,bbox_inches='tight')
plt.show()
print("GMM 타원 저장 완료")"""

EXTRA_KM = """\
top10f_km=corr_df.head(10)['Feature'].tolist()
top10i_km=[fcols.index(f) for f in top10f_km]
clust_dr_=[]; clust_n_=[]
for k in range(km.n_clusters):
    in_k=labels_l==k; n_t=in_k.sum(); n_d=y_lab[in_k].sum() if n_t>0 else 0
    clust_dr_.append(n_d/n_t if n_t>0 else 0); clust_n_.append(n_t)
sort_i=np.argsort(clust_dr_)[::-1]; top_k_=min(15,km.n_clusters)
tc=sort_i[:top_k_]
cent_feat=np.array([km.cluster_centers_[i,top10i_km] for i in tc])
cent_z=(cent_feat-cent_feat.mean(axis=0))/(cent_feat.std(axis=0)+1e-9)
row_lbls=[f"C{i}({clust_dr_[i]*100:.1f}%)" for i in tc]
fig,axes=plt.subplots(1,2,figsize=(16,6))
sns.heatmap(cent_z,annot=True,fmt='.1f',cmap='RdBu_r',center=0,ax=axes[0],
            xticklabels=top10f_km,yticklabels=row_lbls,linewidths=0.5)
axes[0].set_title(f'K-Means Centroid 히트맵 (불량률 상위{top_k_})',fontsize=10)
axes[0].tick_params(axis='x',labelrotation=30,labelsize=8)
sorted_dr_=[clust_dr_[i] for i in tc]; ns_=[clust_n_[i] for i in tc]
bars_=axes[1].bar(range(top_k_),[d*100 for d in sorted_dr_],
                  color=['#F44336' if d>0 else '#4CAF50' for d in sorted_dr_])
axes[1].axhline(DR*100,color='gray',ls='--',label=f'전체DR={DR*100:.2f}%')
for bar_,n_ in zip(bars_,ns_):
    axes[1].text(bar_.get_x()+bar_.get_width()/2,bar_.get_height()+0.01,f'n={n_}',
                ha='center',va='bottom',fontsize=7)
axes[1].set_xlabel('Cluster(불량률 내림차순)'); axes[1].set_ylabel('불량률(%)')
axes[1].set_title('클러스터별 불량률'); axes[1].legend()
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_centroid_heat.png',dpi=150,bbox_inches='tight')
plt.show()
print("Centroid 히트맵 저장 완료")"""

EXTRA_LOF = """\
from sklearn.neighbors import NearestNeighbors as NN_LOF
N_KDPL=min(5000,len(Xu))
rng_kd=np.random.RandomState(SEED+1)
idx_kd=rng_kd.choice(len(Xu),N_KDPL,replace=False)
nbrs_=NN_LOF(n_neighbors=int(best_nn)+1,n_jobs=-1); nbrs_.fit(Xu[idx_kd])
dists_kd,_=nbrs_.kneighbors(Xu[idx_kd]); k_dists=dists_kd[:,-1]
sorted_kd=np.sort(k_dists)[::-1]
diff2=np.gradient(np.gradient(sorted_kd))
ks_=int(len(diff2)*0.05); ke_=int(len(diff2)*0.5)
knee_i=np.argmax(np.abs(diff2[ks_:ke_]))+ks_

fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].plot(sorted_kd,color='#2196F3',lw=1.5)
axes[0].axvline(knee_i,color='red',ls='--',lw=1.5,
                label=f'Knee≈{sorted_kd[knee_i]:.3f}(→DBSCAN eps 추천)')
axes[0].set_xlabel('포인트(거리 내림차순)'); axes[0].set_ylabel(f'k={int(best_nn)} 최근접거리')
axes[0].set_title('k-distance 그래프 (DBSCAN eps 선택 참고)'); axes[0].legend()

axes[1].hist(sc_lab[mask_n],bins=40,alpha=0.5,color='#4CAF50',density=True,label=f'양품')
axes[1].hist(sc_lab[mask_d],bins=12,alpha=0.8,color='#F44336',density=True,label=f'불량')
axes[1].axvline(np.median(sc_lab[mask_n]),color='#388E3C',ls='--',lw=1.5,
                label=f'양품중앙={np.median(sc_lab[mask_n]):.3f}')
axes[1].axvline(np.median(sc_lab[mask_d]),color='#C62828',ls='--',lw=1.5,
                label=f'불량중앙={np.median(sc_lab[mask_d]):.3f}')
axes[1].set_xlabel('LOF 이상 점수'); axes[1].set_title('LOF 점수 밀도'); axes[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_kdist.png',dpi=150,bbox_inches='tight')
plt.show()
print(f"k-distance Knee ≈ {sorted_kd[knee_i]:.4f} → DBSCAN eps 추천값")"""

EXTRA_HBOS = """\
ncols_h=5; nrows_h=(len(fcols)+ncols_h-1)//ncols_h
fig,axes=plt.subplots(nrows_h,ncols_h,figsize=(20,nrows_h*3))
fig.suptitle('HBOS — 전체 피처 히스토그램 (양품 vs 불량, KS 통계)',fontsize=13)
for i,(ax,feat) in enumerate(zip(axes.flatten(),fcols)):
    ax.hist(Xl[mask_n,i],bins=25,alpha=0.45,color='#4CAF50',density=True,label='양품')
    ax.hist(Xl[mask_d,i],bins=8,alpha=0.75,color='#F44336',density=True,label='불량')
    ks_s,ks_p=ks_2samp(Xl[mask_n,i],Xl[mask_d,i])
    sig='*' if ks_p<0.05 else ''
    ax.set_title(f'{feat}\\nKS={ks_s:.3f}{sig}',fontsize=7)
    ax.tick_params(labelsize=6)
    if i==0: ax.legend(fontsize=6)
for ax in axes.flatten()[len(fcols):]: ax.axis('off')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_all_hist.png',dpi=150,bbox_inches='tight')
plt.show()
print("전체 피처 히스토그램 저장 완료 (*=KS p<0.05 유의)")"""

EXTRA_OCSVM = """\
norm_ctr=Xl[mask_n].mean(axis=0)
eucl_d=np.sqrt(((Xl-norm_ctr)**2).sum(axis=1))
scA_n=(scA-scA.min())/(scA.max()-scA.min()+1e-9)
scB_n=(scB-scB.min())/(scB.max()-scB.min()+1e-9)
fig,axes=plt.subplots(1,3,figsize=(18,5))
fig.suptitle(f'{NB_NAME} — Config 비교 분석',fontsize=13)
axes[0].scatter(scA_n[mask_n],scB_n[mask_n],alpha=0.3,s=15,c='#4CAF50',label='양품')
axes[0].scatter(scA_n[mask_d],scB_n[mask_d],c='#F44336',marker='*',s=100,label='불량',zorder=4)
axes[0].plot([0,1],[0,1],'--',color='gray',lw=1,label='y=x')
axes[0].set_xlabel('Config A 점수(정규화)'); axes[0].set_ylabel('Config B 점수(정규화)')
axes[0].set_title(f'A(ROC={roc_auc_score(y_lab,scA):.4f}) vs B(ROC={roc_auc_score(y_lab,scB):.4f})')
axes[0].legend(fontsize=8)
axes[1].scatter(eucl_d[mask_n],scA[mask_n],alpha=0.3,s=15,c='#4CAF50',label='양품')
axes[1].scatter(eucl_d[mask_d],scA[mask_d],c='#F44336',marker='*',s=100,label='불량',zorder=4)
r_c=np.corrcoef(eucl_d,scA)[0,1]
axes[1].set_xlabel('유클리드거리(양품centroid)'); axes[1].set_ylabel('OC-SVM A Score')
axes[1].set_title(f'유클리드 vs OC-SVM 점수 (r={r_c:.3f})'); axes[1].legend(fontsize=8)
for sc_x,cfg,clr in [(scA,'Config A','#2196F3'),(scB,'Config B','#FF9800')]:
    axes[2].hist(sc_x[mask_n],bins=30,alpha=0.4,density=True,color=clr,label=f'{cfg}:양품')
for sc_x,clr in [(scA,'#2196F3'),(scB,'#FF9800')]:
    for v in sc_x[mask_d]: axes[2].axvline(v,color=clr,alpha=0.5,lw=0.8)
axes[2].set_xlabel('이상 점수'); axes[2].set_title('양품분포 + 불량위치'); axes[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_config_compare.png',dpi=150,bbox_inches='tight')
plt.show()
print("Config A vs B 비교 저장 완료")"""

EXTRA_EE = """\
try:
    from scipy.stats import chi2 as _chi2
    maha_sq=clf.mahalanobis(Xl)
    n_dim=Xl.shape[1]
    sorted_m=np.sort(maha_sq[mask_n])
    chi2_th=_chi2.ppf(np.linspace(0.01,0.99,len(sorted_m)),df=n_dim)
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    fig.suptitle(f'{NB_NAME} — Mahalanobis 거리 분석',fontsize=13)
    axes[0].scatter(chi2_th,sorted_m,alpha=0.5,s=15,color='#2196F3')
    mn_=min(chi2_th.min(),sorted_m.min()); mx_=max(chi2_th.max(),sorted_m.max())
    axes[0].plot([mn_,mx_],[mn_,mx_],'r--',lw=1.5,label='이론선')
    axes[0].set_xlabel(f'chi²({n_dim}) 이론 분위수')
    axes[0].set_ylabel('관측 마할라노비스²(양품)'); axes[0].set_title('Q-Q Plot'); axes[0].legend()
    axes[1].hist(maha_sq[mask_n],bins=40,alpha=0.5,color='#4CAF50',density=True,label=f'양품')
    axes[1].hist(maha_sq[mask_d],bins=10,alpha=0.8,color='#F44336',density=True,label=f'불량')
    th95=np.percentile(maha_sq[mask_n],95)
    axes[1].axvline(th95,color='orange',ls='--',lw=1.5,label=f'양품95%={th95:.2f}')
    axes[1].set_xlabel('마할라노비스 거리²'); axes[1].set_title('거리 분포'); axes[1].legend(fontsize=8)
    top6_ee=[f for f in corr_df.head(6)['Feature'].tolist() if f in fcols][:6]
    top6i_ee=[fcols.index(f) for f in top6_ee]
    pm=X_unl[pseudo_d][:,top6i_ee].mean(axis=0) if pseudo_d.sum()>0 else np.zeros(len(top6_ee))
    dm=X_lab[mask_d][:,top6i_ee].mean(axis=0); nm=X_lab[mask_n][:,top6i_ee].mean(axis=0)
    x_=np.arange(len(top6_ee)); w=0.25
    axes[2].bar(x_-w,pm,w,label='Pseudo-불량',color='#FF9800',alpha=0.8)
    axes[2].bar(x_,dm,w,label='Labeled-불량',color='#F44336',alpha=0.8)
    axes[2].bar(x_+w,nm,w,label='Labeled-양품',color='#4CAF50',alpha=0.8)
    axes[2].set_xticks(x_); axes[2].set_xticklabels(top6_ee,rotation=25,fontsize=8)
    axes[2].set_title('Pseudo vs Labeled 불량 피처 비교'); axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f'{NB_PRE}_fig_mahal.png',dpi=150,bbox_inches='tight')
    plt.show(); print("Mahalanobis Q-Q 저장 완료")
except Exception as e:
    print(f"Q-Q 실패: {e}")"""

EXTRA_KNN = """\
from sklearn.neighbors import NearestNeighbors as NN_KNN2
N_KD2=min(5000,len(Xu))
rng_kd2=np.random.RandomState(SEED+2)
idx_kd2=rng_kd2.choice(len(Xu),N_KD2,replace=False)
nbrs2_=NN_KNN2(n_neighbors=int(best_k_knn)+1,n_jobs=-1); nbrs2_.fit(Xu[idx_kd2])
klab_d,_=nbrs2_.kneighbors(Xl)
mean_d=klab_d[:,1:].mean(axis=1) if klab_d.shape[1]>1 else klab_d[:,0]

fig,axes=plt.subplots(1,3,figsize=(18,5))
fig.suptitle(f'{NB_NAME} — KNN 거리 분포 분석',fontsize=13)
kd_all_=np.sort(klab_d[:,-1])[::-1]
axes[0].plot(kd_all_,color='#2196F3',lw=1.5)
rank_d=np.argsort(klab_d[:,-1])[::-1]
d_ranks=[np.where(rank_d==i)[0][0] for i in np.where(mask_d)[0]]
if d_ranks: axes[0].scatter(d_ranks,kd_all_[d_ranks],c='#F44336',s=40,zorder=4,label='불량',alpha=0.8)
axes[0].set_xlabel('샘플(k-거리 내림차순)'); axes[0].set_ylabel(f'k={best_k_knn} 최근접거리')
axes[0].set_title(f'Sorted k-distance 그래프'); axes[0].legend()

axes[1].hist(mean_d[mask_n],bins=40,alpha=0.5,color='#4CAF50',density=True,label=f'양품')
axes[1].hist(mean_d[mask_d],bins=12,alpha=0.8,color='#F44336',density=True,label=f'불량')
axes[1].axvline(np.median(mean_d[mask_n]),color='#388E3C',ls='--',lw=1.5,
                label=f'양품중앙={np.median(mean_d[mask_n]):.3f}')
axes[1].axvline(np.median(mean_d[mask_d]),color='#C62828',ls='--',lw=1.5,
                label=f'불량중앙={np.median(mean_d[mask_d]):.3f}')
axes[1].set_xlabel('평균 kNN거리'); axes[1].set_title('평균 kNN거리 분포'); axes[1].legend(fontsize=7)

axes[2].scatter(mean_d[mask_n],sc_lab[mask_n],alpha=0.3,s=15,c='#4CAF50',label='양품')
axes[2].scatter(mean_d[mask_d],sc_lab[mask_d],c='#F44336',marker='*',s=100,label='불량',zorder=4)
r_c_=np.corrcoef(mean_d,sc_lab)[0,1]
axes[2].set_xlabel('평균 kNN거리'); axes[2].set_ylabel('KNN 이상점수')
axes[2].set_title(f'거리 vs 이상점수 (r={r_c_:.3f})'); axes[2].legend()
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_knn_dist.png',dpi=150,bbox_inches='tight')
plt.show()
print("KNN 거리 분석 저장 완료")"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Method-specific configs
# ══════════════════════════════════════════════════════════════════════════════

METHODS = [

  { 'id':'08a','pre':'NB08a','name':'COPOD (Copula-Based Outlier Detection)',
    'desc':"""## COPOD — Copula 기반 이상탐지

**원리:** 각 피처의 주변 분포(empirical CDF)와 피처 간 Copula 결합 구조를 모델링.
이상 점수 = 관측값의 copula tail probability (낮을수록 이상).

**장점:** 가정 없는 비모수적 방법, 빠른 계산, 다변량 의존 구조 포착
**학습 데이터:** Unlabeled 71,180행 (CN7/RG3 필터)
**NB08 결과:** ROC=0.8793 — 완전 비지도 기법 중 최고""",
    'sweep': """\
print("COPOD 하이퍼파라미터 탐색 — contamination threshold")
from pyod.models.copod import COPOD
threshs=[0.005,0.0089,0.02,0.05,0.10]
rows=[]
for t in threshs:
    clf=COPOD(contamination=t); clf.fit(Xu)
    sc_=clf.decision_function(Xl)
    roc=roc_auc_score(y_lab,sc_); pr=average_precision_score(y_lab,sc_)
    n_flag=int(t*len(Xu))
    rows.append({'contamination':f'{t*100:.2f}%','n_flagged':n_flag,'ROC':roc,'PR':pr})
    print(f"  contamination={t*100:.2f}%  flagged={n_flag:,}  ROC={roc:.4f}  PR={pr:.4f}")
sw_df=pd.DataFrame(rows); print("\\n",sw_df.to_string(index=False))""",
    'fit': """\
print("COPOD 최적 모델 fitting (contamination=DR)...")
from pyod.models.copod import COPOD
clf=COPOD(contamination=DR); clf.fit(Xu)
sc_lab=clf.decision_function(Xl)
sc_unl=clf.decision_function(Xu)
print(f"  Labeled 이상 점수 범위: [{sc_lab.min():.4f}, {sc_lab.max():.4f}]")
print(f"  Unlabeled 이상 점수 범위: [{sc_unl.min():.4f}, {sc_unl.max():.4f}]")
NB_NAME=NB_NAME; NB_PRE=NB_PRE""",
    'extra': EXTRA_COPOD },

  { 'id':'08b','pre':'NB08b','name':'Isolation Forest (iForest)',
    'desc':"""## Isolation Forest — 격리 트리 기반 이상탐지

**원리:** 랜덤 분기(random split)로 샘플을 격리하는 데 필요한 평균 분기 수.
이상 샘플은 적은 분기로 격리 가능 → 평균 경로 길이가 짧음.

**장점:** 선형 시간복잡도, 고차원에서 안정적, 하이퍼파라미터 민감도 낮음
**학습 데이터:** Unlabeled 71,180행""",
    'sweep': """\
print("Isolation Forest — n_estimators 탐색")
rows=[]
for n_est in [50,100,200,500]:
    clf=IsolationForest(n_estimators=n_est,contamination=DR,random_state=SEED,n_jobs=-1)
    clf.fit(Xu); sc_=-clf.score_samples(Xl)
    roc=roc_auc_score(y_lab,sc_); pr=average_precision_score(y_lab,sc_)
    rows.append({'n_estimators':n_est,'ROC':roc,'PR':pr})
    print(f"  n_estimators={n_est}  ROC={roc:.4f}  PR={pr:.4f}")
sw_df=pd.DataFrame(rows); print("\\n",sw_df.to_string(index=False))

print("\\nIsolation Forest — max_samples 탐색")
rows2=[]
for ms in [256,1024,4096,'auto']:
    clf=IsolationForest(n_estimators=200,max_samples=ms,contamination=DR,random_state=SEED,n_jobs=-1)
    clf.fit(Xu); sc_=-clf.score_samples(Xl)
    roc=roc_auc_score(y_lab,sc_); pr=average_precision_score(y_lab,sc_)
    rows2.append({'max_samples':str(ms),'ROC':roc,'PR':pr})
    print(f"  max_samples={ms}  ROC={roc:.4f}  PR={pr:.4f}")
print(pd.DataFrame(rows2).to_string(index=False))""",
    'fit': """\
clf=IsolationForest(n_estimators=200,contamination=DR,random_state=SEED,n_jobs=-1)
clf.fit(Xu); sc_lab=-clf.score_samples(Xl); sc_unl=-clf.score_samples(Xu)
print(f"IF 최적 모델 (n_estimators=200) fitting 완료")
print(f"  Labeled 점수 범위: [{sc_lab.min():.4f},{sc_lab.max():.4f}]")
# 피처별 평균 격리 깊이 proxy (feature importances)
try:
    depths=np.zeros(Xl.shape[1])
    for tree in clf.estimators_:
        fi=tree.tree_.feature
        for f in fi:
            if f>=0: depths[f]+=1
    depths/=len(clf.estimators_)
    feat_imp=pd.DataFrame({'Feature':fcols,'AvgSplitCount':depths}).sort_values('AvgSplitCount',ascending=False)
    print("\\n▶ 피처별 평균 분기 횟수 (높을수록 자주 분기에 사용):")
    print(feat_imp.head(10).to_string(index=False))
except Exception as e:
    print(f"Feature importance 계산 실패: {e}")""",
    'extra': EXTRA_IF },

  { 'id':'08c','pre':'NB08c','name':'GMM (Gaussian Mixture Model)',
    'desc':"""## GMM — 가우시안 혼합 모델 기반 이상탐지

**원리:** 데이터를 K개 가우시안 혼합 분포로 모델링.
이상 점수 = 해당 샘플의 음의 로그 가능도 (-log likelihood).

**장점:** 확률론적 밀도 추정, BIC/AIC로 최적 k 선택 가능
**학습 데이터:** Unlabeled 71,180행
**핵심 질문:** 어떤 k에서 unlabeled 데이터 구조가 가장 잘 포착되는가?""",
    'sweep': """\
print("GMM — BIC/AIC로 최적 k 탐색 (k=2..25)")
bics=[]; aics=[]; rocs=[]
ks=list(range(2,26))
for k in ks:
    gmm=GaussianMixture(n_components=k,random_state=SEED,n_init=2,max_iter=100)
    gmm.fit(Xu)
    bics.append(gmm.bic(Xu)); aics.append(gmm.aic(Xu))
    sc_=-gmm.score_samples(Xl)
    rocs.append(roc_auc_score(y_lab,sc_))
    if k in [2,5,10,15,20,25]: print(f"  k={k:2d}  BIC={gmm.bic(Xu):,.0f}  AIC={gmm.aic(Xu):,.0f}  ROC={rocs[-1]:.4f}")

best_bic_k=ks[np.argmin(bics)]; best_roc_k=ks[np.argmax(rocs)]
print(f"\\n최적 k (BIC): {best_bic_k}  최적 k (ROC): {best_roc_k}")

fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].plot(ks,bics,'b-o',label='BIC'); axes[0].plot(ks,aics,'r-o',label='AIC')
axes[0].axvline(best_bic_k,color='blue',ls='--',alpha=0.5,label=f'BIC opt k={best_bic_k}')
axes[0].set_xlabel('k'); axes[0].set_title('BIC/AIC'); axes[0].legend()
axes[1].plot(ks,rocs,'g-o',label='ROC-AUC')
axes[1].axvline(best_roc_k,color='green',ls='--',alpha=0.5,label=f'ROC opt k={best_roc_k}')
axes[1].set_xlabel('k'); axes[1].set_title('ROC-AUC by k'); axes[1].legend()
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_bic_roc.png',dpi=150,bbox_inches='tight')
plt.show()""",
    'fit': """\
best_k=max(ks,key=lambda k:rocs[ks.index(k)])
print(f"GMM 최적 k={best_k} (ROC 기준) 재학습...")
clf=GaussianMixture(n_components=best_k,random_state=SEED,n_init=3)
clf.fit(Xu); sc_lab=-clf.score_samples(Xl); sc_unl=-clf.score_samples(Xu)
labels_u=clf.predict(Xu); labels_l=clf.predict(Xl)
print(f"  k={best_k} fitting 완료 | ROC={roc_auc_score(y_lab,sc_lab):.4f}")
print(f"\\n▶ 클러스터별 불량 분포:")
for k in range(best_k):
    in_k=labels_l==k; n_def=y_lab[in_k].sum(); n_tot=in_k.sum()
    if n_tot>0: print(f"  Cluster {k}: {n_tot:4d}개 중 불량 {n_def}개 ({n_def/n_tot*100:.2f}%)")

# 클러스터 PCA 시각화
fig,ax=plt.subplots(figsize=(10,7))
cmap=plt.cm.get_cmap('tab20',best_k)
s=max(1,len(Xu_p)//3000)
for k in range(best_k):
    mk=labels_u==k; ax.scatter(Xu_p[::s][mk[::s],0],Xu_p[::s][mk[::s],1],
                                color=cmap(k),alpha=0.2,s=5,label=f'C{k}' if k<8 else '')
ax.scatter(Xl_p[mask_n,0],Xl_p[mask_n,1],alpha=0.7,s=20,c='#4CAF50',marker='o',label='양품',zorder=3)
ax.scatter(Xl_p[mask_d,0],Xl_p[mask_d,1],alpha=0.95,s=100,c='#F44336',marker='*',label='불량',zorder=4)
ax.set_title(f'GMM k={best_k} 클러스터 (PCA 2D)'); ax.legend(fontsize=6,ncol=3)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_clusters.png',dpi=150,bbox_inches='tight')
plt.show(); print("클러스터 시각화 저장 완료")""",
    'extra': EXTRA_GMM },

  { 'id':'08d','pre':'NB08d','name':'K-Means 거리 기반 이상탐지',
    'desc':"""## K-Means — Centroid 거리 기반 이상탐지

**원리:** MiniBatchKMeans로 클러스터 centroid 학습.
이상 점수 = 가장 가까운 centroid까지의 유클리드 거리.

**NB08 결과:** k=20에서 ROC=0.4757 (Phase7의 ROC=0.47 재현)
**핵심 질문:** 필터링 후에도 성능이 낮은 이유? 최적 k는?""",
    'sweep': """\
print("K-Means — Elbow + Silhouette + ROC by k")
from sklearn.metrics import silhouette_score
ks=[2,5,10,20,30,50,80,100]; inertias=[]; sils=[]; rocs_km=[]
for k in ks:
    km=MiniBatchKMeans(n_clusters=k,random_state=SEED,n_init=5)
    km.fit(Xu)
    inertias.append(km.inertia_)
    try:
        ss=silhouette_score(Xu[::50],km.labels_[::50])
    except: ss=np.nan
    sils.append(ss)
    dists=np.min(np.linalg.norm(Xl[:,None,:]-km.cluster_centers_[None,:,:],axis=2),axis=1)
    roc=roc_auc_score(y_lab,dists); rocs_km.append(roc)
    print(f"  k={k:3d}  inertia={km.inertia_:,.0f}  sil={ss:.4f}  ROC={roc:.4f}")

fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].plot(ks,inertias,'b-o'); axes[0].set_xlabel('k'); axes[0].set_title('Elbow (Inertia)')
axes[1].plot(ks,sils,'g-o'); axes[1].set_xlabel('k'); axes[1].set_title('Silhouette Score')
axes[2].plot(ks,rocs_km,'r-o'); axes[2].set_xlabel('k'); axes[2].set_title('ROC-AUC')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_km_sweep.png',dpi=150,bbox_inches='tight')
plt.show()
best_k_km=ks[np.argmax(rocs_km)]
print(f"\\n최적 k (ROC): {best_k_km}  최대 ROC={max(rocs_km):.4f}")""",
    'fit': """\
km=MiniBatchKMeans(n_clusters=best_k_km,random_state=SEED,n_init=10); km.fit(Xu)
sc_lab=np.min(np.linalg.norm(Xl[:,None,:]-km.cluster_centers_[None,:,:],axis=2),axis=1)
sc_unl=np.min(np.linalg.norm(Xu[:,None,:]-km.cluster_centers_[None,:,:],axis=2),axis=1)
labels_u=km.predict(Xu); labels_l=km.predict(Xl)
print(f"K-Means k={best_k_km} 완료 | ROC={roc_auc_score(y_lab,sc_lab):.4f}")

# 클러스터별 불량 분포
print("\\n▶ 클러스터별 불량:")
for k in range(min(best_k_km,15)):
    in_k=labels_l==k; n_d=y_lab[in_k].sum(); n_t=in_k.sum()
    bar='█'*int(n_d/mask_d.sum()*20) if n_t>0 else ''
    if n_t>0: print(f"  C{k:2d}: {n_t:4d}개 | 불량 {n_d}  ({n_d/n_t*100:.1f}%)  {bar}")

# PCA 클러스터 시각화
fig,axes=plt.subplots(1,2,figsize=(16,6))
cmap=plt.cm.get_cmap('tab20',best_k_km)
s=max(1,len(Xu_p)//3000)
for k in range(best_k_km):
    mk=labels_u==k
    axes[0].scatter(Xu_p[::s][mk[::s],0],Xu_p[::s][mk[::s],1],color=cmap(k),alpha=0.2,s=5)
axes[0].scatter(Xl_p[mask_d,0],Xl_p[mask_d,1],c='black',marker='*',s=100,label='불량',zorder=4)
axes[0].set_title(f'K-Means k={best_k_km} 클러스터 (PCA2D)')

ns_=(sc_lab-sc_lab.min())/(sc_lab.max()-sc_lab.min()+1e-9)
sc2=axes[1].scatter(Xl_p[:,0],Xl_p[:,1],c=ns_,cmap='RdYlGn_r',s=30,alpha=0.8,zorder=3,vmin=0,vmax=1)
axes[1].scatter(Xl_p[mask_d,0],Xl_p[mask_d,1],c='black',marker='*',s=100,label='불량',zorder=5)
plt.colorbar(sc2,ax=axes[1],label='거리 점수')
axes[1].set_title('Centroid 거리 이상 점수 (PCA2D)')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_km_viz.png',dpi=150,bbox_inches='tight')
plt.show(); print("K-Means 시각화 저장 완료")""",
    'extra': EXTRA_KM },

  { 'id':'08e','pre':'NB08e','name':'LOF (Local Outlier Factor)',
    'desc':"""## LOF — 지역 밀도 기반 이상탐지

**원리:** 각 샘플 주변 k개 이웃의 local reachability density를 비교.
이웃보다 밀도가 낮으면 이상. LOF score > 1이면 이상.

**설정:** novelty=True (unseen 데이터 스코어링 가능)
**학습:** Unlabeled 10K 서브샘플 (메모리 제약)""",
    'sweep': """\
print("LOF — n_neighbors 탐색")
N_LOF=10000
rng_l=np.random.RandomState(SEED)
idx_l=rng_l.choice(len(Xu),N_LOF,replace=False)
Xu_lof=Xu[idx_l]
rows=[]
for nn in [5,10,20,30,50]:
    lof=LocalOutlierFactor(n_neighbors=nn,novelty=True,contamination=DR,n_jobs=-1)
    lof.fit(Xu_lof)
    sc_=-lof.score_samples(Xl)
    roc=roc_auc_score(y_lab,sc_); pr=average_precision_score(y_lab,sc_)
    rows.append({'n_neighbors':nn,'ROC':roc,'PR':pr})
    print(f"  n_neighbors={nn}  ROC={roc:.4f}  PR={pr:.4f}")
sw_df=pd.DataFrame(rows); print("\\n",sw_df.to_string(index=False))
best_nn=sw_df.loc[sw_df['ROC'].idxmax(),'n_neighbors']
print(f"\\n최적 n_neighbors={best_nn}")""",
    'fit': """\
lof=LocalOutlierFactor(n_neighbors=int(best_nn),novelty=True,contamination=DR,n_jobs=-1)
lof.fit(Xu_lof)
sc_lab=-lof.score_samples(Xl); sc_unl=-lof.score_samples(Xu_lof)
print(f"LOF (n_neighbors={best_nn}) 완료 | ROC={roc_auc_score(y_lab,sc_lab):.4f}")
print(f"  Labeled 점수 범위: [{sc_lab.min():.4f},{sc_lab.max():.4f}]")
print(f"  양품 LOF median={np.median(sc_lab[mask_n]):.4f}  불량 LOF median={np.median(sc_lab[mask_d]):.4f}")
print(f"\\n▶ LOF 해석: 값>1이면 이웃보다 밀도 낮음(이상 후보)")
print(f"  LOF>1인 양품: {np.sum(sc_lab[mask_n]>0)}/{mask_n.sum()}  불량: {np.sum(sc_lab[mask_d]>0)}/{mask_d.sum()}")""",
    'extra': EXTRA_LOF },

  { 'id':'08f','pre':'NB08f','name':'HBOS (Histogram-Based Outlier Score)',
    'desc':"""## HBOS — 히스토그램 기반 이상탐지

**원리:** 각 피처를 독립적으로 히스토그램 밀도 추정.
이상 점수 = 각 피처 bin의 역빈도 합산 (log scale).

**장점:** 선형 시간복잡도, 단순·빠름, 피처 독립 가정
**단점:** 피처 간 상호작용 무시
**학습:** Unlabeled 71,180행""",
    'sweep': """\
print("HBOS — n_bins 탐색")
rows=[]
for nb in [5,10,20,50,100]:
    clf=HBOS(n_bins=nb,contamination=DR); clf.fit(Xu)
    sc_=clf.decision_function(Xl)
    roc=roc_auc_score(y_lab,sc_); pr=average_precision_score(y_lab,sc_)
    rows.append({'n_bins':nb,'ROC':roc,'PR':pr})
    print(f"  n_bins={nb}  ROC={roc:.4f}  PR={pr:.4f}")
sw_df=pd.DataFrame(rows); print("\\n",sw_df.to_string(index=False))
best_nb=sw_df.loc[sw_df['ROC'].idxmax(),'n_bins']
print(f"\\n최적 n_bins={best_nb}")

# 피처별 히스토그램 분포 시각화 (양품 vs 불량)
top6=['Max_Back_Pressure','Cycle_Time','Average_Screw_RPM','Max_Injection_Speed',
      'Cushion_Position','Barrel_Temperature_1']
top6=[f for f in top6 if f in fcols][:6]
fig,axes=plt.subplots(2,3,figsize=(15,8))
fig.suptitle('HBOS — 주요 피처 히스토그램 (양품 vs 불량)', fontsize=12)
for ax,feat in zip(axes.flatten(),top6):
    fi=fcols.index(feat)
    ax.hist(Xl[mask_n,fi],bins=30,alpha=0.5,color='#4CAF50',density=True,label='양품')
    ax.hist(Xl[mask_d,fi],bins=10,alpha=0.8,color='#F44336',density=True,label='불량')
    ax.set_title(feat,fontsize=8); ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_histograms.png',dpi=150,bbox_inches='tight')
plt.show()""",
    'fit': """\
clf=HBOS(n_bins=int(best_nb),contamination=DR); clf.fit(Xu)
sc_lab=clf.decision_function(Xl); sc_unl=clf.decision_function(Xu)
print(f"HBOS (n_bins={best_nb}) 완료 | ROC={roc_auc_score(y_lab,sc_lab):.4f}")""",
    'extra': EXTRA_HBOS },

  { 'id':'08g','pre':'NB08g','name':'OC-SVM (One-Class SVM) 비교',
    'desc':"""## OC-SVM — 두 학습 방식 비교

**Config A:** labeled 양품만으로 학습 (Phase7 방식, ref=0.8814)
**Config B:** Unlabeled 5K 서브샘플로 학습 (완전 비지도)

**원리:** 정상 데이터를 감싸는 초구(hypersphere) 또는 초평면 학습.
nu = 이상 비율 상한.

**핵심 질문:** 양품 레이블 사용 유무가 성능에 얼마나 영향을 미치는가?""",
    'sweep': """\
print("OC-SVM — nu 파라미터 탐색 (Config A: labeled normal)")
rows=[]
for nu in [0.005,0.0089,0.02,0.05,0.10]:
    clf=OneClassSVM(nu=nu,kernel='rbf',gamma='scale')
    clf.fit(Xl[mask_n])
    sc_=-clf.score_samples(Xl)
    roc=roc_auc_score(y_lab,sc_); pr=average_precision_score(y_lab,sc_)
    rows.append({'nu':nu,'config':'A: labeled normal','ROC':roc,'PR':pr})
    print(f"  [A] nu={nu:.4f}  ROC={roc:.4f}  PR={pr:.4f}")

print("\\nOC-SVM — Config B: unlabeled 5K")
N_OC=5000; rng_o=np.random.RandomState(SEED); idx_o=rng_o.choice(len(Xu),N_OC,replace=False)
for nu in [0.005,0.0089,0.02,0.05]:
    clf=OneClassSVM(nu=nu,kernel='rbf',gamma='scale')
    clf.fit(Xu[idx_o])
    sc_=-clf.score_samples(Xl)
    roc=roc_auc_score(y_lab,sc_); pr=average_precision_score(y_lab,sc_)
    rows.append({'nu':nu,'config':'B: unlabeled 5K','ROC':roc,'PR':pr})
    print(f"  [B] nu={nu:.4f}  ROC={roc:.4f}  PR={pr:.4f}")

sw_df=pd.DataFrame(rows)
print("\\n=== 전체 결과 ===")
print(sw_df.to_string(index=False))""",
    'fit': """\
# Best Config A (labeled normal, nu=DR)
clfA=OneClassSVM(nu=DR,kernel='rbf',gamma='scale'); clfA.fit(Xl[mask_n])
scA=-clfA.score_samples(Xl)
roc_A=roc_auc_score(y_lab,scA)

# Best Config B (unlabeled 5K, nu=DR)
N_OC=5000; rng_o=np.random.RandomState(SEED); idx_o=rng_o.choice(len(Xu),N_OC,replace=False)
clfB=OneClassSVM(nu=DR,kernel='rbf',gamma='scale'); clfB.fit(Xu[idx_o])
scB=-clfB.score_samples(Xl)
roc_B=roc_auc_score(y_lab,scB)

print(f"Config A (labeled normal, nu={DR:.4f}): ROC={roc_A:.4f}")
print(f"Config B (unlabeled 5K, nu={DR:.4f}) : ROC={roc_B:.4f}")
print(f"Phase7 ref (labeled normal)           : ROC=0.8814")
print(f"\\n▶ labeled 양품 정보 사용 유무에 따른 ROC 차이: {roc_A-roc_B:+.4f}")
sc_lab=scA  # 최종 분석은 Config A (성능 우수) 사용
print("\\n이후 분석: Config A (labeled normal) 기준")""",
    'extra': EXTRA_OCSVM },

  { 'id':'08h','pre':'NB08h','name':'Elliptic Envelope (Mahalanobis 거리)',
    'desc':"""## Elliptic Envelope — Mahalanobis 거리 기반 이상탐지

**원리:** labeled 양품의 공분산 행렬을 추정 (Robust MCD).
이상 점수 = Mahalanobis 거리.

**NB08 결과:** ROC=0.8938 — 전체 최고 (단, labeled 양품 사용)
**핵심 이슈:** pseudo-label로 생성된 '불량'이 기계정지 상태 → Step4 실패 원인 분석""",
    'sweep': """\
print("Elliptic Envelope — support_fraction 탐색")
rows=[]
for sf in [0.7,0.8,0.85,0.9,0.95,0.99]:
    try:
        clf=EllipticEnvelope(contamination=DR,random_state=SEED,support_fraction=sf)
        clf.fit(Xl[mask_n])
        sc_=-clf.score_samples(Xl)
        roc=roc_auc_score(y_lab,sc_); pr=average_precision_score(y_lab,sc_)
        rows.append({'support_fraction':sf,'ROC':roc,'PR':pr})
        print(f"  support_fraction={sf}  ROC={roc:.4f}  PR={pr:.4f}")
    except Exception as e:
        print(f"  support_fraction={sf}  ERROR: {e}")
sw_df=pd.DataFrame(rows); print("\\n",sw_df.to_string(index=False))
best_sf=sw_df.loc[sw_df['ROC'].idxmax(),'support_fraction']
print(f"\\n최적 support_fraction={best_sf}")""",
    'fit': """\
clf=EllipticEnvelope(contamination=DR,random_state=SEED,support_fraction=float(best_sf))
clf.fit(Xl[mask_n]); sc_lab=-clf.score_samples(Xl); sc_unl=-clf.score_samples(Xu)
roc=roc_auc_score(y_lab,sc_lab)
print(f"Elliptic Envelope (sf={best_sf}) 완료 | ROC={roc:.4f}")

# pseudo-label 생성 및 검증 (NB08 Step3 재현)
thresh=np.percentile(sc_unl,(1-DR)*100)
pseudo_d=(sc_unl>=thresh)
print(f"\\n▶ Pseudo-label 생성 (contamination={DR*100:.4f}%):")
print(f"  Unlabeled {len(Xu):,}행 중 불량 후보: {pseudo_d.sum():,}개")

# 핵심 진단: pseudo-defect 피처 평균 vs labeled 불량
X_pseudo=Xu[pseudo_d]  # 원본 스케일이 아닌 스케일된 값 (Xu=Xu_sc)
X_pseudo_raw=X_unl[pseudo_d]
comp=pd.DataFrame({'Pseudo-불량 평균':X_pseudo_raw.mean(axis=0),
                   'Labeled-불량 평균':X_lab[mask_d].mean(axis=0),
                   'Labeled-양품 평균':X_lab[mask_n].mean(axis=0)},index=fcols)
print(f"\\n▶ Pseudo-defect vs Labeled 불량 핵심 피처 비교:")
print(f"  {'피처':<30} {'Pseudo':<10} {'Lab불량':<10} {'Lab양품':<10}")
print("  "+"-"*62)
for f in fcols:
    pv=comp.loc[f,'Pseudo-불량 평균']; dv=comp.loc[f,'Labeled-불량 평균']
    nv=comp.loc[f,'Labeled-양품 평균']
    flag='★기계정지★' if abs(pv)<5 and abs(dv)>20 else ''
    if abs(pv-dv)/(abs(dv)+1e-9)>0.5:
        print(f"  {f:<30} {pv:10.2f} {dv:10.2f} {nv:10.2f}  {flag}")""",
    'extra': EXTRA_EE },

  { 'id':'08i','pre':'NB08i','name':'KNN Distance (k-최근접이웃 거리)',
    'desc':"""## KNN Distance — k-최근접이웃 거리 기반 이상탐지

**원리:** 각 샘플의 k개 최근접 이웃까지의 평균 거리.
거리가 클수록 주변에 이웃이 없음 → 이상.

**방법 비교:** mean / max / median aggregation
**학습:** Unlabeled 10K 서브샘플 (메모리 제약)""",
    'sweep': """\
print("KNN Distance — k값 및 집계 방법 탐색")
N_KNN=10000; rng_k=np.random.RandomState(SEED); idx_k=rng_k.choice(len(Xu),N_KNN,replace=False)
Xu_knn=Xu[idx_k]
rows=[]
for nn in [5,10,20,30,50]:
    for meth in ['mean','largest','median']:
        clf=PyODKNN(n_neighbors=nn,contamination=DR,method=meth)
        clf.fit(Xu_knn); sc_=clf.decision_function(Xl)
        roc=roc_auc_score(y_lab,sc_); pr=average_precision_score(y_lab,sc_)
        rows.append({'k':nn,'method':meth,'ROC':roc,'PR':pr})
        print(f"  k={nn:2d}  method={meth:<6}  ROC={roc:.4f}  PR={pr:.4f}")
sw_df=pd.DataFrame(rows); best_row=sw_df.loc[sw_df['ROC'].idxmax()]
print(f"\\n최적: k={best_row['k']}  method={best_row['method']}  ROC={best_row['ROC']:.4f}")""",
    'fit': """\
best_k_knn=int(best_row['k']); best_meth=best_row['method']
clf=PyODKNN(n_neighbors=best_k_knn,contamination=DR,method=best_meth)
clf.fit(Xu_knn); sc_lab=clf.decision_function(Xl); sc_unl=clf.decision_function(Xu_knn)
print(f"KNN (k={best_k_knn}, method={best_meth}) 완료 | ROC={roc_auc_score(y_lab,sc_lab):.4f}")

# k별 ROC 시각화
fig,axes=plt.subplots(1,2,figsize=(14,5))
for meth,color in [('mean','#2196F3'),('largest','#F44336'),('median','#4CAF50')]:
    sub=sw_df[sw_df['method']==meth]
    axes[0].plot(sub['k'],sub['ROC'],'-o',color=color,label=meth)
    axes[1].plot(sub['k'],sub['PR'],'-o',color=color,label=meth)
axes[0].set_xlabel('k'); axes[0].set_ylabel('ROC-AUC'); axes[0].legend(); axes[0].set_title('ROC by k')
axes[1].set_xlabel('k'); axes[1].set_ylabel('PR-AUC'); axes[1].legend(); axes[1].set_title('PR by k')
plt.tight_layout()
plt.savefig(FIG_DIR/f'{NB_PRE}_fig_knn_sweep.png',dpi=150,bbox_inches='tight')
plt.show(); print("KNN 탐색 저장 완료")""",
    'extra': EXTRA_KNN },
]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Generate notebooks
# ══════════════════════════════════════════════════════════════════════════════
print("=== Generating method notebooks ===")

for m in METHODS:
    cells = [
        mc(f"# {m['name']} — 상세 EDA\n{m['desc']}"),
        cc(f"NB_NAME = '{m['name']}'\nNB_PRE  = '{m['pre']}'"),
        mc("## 0. 환경 설정"),
        cc(T_SETUP),
        mc("## 1. 데이터 로드 + CN7/RG3 필터링"),
        cc(T_DATA),
        mc("## 2. PCA 2D + 3D 시각화 (데이터 구조 파악)"),
        cc(T_PCA),
        mc("## 3. UMAP 2D + 3D 시각화"),
        cc(T_UMAP),
        mc(f"## 4. {m['name']} — 하이퍼파라미터 탐색"),
        cc(m['sweep']),
        mc("## 5. 최적 모델 학습 + 이상 점수 계산"),
        cc(m['fit']),
        mc("## 6. 이상 점수 PCA/UMAP 2D+3D overlay"),
        cc(T_SCORE_VIZ),
        mc("## 7. 이상 점수 분포 + KS 검정"),
        cc(T_DIST),
        mc("## 7b. ROC / PR 곡선"),
        cc(T_ROC_PR),
        mc("## 7c. Violin Plot + ECDF"),
        cc(T_VIOLIN),
        mc("## 7d. 임계값 분석 (Precision-Recall Trade-off)"),
        cc(T_THRESHOLD),
        mc("## 8. 피처별 이상 점수 상관 분석"),
        cc(T_FEAT),
        mc("## 8b. 병렬 좌표계 (상위이상 vs 불량 vs 양품)"),
        cc(T_PARALLEL),
        mc("## 8c. 피처 평균 히트맵 (이상점수 분위별)"),
        cc(T_CORR_HEAT),
        mc("## 9. 상위 이상 샘플 분석 + Labeled 불량 비교"),
        cc(T_TOP),
        mc(f"## 10. {m['name']} — 방법 특화 심층 분석"),
        cc(m['extra']),
    ]
    nb_path = NB_DIR / f"{m['id']}_{m['pre'].lower().replace('nb0','')}_eda.ipynb"
    save_nb(make_nb(cells), nb_path)
    print(f"  [OK] {nb_path.name}  ({len(cells)} cells)")

print("\nAll done!")
