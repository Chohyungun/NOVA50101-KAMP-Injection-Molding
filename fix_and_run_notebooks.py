# -*- coding: utf-8 -*-
"""
노트북 일괄 수정 + 재실행 스크립트.

핵심 문제:
- sns.set_theme() 이 matplotlib font.family 를 'sans-serif' 로 덮어씀
- setup_korean_font() 는 sns.set_theme() 이후에 호출돼야 함
- 현재 노트북들은 setup_korean_font() 가 sns.set_theme() 이전에 위치

수정 내용:
1. 00_EDA.ipynb: 날짜 파싱 수정 (오전/오후 -> AM/PM)
2. 모든 노트북: sns.set_theme() 바로 다음 줄에 setup_korean_font() 삽입
   (이미 바로 다음에 있으면 skip)
3. nbconvert 로 전체 재실행
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parent
NB_DIR  = PROJECT / "notebooks"

NOTEBOOKS_ORDER = [
    "00_EDA.ipynb",
    "01_preprocessing_ablation.ipynb",
    "02_linear_baselines.ipynb",
    "03_dimensionality_reduction.ipynb",
    "04_ensemble_nn.ipynb",
    "05_cnn_aux_stacking.ipynb",
    "06_results_summary.ipynb",
]

AM_KOR = "오전"
PM_KOR = "오후"

# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def get_source_str(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def set_source(cell, new_str):
    if isinstance(cell["source"], list):
        cell["source"] = new_str.splitlines(keepends=True)
    else:
        cell["source"] = new_str


# ---------------------------------------------------------------------------
# 1. 날짜 파싱 수정 (00_EDA 전용)
# ---------------------------------------------------------------------------

def fix_date_parsing(nb_path: Path) -> bool:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = False

    new_parse_dt = (
        "pd.to_datetime(\n"
        f"    df_ld['PART_FACT_PLAN_DATE']\n"
        f"      .str.replace('{AM_KOR}', 'AM', regex=False)\n"
        f"      .str.replace('{PM_KOR}', 'PM', regex=False),\n"
        f"    format='%Y-%m-%d %p %I:%M:%S', errors='coerce'\n"
        f")"
    )

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source_str(cell)

        old1 = "pd.to_datetime(df_ld['PART_FACT_PLAN_DATE'], errors='coerce').dt.date"
        new1 = new_parse_dt + ".dt.date"

        old2 = "pd.to_datetime(df_ld['PART_FACT_PLAN_DATE'], errors='coerce')\n"
        new2 = new_parse_dt + "\n"

        if old1 in src:
            src = src.replace(old1, new1)
            changed = True
        if old2 in src and "sort_values" in src:
            src = src.replace(old2, new2)
            changed = True

        set_source(cell, src)

    if changed:
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [fixed] {nb_path.name}: date parsing updated")
    return changed


# ---------------------------------------------------------------------------
# 2. sns.set_theme() 바로 다음 줄에 setup_korean_font() 삽입
# ---------------------------------------------------------------------------

def ensure_font_after_seaborn(nb_path: Path) -> bool:
    """각 코드 셀에서 sns.set_theme() 호출 줄의 바로 다음 줄이
    setup_korean_font() 가 아니면 삽입한다."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb_changed = False

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = get_source_str(cell)
        if "sns.set_theme(" not in src:
            continue

        lines = src.splitlines(keepends=True)
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)
            stripped = line.rstrip()

            # sns.set_theme(...) 호출이 이 줄에서 끝나면
            if "sns.set_theme(" in stripped and stripped.endswith(")"):
                # 다음 줄이 setup_korean_font() 인지 확인
                next_stripped = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if next_stripped != "setup_korean_font()":
                    new_lines.append("setup_korean_font()\n")
                    nb_changed = True
            i += 1

        if nb_changed:
            set_source(cell, "".join(new_lines))

    if nb_changed:
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [fixed] {nb_path.name}: setup_korean_font() placed after sns.set_theme()")
    else:
        print(f"  [ok]    {nb_path.name}: already correct")
    return nb_changed


# ---------------------------------------------------------------------------
# 3. 노트북 재실행
# ---------------------------------------------------------------------------

def run_notebook(nb_path: Path, timeout: int = 3600) -> bool:
    print(f"\n{'='*60}")
    print(f"  Running: {nb_path.name}")
    print(f"{'='*60}")
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(nb_path),
    ]
    result = subprocess.run(cmd, capture_output=False)
    ok = result.returncode == 0
    print(f"  [{'OK' if ok else 'FAIL'}] {nb_path.name}")
    return ok


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Step 1: 노트북 수정 ===")

    fix_date_parsing(NB_DIR / "00_EDA.ipynb")

    for nb_name in NOTEBOOKS_ORDER:
        ensure_font_after_seaborn(NB_DIR / nb_name)

    print("\n=== Step 2: 노트북 재실행 (순서대로) ===")
    results = {}
    for nb_name in NOTEBOOKS_ORDER:
        ok = run_notebook(NB_DIR / nb_name)
        results[nb_name] = "OK" if ok else "FAIL"

    print("\n=== 실행 결과 요약 ===")
    for nb, status in results.items():
        mark = "v" if status == "OK" else "x"
        print(f"  {mark} {nb}: {status}")
