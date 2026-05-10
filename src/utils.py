"""Utility helpers — seeding, timing, font setup."""
from __future__ import annotations

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix numpy, random, and torch (if available) seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_korean_font() -> str | None:
    """Windows/Linux/Mac에서 한글 matplotlib 폰트 자동 설정."""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    candidates = [
        'Malgun Gothic',      # Windows
        'NanumGothic',        # Linux/installed
        'AppleGothic',        # macOS
        'Noto Sans CJK KR',   # generic
        'NanumBarunGothic',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return name
    # last resort: find any TTF with Korean glyphs by filename
    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in ['malgun', 'nanum', 'gothic', 'cjk']):
            plt.rcParams['font.family'] = f.name
            plt.rcParams['axes.unicode_minus'] = False
            return f.name
    return None
