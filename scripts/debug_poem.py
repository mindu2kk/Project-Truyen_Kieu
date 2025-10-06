# scripts/debug_poem.py
# -*- coding: utf-8 -*-
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "app"))
from poem_tools import poem_ready, get_opening, get_range


# In thử 12 dòng đầu
top = get_opening(12)
for i, ln in enumerate(top, 1):
    print(f"{i:>4}: {ln}")

print("\n— Test khoảng 241–260 —")
seg = get_range(241, 260)
for i, ln in enumerate(seg, 241):
    print(f"{i:>4}: {ln}")
