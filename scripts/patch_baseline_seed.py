"""TFT·DLinear의 하드코딩된 시드를 환경변수로 노출 (절대 규칙 5: 시드 3회 반복).

원 리포는 `fix_seed = 2025`가 모듈 최상단에 박혀 있어 반복 실행이 불가능하다.
모델링 로직은 건드리지 않고 시드 출처만 환경변수 FRN_SEED로 바꾼다(기본값 2025 유지).

실행: python scripts/patch_baseline_seed.py <repo_root>
"""

import sys
from pathlib import Path

OLD = "fix_seed = 2025"
NEW = "import os as _os\nfix_seed = int(_os.environ.get('FRN_SEED', 2025))"

TARGETS = [
    "demand_forecasting/TFT/trainTFT.py",
    "demand_forecasting/DLinear/main.py",
]

if __name__ == "__main__":
    repo = Path(sys.argv[1]).expanduser()
    for rel in TARGETS:
        p = repo / rel
        t = p.read_text()
        if "FRN_SEED" in t:
            print(f"already patched: {rel}")
            continue
        if OLD not in t:
            print(f"target not found: {rel}")
            continue
        p.write_text(t.replace(OLD, NEW, 1))
        print(f"patched: {rel}")
