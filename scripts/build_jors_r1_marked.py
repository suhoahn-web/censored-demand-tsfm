"""JORS R1: 변경 표시본(changes highlighted) docx.

build_jors_r1.py가 만든 build/Manuscript_R1_anonymous.md의 각 문단을 원 제출본 소스
(JORS/draft-00/01/02.md, 2026-08-12)와 비교해, 원문에 없는 문단을 노란 음영([...]{.mark})으로
표시한다. 표는 전체를 음영 처리할 수 없으므로 새 표·재생성 표 앞에 표시 줄을 넣는다.

사용: python scripts/build_jors_r1_marked.py  (build_jors_r1.py 실행 후)
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_post import finalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JORS = ROOT.parent / "Target Journal" / "Journal of the Operational Research Society"
REV = JORS / "Revision-1"
BUILD = REV / "build"
FIG = ROOT / "results" / "figures"


def norm(p: str) -> str:
    return re.sub(r"\s+", " ", p).strip()


orig = "\n\n".join((JORS / f).read_text(encoding="utf-8") for f in
                   ["draft-00-frontmatter.md", "draft-01-intro-methods.md",
                    "draft-02-results-discussion.md"])
orig_paras = {norm(p) for p in orig.split("\n\n")}
orig_text = norm(orig)

src = (BUILD / "Manuscript_R1_anonymous.md").read_text(encoding="utf-8")
out, n_marked, n_total = [], 0, 0
new_tables = {"Table 4.", "Table B2.", "Table B7.", "Table B8.", "Table B9.", "Table C4.", "Table C5.",
              "Table C6.", "Table C7.", "Table C8.", "Table C9.", "Table A1."}
for para in src.split("\n\n"):
    s = para.strip()
    if not s:
        continue
    if s.startswith("#") or s.startswith("![") or s.startswith("**Fig."):
        out.append(para)
        continue
    if s.startswith("|"):
        out.append(para)
        continue
    if s.startswith("**Table"):
        label = s.split("**")[1]
        if any(label.startswith(t.rstrip(".")) for t in new_tables) or label.startswith("Table 5.") \
                or label.startswith("Table 7.") or label.startswith("Table C1.") or label.startswith("Table C2."):
            out.append("[Table new or regenerated in revision]{.mark}\n\n" + para)
        else:
            out.append(para)
        continue
    n_total += 1
    key = norm(s)
    # 원문 문단과 동일하거나 원문 전체에 그대로 포함되면 미변경
    if key in orig_paras or key in orig_text:
        out.append(para)
    else:
        n_marked += 1
        # 수식·강조가 섞인 문단도 span으로 감쌀 수 있도록 줄바꿈을 공백으로
        out.append("[" + re.sub(r"\n", " ", s) + "]{.mark}")

marked = "\n\n".join(out) + "\n"
md_path = BUILD / "Manuscript_R1_anonymous_changes_marked.md"
md_path.write_text(marked, encoding="utf-8")
docx = REV / "Manuscript_R1_anonymous_changes_marked.docx"
r = subprocess.run(["pandoc", str(md_path), "-o", str(docx), "--from",
                    "markdown+tex_math_dollars+pipe_tables+implicit_figures+bracketed_spans",
                    "--to", "docx", "--resource-path", str(FIG)], capture_output=True, text=True)
if r.returncode != 0:
    sys.exit(r.stderr[-2000:])
finalize(docx)
print(f"marked paragraphs: {n_marked} of {n_total}; built: {docx}")
