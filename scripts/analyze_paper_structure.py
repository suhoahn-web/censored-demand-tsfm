"""논문 문단 구조 분석 — 키페이퍼와 우리 초안의 문단 수·문장 수·역할 대조.

PDF(2단 조판)는 컬럼 단위로 텍스트를 추출해 문단을 복원한 뒤,
문단별 문장 수와 첫 문장을 출력한다.

사용: python scripts/analyze_paper_structure.py <pdf|md> [--pages 1-3] [--section Introduction]
"""

import argparse
import re
from pathlib import Path

ABBREV = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bet al)(?<!\bcf)(?<!\bvs)(?<!\bFig)(?<!\bEq)(?<!\bNo)(?<!\bMr)(?<!\bDr)"
SENT_END = re.compile(ABBREV + r"(?<=[.!?])\s+(?=[A-Z(])")


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s for s in SENT_END.split(text) if s.strip()]


def pdf_paragraphs(path: Path, pages: str) -> list[str]:
    import fitz
    doc = fitz.open(path)
    lo, hi = (pages.split("-") + [pages])[:2]
    paras, buf = [], ""
    for pno in range(int(lo) - 1, int(hi)):
        page = doc[pno]
        w = page.rect.width
        # 2단 조판: 좌/우 컬럼을 각각 순서대로 읽는다
        for clip in [(0, 0, w / 2, page.rect.height), (w / 2, 0, w, page.rect.height)]:
            blocks = page.get_text("blocks", clip=clip)
            for b in sorted(blocks, key=lambda x: x[1]):
                t = b[4].strip()
                if len(t) < 40 or t.count("\n") == 0:
                    continue          # 머리글·캡션·짧은 조각 제외
                t = re.sub(r"-\n", "", t)
                t = re.sub(r"\s*\n\s*", " ", t)
                if re.match(r"^(Fig\.|Table|Abstract|Keywords|©|http)", t):
                    continue
                # 들여쓰기/문장 종료로 문단 경계 추정
                if buf and not buf.rstrip().endswith((".", ":", "?", "!")):
                    buf += " " + t
                else:
                    if buf:
                        paras.append(buf)
                    buf = t
    if buf:
        paras.append(buf)
    return paras


def md_paragraphs(path: Path, section: str | None) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if section:
        m = re.search(rf"^##+ .*{re.escape(section)}.*$", text, re.M)
        if m:
            rest = text[m.end():]
            nxt = re.search(r"^##+ ", rest, re.M)
            text = rest[:nxt.start()] if nxt else rest
    out = []
    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if not b or b.startswith(("#", ">", "|", "-", "*", "$$", "```", "1.", "2.", "3.", "4.")):
            continue
        b = re.sub(r"\*\*(.+?)\*\*", r"\1", b)
        b = re.sub(r"\s*\n\s*", " ", b)
        if len(b) > 80:
            out.append(b)
    return out


def report(paras: list[str], title: str) -> None:
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    total = 0
    for i, p in enumerate(paras, 1):
        ss = sentences(p)
        total += len(ss)
        words = len(p.split())
        head = ss[0][:105] + ("..." if len(ss[0]) > 105 else "") if ss else ""
        print(f"P{i:<2} | {len(ss):>2} sent | {words:>4} words | {head}")
    n = len(paras)
    if n:
        print(f"\n합계 {n}문단 / {total}문장 (문단당 평균 {total/n:.1f}문장)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--pages", default="1-2")
    ap.add_argument("--section", default=None)
    a = ap.parse_args()
    p = Path(a.path)
    paras = pdf_paragraphs(p, a.pages) if p.suffix == ".pdf" else md_paragraphs(p, a.section)
    report(paras, f"{p.name}" + (f" [{a.section}]" if a.section else f" [pages {a.pages}]"))
