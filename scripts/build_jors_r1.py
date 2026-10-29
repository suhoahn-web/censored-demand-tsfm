"""JORS Revision 1: markdown 소스 → 조립 → docx (pandoc).

입력: Target Journal/Journal of the Operational Research Society/Revision-1/source/*.md
      코드/results/tables/*.md, 코드/results/figures/*.png,
      Revision-1/tables/compute_cost_table.md
출력: Revision-1/build/Manuscript_R1_assembled.md
      Revision-1/Manuscript_R1_anonymous.docx, Manuscript_R1_with_author.docx
      Revision-1/build/wordcount.txt

사용: python scripts/build_jors_r1.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JORS = ROOT.parent / "Target Journal" / "Journal of the Operational Research Society"
REV = JORS / "Revision-1"
SRC = REV / "source"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
BUILD = REV / "build"
BUILD.mkdir(exist_ok=True)

AUTHOR_BLOCK = (
    "Suho Ahn\n\nSchool of Business and Technology Management, Korea Advanced Institute of "
    "Science and Technology (KAIST), Daejeon, Republic of Korea. E-mail: dkstngh3502@kaist.ac.kr\n\n"
    "Reproduction package: https://github.com/suhoahn-web/censored-demand-tsfm "
    "(archived at https://doi.org/10.5281/zenodo.23754720).\n\n"
)


def read(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def strip_notes(t: str) -> str:
    """파일 머리의 '> ...' 작업 메모와 첫 H1을 제거."""
    lines = t.splitlines()
    out = []
    for ln in lines:
        if ln.startswith("> ") or ln.strip() == ">":
            continue
        if ln.startswith("# "):
            continue
        out.append(ln)
    return "\n".join(out).strip() + "\n"


def section(t: str, start: str, end: str | None) -> str:
    i = t.index(start)
    j = t.index(end, i) if end else len(t)
    return t[i:j].strip() + "\n"


def table_rows(md_path: Path, n_rows: int | None = None) -> str:
    """results/tables의 md에서 표 본문만 (캡션 제외). n_rows면 상위 n개 데이터 행."""
    lines = [ln for ln in md_path.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    if n_rows is not None:
        lines = lines[:2 + n_rows]
    return "\n".join(lines)


def table_full(md_path: Path, caption_prefix: str) -> str:
    t = md_path.read_text(encoding="utf-8").strip()
    # 첫 줄이 **caption** 형식이면 라벨만 바꾼다
    first, rest = t.split("\n", 1)
    if first.startswith("**Table"):
        return t
    if first.startswith("**"):
        return f"**{caption_prefix}** " + first.strip("*") + "\n" + rest
    return f"**{caption_prefix}**\n\n" + t


def fig(n: int, fname: str, caption: str, width: str = "100%") -> str:
    p = FIG / fname
    if not p.exists():
        sys.exit(f"missing figure {p}")
    return f"\n![{caption}]({p.as_posix()}){{ width={width} }}\n\n**Fig. {n}.** {caption}\n\n"


def insert_after(t: str, anchor: str, block: str) -> str:
    n = t.count(anchor)
    if n != 1:
        sys.exit(f"anchor found {n} times: {anchor[:70]}")
    i = t.index(anchor) + len(anchor)
    # 문단 끝까지 이동
    j = t.find("\n\n", i)
    j = len(t) if j < 0 else j
    return t[:j] + "\n\n" + block.strip() + "\n" + t[j:]


def insert_before(t: str, anchor: str, block: str) -> str:
    n = t.count(anchor)
    if n != 1:
        sys.exit(f"anchor found {n} times: {anchor[:70]}")
    i = t.index(anchor)
    return t[:i] + block.strip() + "\n\n" + t[i:]


def main() -> None:
    d0, d1, d2, refs = read("draft-00-frontmatter.md"), read("draft-01-intro-methods.md"), \
        read("draft-02-results-discussion.md"), read("references.md")

    title = section(d0, "## Title", "## Highlights").replace("## Title", "").strip().strip("*")
    abstract = section(d0, "## Abstract", "## Practitioner Summary")
    psum = section(d0, "## Practitioner Summary", "## Keywords")
    psum = re.sub(r"\*\(JORS[^\n]*\)\*\n?", "", psum)
    keywords = section(d0, "## Keywords", "## Appendix A")
    app_a = section(d0, "## Appendix A", "## Declarations")

    body1 = strip_notes(d1)
    app_d = section(body1, "## Appendix D", None)
    body1 = body1[:body1.index("## Appendix D")].rstrip().rstrip("-").rstrip() + "\n"

    body2 = strip_notes(d2)
    results = section(body2, "## 5. Results", "## Appendix B")
    app_b = section(body2, "## Appendix B", "## 6. Discussion")
    disc = section(body2, "## 6. Discussion", None)
    results = results.rstrip().rstrip("-").rstrip() + "\n"
    app_b = app_b.rstrip().rstrip("-").rstrip() + "\n"

    # ── 본문 표 삽입 ──
    acc_rows = table_rows(TAB / "table4_accuracy_lt.md", 8)
    results = insert_after(
        results, "Table 4 reports accuracy on the long-horizon release",
        f"**Table 4.** Forecast accuracy on FreshRetailNet-LT (official 7-day evaluation, days "
        f"without stockouts only). WAPE and WPE in %; WPE < 0 indicates under-forecasting.\n\n{acc_rows}\n\n"
        f"*C2 = Chronos-2; EQ = empirical residual quantiles; mask = stockout days masked. "
        f"All methods in Table C1.*")
    rec_tab = table_rows(TAB / "table7_censoring.md")
    results = insert_before(
        results, "The locally trained models are means over three seeds",
        f"**Table 5.** Effect of latent demand recovery on accuracy and bias (FreshRetailNet-50K). "
        f"Models are trained on censored sales versus recovered demand and evaluated against observed "
        f"sales in both cases.\n\n{rec_tab}\n\n*WAPE and WPE in %.*")
    inv_rows = table_rows(TAB / "table6_inventory.md", 8)
    results = insert_after(
        results, "reports the outcome and Fig. 5 relates it to forecast accuracy.",
        f"**Table 6.** Inventory cost per unit of demand under the three prespecified cells "
        f"(FreshRetailNet-LT, shelf life 2 days, lead time 0, observed-demand path). All methods order "
        f"at the same quantile within a cell (0.5, 0.7 and 0.9 respectively).\n\n{inv_rows}\n\n"
        f"*Unit purchase cost 1; b is the unit shortage cost. All methods, including point forecasters "
        f"that order at the median only, in Table C2.*")

    # ── 그림 삽입 ──
    body1 = insert_before(body1, "## 2. Related work", fig(
        1, "fig1_framework.png",
        "Overview of the experimental design. Censored sales enter under three input treatments; "
        "forecasts from each method family become orders under a fixed-quantile policy, evaluated on "
        "uncensored days by forecast error and, under four demand paths, by inventory cost."))
    results = insert_after(results, "TimesFM (13.79) fall", fig(
        2, "fig2_mcb_lt.png",
        "Multiple comparisons with the best on FreshRetailNet-LT: mean rank of per-series MASE over "
        "10,000 series, with critical-distance intervals; intervals entirely right of the dashed line "
        "are significantly worse than the best."))
    results = insert_after(results, "82% of this dataset, dominate the aggregate.", fig(
        3, "fig4_quadrant_lt.png",
        "Mean rank (lower is better) by demand-pattern quadrant (ADI × CV² classification) for "
        "selected methods; stockout masking shown dashed."))
    results = insert_after(results, "$+8.9\\%$ after recovery.", fig(
        4, "fig3_censoring_factorial.png",
        "Effect of latent demand recovery on bias (WPE, left) and accuracy (WAPE, right), 90-day "
        "release: zero-shot Chronos-2, zero-shot Moirai-2 and LoRA-adapted Chronos-2 (seed 0)."))
    results = insert_after(results, "configuration is stable across all three cells.", fig(
        5, "fig5_accuracy_vs_cost_lt.png",
        "WAPE against inventory cost per unit of demand in the three prespecified cells, "
        "observed-demand path; τ is the Kendall rank correlation (* p < 0.05)."))
    results = insert_after(results, "$\\tau = +0.06$ ($p = 0.74$) to $\\tau = +0.51$ ($p = 0.0007$).", fig(
        6, "fig7_bias_crossover_lt.png",
        "Kendall rank correlation of inventory cost with absolute bias (|WPE|) and with distributional "
        "quality (WQL) as the ordering quantile rises at a fixed cost ratio (b = 1); filled markers p < 0.05."))
    results = insert_after(results, "WPE $-8.59\\% \\to -2.29\\%$).", fig(
        7, "fig6_tradeoff_lt.png",
        "Average inventory against achieved fill rate as the ordering quantile sweeps 0.2–0.95 "
        "(medium-margin cell, selected methods). Methods with a predictive distribution trace a "
        "frontier; a point forecaster collapses to one operating point."))

    # ── 부록 B 표 ──
    app_b = insert_after(app_b, "quantiles); plain ETS is cheapest at one origin there.",
                         (REV / "tables" / "table_B3_origins.md").read_text(encoding="utf-8"))
    app_b = insert_after(app_b, "quoted in Sections 5.3.2 and\n5.4.",
                         (REV / "tables" / "table_B4_50k_inventory.md").read_text(encoding="utf-8"))
    app_b = app_b + "\n\n" + (TAB / "r1_B7_synthetic_censoring.md").read_text(encoding="utf-8") \
        + "\n\n" + (TAB / "r1_B8_eq_uncensored.md").read_text(encoding="utf-8") \
        + "\n\n" + (TAB / "r1_B9_demand_paths.md").read_text(encoding="utf-8") + "\n"

    # ── 부록 C ──
    comp = (REV / "tables" / "compute_cost_table.md").read_text(encoding="utf-8")
    comp = comp.split("\n", 1)[1]  # 머리 H1 제거
    app_c = "## Appendix C. Complete result tables\n\n" \
        + table_full(TAB / "table4_accuracy_lt.md", "Table C1.") + "\n\n" \
        + re.sub(r"\bnan\b", "–", table_full(TAB / "table6_inventory.md", "Table C2.")) + "\n\n" \
        + table_full(TAB / "table5_adaptation.md", "Table C3.").replace(
            "Adaptation strategies for Chronos-2 on FreshRetailNet-LT.",
            "Adaptation strategies for Chronos-2 on FreshRetailNet-LT (LoRA learning rate $10^{-4}$ unless stated; 1,000 steps).") + "\n\n" \
        + (TAB / "r1_C4_quantile_losses.md").read_text(encoding="utf-8") + "\n\n" \
        + (TAB / "r1_C5_bias_definitions.md").read_text(encoding="utf-8") + "\n\n" \
        + (TAB / "r1_C6_achieved_service.md").read_text(encoding="utf-8") + "\n\n" \
        + (TAB / "r1_C7_mase_weighting.md").read_text(encoding="utf-8") + "\n\n" \
        + (TAB / "r1_C8_coverage.md").read_text(encoding="utf-8") + "\n\n" \
        + "**Table C9.** Computational cost on a common workload.\n\n" + comp + "\n\n" \
        + (REV / "tables" / "table_C10_grid.md").read_text(encoding="utf-8") + "\n"

    app_c = app_c.replace("tune its service level", "tune its ordering quantile")
    decl = (
        "## Disclosure statement\n\nThe author reports there are no competing interests to declare.\n\n"
        "## Funding\n\nThis research did not receive any specific grant from funding agencies in the "
        "public, commercial, or not-for-profit sectors.\n\n"
        "## Data availability\n\nThe FreshRetailNet datasets are publicly available under CC-BY-4.0 at "
        "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K and "
        "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-LT. All code, configurations and "
        "per-run metadata (checkpoint hashes, seeds, library versions) are archived in a public repository with a persistent DOI; the identifier is "
        "withheld for anonymized review and given on the title page.\n\n"
        "## Declaration of generative AI and AI-assisted technologies in the writing process\n\n"
        "During the preparation of this work the author used Claude (Anthropic; Claude Opus 5 and "
        "Claude Fable 5, accessed through Claude Code, July–October 2026) to assist with language "
        "editing and drafting of the manuscript and with implementing the analysis and simulation code. "
        "All reported results were produced by that code and checked against the stored experimental "
        "output. The author reviewed and edited all content and takes full responsibility for the "
        "publication.\n\n")

    refs_body = strip_notes(refs)
    refs_body = re.sub(r"\s*⚠[^\n]*(?:\n(?!\n)[^\n]*)*", "", refs_body)  # 내부 검토 메모 제거
    refs_body = "## References\n\n" + refs_body.replace("## References", "").strip() + "\n"

    def assemble(anonymous: bool) -> str:
        head = f"# {title}\n\n" + ("" if anonymous else AUTHOR_BLOCK)
        return (head + abstract + "\n" + psum + "\n" + keywords + "\n" + body1 + "\n" + results
                + "\n" + disc + "\n" + decl + refs_body + "\n" + app_a + "\n" + app_b + "\n"
                + app_c + "\n" + app_d)

    anon = assemble(True)
    full = assemble(False)
    (BUILD / "Manuscript_R1_assembled.md").write_text(anon, encoding="utf-8")

    # ── 단어 수 (본문 §1–7, 표·그림 캡션 제외) ──
    main_text = body1 + results + disc
    no_tables = "\n".join(ln for ln in main_text.splitlines()
                          if not ln.startswith("|") and not ln.startswith("![")
                          and not ln.startswith("**Fig.") and not ln.startswith("**Table"))
    words = len(re.findall(r"\S+", no_tables))
    words_all = len(re.findall(r"\S+", main_text))
    (BUILD / "wordcount.txt").write_text(
        f"main text 1-7 excluding tables/captions: {words}\nincluding tables/captions: {words_all}\n",
        encoding="utf-8")
    print(f"main text words (excl. tables/captions): {words}; incl.: {words_all}")

    for name, text in [("Manuscript_R1_anonymous.docx", anon), ("Manuscript_R1_with_author.docx", full)]:
        src = BUILD / (name.replace(".docx", ".md"))
        src.write_text(text, encoding="utf-8")
        cmd = ["pandoc", str(src), "-o", str(REV / name), "--from",
               "markdown+tex_math_dollars+pipe_tables+implicit_figures", "--to", "docx",
               "--resource-path", str(FIG)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(r.stderr[-2000:])
        print("built:", REV / name)


if __name__ == "__main__":
    main()
