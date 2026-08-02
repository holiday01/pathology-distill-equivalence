#!/usr/bin/env python3
"""Parse outputs/downstream_eval_queue.log into a tidy (teacher, student) table.

The 2026-05-05/06 queue crashed mid-evaluation but printed CKA + C16 linear
results for ~34 of 36 runs before failing in cohen_kappa / MLP probe. This
script recovers those numbers into a CSV + markdown table so we have *something*
to fill the F1/F6/F2 figures while the full eval reruns.

Output rows are (teacher, student, cka, cos, c16lin_T_acc, c16lin_S_acc,
c16lin_d_mean, c16lin_d_ci90_lo, c16lin_d_ci90_hi, c16lin_kappa,
c16lin_tost_equiv, c16lin_p_lower, c16lin_p_upper, c16lin_n_slides).
Missing fields are left blank.

The parser is line-oriented and stateful: it tracks the active (teacher,
student) from `[eval]   FM × STU  TIMESTAMP` markers and absorbs the next
`[sim]` and `[C16 linear]` lines into that bucket. A new `[eval]` flushes
the previous bucket.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Anomaly screening rule (NOT a hypothesis test):
#     RAW CKA >= ANOMALY_CKA_MIN  AND  |Δacc| >= ANOMALY_DELTA_MIN
#     AND  CI strictly excludes 0  AND  CI is NOT degenerate at n_sl=2
# Thresholds are ad-hoc *screening* cutoffs to flag rows for human review,
# not significance levels. Rationale:
#   - CKA >= 0.80: "high alignment" relative to the sweep median (~0.85);
#     value chosen so the flag set is meaningful but not trivial.
#   - |Δacc| >= 0.10: ~3x the TOST equivalence margin of 0.03, large enough
#     to be conspicuous against the at-chance baseline.
#   - CI strictly excludes 0: confidently signed divergence direction.
#   - Degenerate CI exclusion: prevents the `※` artifact-tight CI from
#     laundering bootstrap noise into an anomaly signal (Stats Round 5).
# Catches both S≫T and S≪T. Applied to *raw* CKA so hibou `§` rows (CKA
# blanked to NaN in CSV) can still be evaluated.
ANOMALY_CKA_MIN = 0.80
ANOMALY_DELTA_MIN = 0.10

# Optional custom footnote text per (teacher, student); rows matching the
# criterion but not listed here get a generic auto-generated footnote.
ANOMALY_FOOTNOTES: dict[tuple[str, str], str] = {
    ("virchow", "vit-base"): "S≫T anomaly (Δacc=+0.341, CI strictly positive). "
                              "REDACTED from interim aggregates; likely probe/cache "
                              "bug — investigate before paper use.",
}


def _is_anomaly(row) -> bool:
    """Apply the screening rule on raw row values (pre-blanking).
    Returns False for degenerate-CI rows: a `※` row's CI bounds are bootstrap
    artifacts at n_sl=2, so using them to claim "CI excludes 0" launders
    a known artifact into an anomaly signal."""
    if (row.cka is None or row.c16lin_d_mean is None
            or row.c16lin_d_ci90_lo is None or row.c16lin_d_ci90_hi is None):
        return False
    if _is_degenerate_ci(row):
        return False
    if row.cka < ANOMALY_CKA_MIN:
        return False
    if abs(row.c16lin_d_mean) < ANOMALY_DELTA_MIN:
        return False
    # CI strictly excludes 0 ⇔ both bounds have the same sign and neither is 0.
    return (row.c16lin_d_ci90_lo > 0 and row.c16lin_d_ci90_hi > 0) or \
           (row.c16lin_d_ci90_lo < 0 and row.c16lin_d_ci90_hi < 0)


def _anomaly_footnote(row) -> str:
    """Get the footnote text for an anomaly row: custom if listed, else
    auto-generated from the rule."""
    key = (row.teacher, row.student)
    if key in ANOMALY_FOOTNOTES:
        return ANOMALY_FOOTNOTES[key]
    direction = "S≫T" if row.c16lin_d_mean > 0 else "S≪T"
    return (f"{direction} anomaly: CKA={row.cka:.3f} (≥{ANOMALY_CKA_MIN}) with "
            f"Δacc={row.c16lin_d_mean:+.3f}, CI=[{row.c16lin_d_ci90_lo:+.3f}, "
            f"{row.c16lin_d_ci90_hi:+.3f}] strictly "
            f"{'positive' if row.c16lin_d_mean > 0 else 'negative'}. "
            "Probe/cache leak or representation/decision mismatch; investigate.")
# Teachers whose eval CKA may not reflect distillation quality. The low CKA
# values (0.19–0.30) co-occur with the *lowest* training val_loss in the sweep
# (0.007–0.020) — either (a) an extraction-path bug in
# evaluate_multi.extract_feats / distill_wsi_model.py:317 (e.g. taking
# last_hidden_state[:,0,:] for hibou which uses register-tokens), or (b)
# teacher-feature collapse that lets the projection head minimize cosine loss
# trivially while leaving the raw teacher space orthogonal to the student.
# Both possibilities need to be ruled out before the values are used.
EXTRACTION_BUG_TEACHERS = {"hibou-b", "hibou-l"}
# Rows where Δacc CI width is narrower than this AND n_slides=2 are
# bootstrap-degenerate (the n=2 slide-cluster bootstrap can only resample
# {AA, AB, BB}, so CI widths collapse to artificially tight intervals).
DEGENERATE_CI_WIDTH = 0.02


def _is_redacted(row) -> bool:
    """Redacted rows have their CKA/Δacc/κ/TOST values blanked in CSV + MD —
    keeping them visible would invite quoting numbers known to be wrong."""
    return (row.teacher, row.student) == ("virchow", "vit-base")


def _is_suspect_cka(row) -> bool:
    """CKA value is suspect if from a teacher family with the eval-extraction
    pattern AND the value matches that pattern (low CKA + low val_loss).
    Rows in EXTRACTION_BUG_TEACHERS but with CKA>0.5 do NOT match the pattern
    (e.g. hibou-l/vit-tiny CKA=0.799) — flag them as "same-family review" but
    don't NaN-out their CKA."""
    return (
        row.teacher in EXTRACTION_BUG_TEACHERS
        and row.cka is not None
        and row.cka < 0.5
    )


def _is_degenerate_ci(row) -> bool:
    if (row.c16lin_d_ci90_lo is None or row.c16lin_d_ci90_hi is None
            or row.c16lin_n_slides is None):
        return False
    return (row.c16lin_n_slides <= 2
            and (row.c16lin_d_ci90_hi - row.c16lin_d_ci90_lo) < DEGENERATE_CI_WIDTH)

_FLOAT = r"[-+]?\d+(?:\.\d+)?"
EVAL_RE = re.compile(r"^\[eval\]\s+(?P<teacher>\S+)\s+×\s+(?P<student>\S+)\s+(?P<ts>\S+)")
SIM_RE = re.compile(rf"^\[sim\]\s+CKA=(?P<cka>{_FLOAT})\s+cos=(?P<cos>{_FLOAT})")
C16LIN_RE = re.compile(
    rf"^\[C16 linear\]\s+T_acc=(?P<T>{_FLOAT})\s+S_acc=(?P<S>{_FLOAT})\s+"
    rf"Δacc=(?P<d>{_FLOAT})\s+\[\s*(?P<lo>{_FLOAT})\s*,\s*(?P<hi>{_FLOAT})\s*\]\s+"
    rf"κ=(?P<kappa>{_FLOAT})\s+TOST equiv=(?P<eq>True|False)\s+"
    rf"\(p_L=(?P<pL>{_FLOAT})\s+p_U=(?P<pU>{_FLOAT})\)\s+"
    rf"n_slides=(?P<n>\d+)"
)
# Terminal markers like "[OK]     conch/vit-tiny" require the FM/STU shape that
# matches an [eval] block. Bare "[skip]   conch (features already extracted)"
# (no slash) is a feature-extraction skip from a different stage and must NOT
# clobber the previous eval row's status.
TERMINAL_RE = re.compile(r"^\[(?P<kind>OK|FAIL|skip)\]\s+\S+/\S+")


@dataclass
class Row:
    teacher: str
    student: str
    eval_ts: str = ""
    cka: float | None = None
    cos: float | None = None
    c16lin_T_acc: float | None = None
    c16lin_S_acc: float | None = None
    c16lin_d_mean: float | None = None
    c16lin_d_ci90_lo: float | None = None
    c16lin_d_ci90_hi: float | None = None
    c16lin_kappa: float | None = None
    c16lin_tost_equiv: bool | None = None
    c16lin_p_lower: float | None = None
    c16lin_p_upper: float | None = None
    c16lin_n_slides: int | None = None
    status: str = ""  # OK / FAIL / skip / partial


def parse(log_path: Path) -> tuple[list[Row], list[tuple[int, int]]]:
    """Return (rows, line_ranges). line_ranges[i] = (start_line, end_line) of
    rows[i]'s eval block in the log, for reproducibility footnoting."""
    rows: list[Row] = []
    line_ranges: list[tuple[int, int]] = []
    cur: Row | None = None
    cur_start: int = 0
    with log_path.open() as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip()
            m = EVAL_RE.match(line)
            if m:
                if cur is not None:
                    rows.append(cur)
                    line_ranges.append((cur_start, lineno - 1))
                cur = Row(teacher=m["teacher"], student=m["student"], eval_ts=m["ts"])
                cur_start = lineno
                continue
            if cur is None:
                continue
            m = SIM_RE.match(line)
            if m:
                cur.cka = float(m["cka"])
                cur.cos = float(m["cos"])
                continue
            m = C16LIN_RE.match(line)
            if m:
                cur.c16lin_T_acc = float(m["T"])
                cur.c16lin_S_acc = float(m["S"])
                cur.c16lin_d_mean = float(m["d"])
                cur.c16lin_d_ci90_lo = float(m["lo"])
                cur.c16lin_d_ci90_hi = float(m["hi"])
                cur.c16lin_kappa = float(m["kappa"])
                cur.c16lin_tost_equiv = m["eq"] == "True"
                cur.c16lin_p_lower = float(m["pL"])
                cur.c16lin_p_upper = float(m["pU"])
                cur.c16lin_n_slides = int(m["n"])
                continue
            m = TERMINAL_RE.match(line)
            if m:
                cur.status = m.group("kind")
                rows.append(cur)
                line_ranges.append((cur_start, lineno))
                cur = None
                cur_start = 0
    if cur is not None:
        rows.append(cur)
        line_ranges.append((cur_start, lineno))
    # Backfill status for any row missing terminal marker.
    for r in rows:
        if not r.status:
            r.status = "partial" if r.cka is not None else "no_data"
    return rows, line_ranges


def write_csv(rows: list[Row], out: Path) -> None:
    """Writes the CSV with two derived columns so downstream code that ignores
    footnotes still gets the warnings: `cka_suspect` (bool) and `redacted`
    (bool). Suspect CKA is also nulled out in `cka` itself; redacted rows have
    all stat fields nulled. The original values are preserved in `cka_raw` /
    `c16lin_*_raw` so triage is still possible from the CSV alone."""
    out.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [f.name for f in Row.__dataclass_fields__.values()]  # type: ignore[attr-defined]
    stat_fields = [
        "cka", "cos", "c16lin_T_acc", "c16lin_S_acc", "c16lin_d_mean",
        "c16lin_d_ci90_lo", "c16lin_d_ci90_hi", "c16lin_kappa",
        "c16lin_tost_equiv", "c16lin_p_lower", "c16lin_p_upper",
        "c16lin_n_slides",
    ]
    derived = ["cka_suspect", "redacted", "degenerate_ci"]
    raw_fields = [f + "_raw" for f in stat_fields]
    fieldnames = base_fields + derived + raw_fields
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            redacted = _is_redacted(r)
            suspect = _is_suspect_cka(r)
            degenerate = _is_degenerate_ci(r)
            # Stash originals BEFORE blanking.
            for field_name in stat_fields:
                d[field_name + "_raw"] = d[field_name]
            if redacted:
                for field_name in stat_fields:
                    d[field_name] = None
            elif suspect:
                d["cka"] = None
            d["cka_suspect"] = suspect
            d["redacted"] = redacted
            d["degenerate_ci"] = degenerate
            w.writerow(d)


def _provenance(log_path: Path, script_path: Path) -> dict:
    """Provenance anchor: script SHA1, log SHA1, log line count, parse ts."""
    def _sha1(p: Path) -> str:
        h = hashlib.sha1()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=script_path.parent.parent, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_rev = "(no-git)"
    with log_path.open() as f:
        n_lines = sum(1 for _ in f)
    return {
        "parsed_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parser_script": str(script_path),
        "parser_sha1": _sha1(script_path),
        "git_rev": git_rev,
        "source_log": str(log_path),
        "source_log_sha1": _sha1(log_path),
        "source_log_lines": n_lines,
    }


def write_md(rows: list[Row], line_ranges: list[tuple[int, int]],
             out: Path, log_path: Path, script_path: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v, spec=".3f"):
        if v is None or v == "":
            return "—"
        if isinstance(v, bool):
            return "✓" if v else "✗"
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)

    prov = _provenance(log_path, script_path)
    n_total = len(rows)
    n_redacted = sum(1 for r in rows if _is_redacted(r))
    # CKA counts split redacted (parsed but unusable) from usable rows; the
    # raw count includes redacted, the usable count excludes them so reports
    # downstream of the redaction policy stay consistent.
    n_cka_raw = sum(1 for r in rows if r.cka is not None)
    n_cka_usable = sum(1 for r in rows if r.cka is not None and not _is_redacted(r))
    n_cka_pattern = sum(
        1 for r in rows
        if r.cka is not None and not _is_redacted(r)
        and r.teacher in EXTRACTION_BUG_TEACHERS and r.cka < 0.5
    )
    n_cka_family_review = sum(
        1 for r in rows
        if r.cka is not None and not _is_redacted(r)
        and r.teacher in EXTRACTION_BUG_TEACHERS and not r.cka < 0.5
    )
    n_c16_raw = sum(1 for r in rows if r.c16lin_d_mean is not None)
    n_c16_usable = sum(
        1 for r in rows
        if r.c16lin_d_mean is not None and not _is_redacted(r)
    )
    n_eq_raw = sum(
        1 for r in rows
        if r.c16lin_tost_equiv is True and not _is_redacted(r)
    )
    n_anomaly = sum(1 for r in rows if _is_anomaly(r))
    n_degenerate = sum(1 for r in rows if _is_degenerate_ci(r))

    lines: list[str] = []
    lines.append("# Partial downstream-eval recovery — INTERIM ONLY")
    lines.append("")
    # Loud blockquote disclaimer. Items use `> **(N)**` literal numbering with
    # no blank quote-line separators so GitHub markdown does not restart the
    # auto-numbered list at each item (a `> 1.` / blank `>` / `> 2.` pattern
    # renders as "1. 1. 1." on github.com).
    lines.append("> ⚠️ **WARNING — DO NOT QUOTE IN PAPER OR EXTERNAL DOCS.**")
    lines.append(">")
    lines.append("> This table is a **best-effort recovery from a crashed eval log**, "
                 "not a results table. SEVEN upstream defects further invalidate the "
                 "numbers below:")
    lines.append(">")
    lines.append("> **(1) CAMELYON16 corpus is truncated.** Local copy has only 23 WSIs "
                 "(12 normal + 11 tumor) — a *development subset* of the original "
                 "Radboud + UMCU release. **This is NOT the published CAMELYON16 "
                 "130-slide test set.** All `n_slides` values are 2 because the C16 "
                 "test partition after `(tag=='c16') & (y>=0)` filtering contains only "
                 "2 distinct slides. C16 linear accuracies are *not* comparable to "
                 "published UNI/Virchow/CONCH numbers, do not reflect multi-center "
                 "generalization, and cannot represent the canonical task.")
    lines.append(">")
    lines.append("> **(2) hibou-b / hibou-l rows have SUSPECT CKA values.** Training "
                 "val_loss = 0.007–0.020 (the lowest of any teacher in the sweep) but "
                 "CKA = 0.19–0.30 for 5 of 6 hibou rows. This is *suspicious* — "
                 "consistent with either (a) an extraction-path bug "
                 "(`distill_wsi_model.py:317` takes `last_hidden_state[:,0,:]` for "
                 "hibou, but hibou uses register tokens and the canonical CLS is via "
                 "`pooler_output`), or (b) teacher feature collapse that lets the "
                 "projection head minimize cosine loss trivially while leaving raw "
                 "teacher space orthogonal to the student. Both need to be ruled out. "
                 "Rows flagged `§` match the pattern (CKA < 0.5) and are nulled-out in "
                 "the CSV; rows flagged `§?` are in the same teacher family but do NOT "
                 "match the pattern (kept in CSV — review without contamination).")
    lines.append(">")
    lines.append("> **(3) Training was `L_CLS + 0.5·L_PAT` — NOT full v4.** Every "
                 "`training_log.json` shows `use_mgd: False, use_dino: False`. "
                 "The runs in this table are the v4-ablation \"L_CLS + L_PAT only\" "
                 "configuration, not the full v4 loss "
                 "`(L_CLS + L_PAT + L_MGD + L_DINO)` described in `proposal_v4.md`. "
                 "Any claim about \"v4 distillation\" must be qualified accordingly.")
    lines.append(">")
    lines.append("> **(4) Two rows have empty stats (`†`).** Their eval crashed "
                 "*before* the `[sim] CKA` print — not because training is missing. "
                 "All 36 runs have `splits.npz` + `training_log.json` + `best.pt` on "
                 "disk (verified 2026-05-26; the stale `slide_split_audit.json` from "
                 "2026-04-24 showed 15 runs as \"missing\" but those completed "
                 "training 4/24–5/01).")
    lines.append(">")
    lines.append("> **(5) `cos` ≈ 0 with CKA up to 0.94.** Student embed dim 256 vs "
                 "teacher 768–1536; cosine is computed between unaligned spaces and "
                 "is not informative. **Ignore the `cos` column.**")
    lines.append(">")
    lines.append("> **(6) TOST is not interpretable at `n_slides=2`.** The "
                 "slide-cluster bootstrap is degenerate — only 3 distinct resample "
                 "compositions {AA, AB, BB} exist, so p-values collapse to "
                 "{0, 0.25, 0.5, 0.75, 1.0} and CI widths can be artificially tight "
                 "(rows flagged `※` below have width < 0.02). The `TOST` column is "
                 "bookkeeping only.")
    chance_rows = [
        f"{i}"
        for i, r in enumerate(rows, 1)
        if r.c16lin_T_acc is not None
        and not _is_redacted(r)
        and abs(r.c16lin_T_acc - 0.5) <= 0.10
    ]
    chance_list = ", ".join(chance_rows) if chance_rows else "(none)"
    lines.append(">")
    lines.append("> **(7) T_acc near 0.5 is at-chance baseline.** With `n_slides=2`, "
                 "the C16 test set is one tumor + one normal slide-cluster. Predicting "
                 "the majority class per slide trivially yields T_acc ≈ 0.5. Rows "
                 f"with T_acc within ±0.10 of 0.50 (rows {chance_list}; "
                 f"{len(chance_rows)} of {n_c16_usable} usable = "
                 f"{100*len(chance_rows)/max(n_c16_usable,1):.0f}%, redacted rows "
                 "excluded) are operationally indistinguishable from a constant "
                 "predictor. Do not interpret S_acc differences against such teachers.")
    lines.append("")
    lines.append("## Recovery stats")
    lines.append("")
    lines.append(f"- Eval blocks parsed: **{n_total}**")
    lines.append(f"- Rows redacted (`‡` with stats fully blanked): **{n_redacted}** "
                 "(excluded from all counts below; raw values preserved in `*_raw` "
                 "CSV columns)")
    lines.append(f"- CKA recovered: raw **{n_cka_raw}** → **{n_cka_usable}** usable "
                 f"after redaction")
    lines.append(f"  - of which **{n_cka_pattern}** match the hibou val_loss/CKA "
                 "inversion pattern (CKA < 0.5; nulled in CSV — see §2)")
    lines.append(f"  - **{n_cka_family_review}** is in `hibou-*` family but does NOT "
                 "match the pattern (CKA ≥ 0.5; kept in CSV, flagged `§?` for review)")
    lines.append(f"- C16 linear recovered: raw **{n_c16_raw}** → **{n_c16_usable}** "
                 f"usable after redaction (of which **{n_degenerate}** have "
                 f"degenerate CI width < {DEGENERATE_CI_WIDTH} at n_sl=2 — "
                 "flagged `※`)")
    lines.append(f"- Other anomaly flags (`‡` rows kept in MD but flagged): "
                 f"**{n_anomaly - n_redacted}**")
    lines.append(f"- TOST equivalent at margin 0.03 over usable rows (raw count, "
                 f"**not interpretable** — see §6): {n_eq_raw}/{n_c16_usable}")
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append("Columns:")
    lines.append("- **CKA** = linear-CKA between teacher and student test-set features. "
                 "Blanked to `—` for rows flagged `§` (suspect; see §2).")
    lines.append("- **cos** = raw cosine between teacher/student means "
                 "(uninformative — see §5).")
    lines.append("- **T_acc / S_acc** = CAMELYON16 tumor/normal linear-probe accuracy. "
                 "**No per-arm CIs in this table** (only Δacc CI was printed pre-crash; "
                 "per-arm CIs are in the JSON when rerun completes).")
    lines.append("- **Δacc** = S_acc − T_acc.")
    lines.append("- **CI90** = 90% bootstrap CI on Δacc, slide-cluster resample over "
                 "`n_sl` slides. Tight CIs at `n_sl=2` are bootstrap artifacts, not "
                 "precision; see `※` flag.")
    lines.append("- **κ** = Cohen's kappa, **point estimate only** "
                 "(no CI; not interpretable at n_sl=2).")
    lines.append("- **TOST** = TOST equivalence verdict at margin 0.03 (`✓`/`✗`). "
                 "**Mechanically degenerate at n_sl=2** — bookkeeping only.")
    lines.append("- **n_sl** = number of unique slides in the C16 test cluster.")
    lines.append("- **`—`** in any numeric cell means the value is *missing* "
                 "(eval crashed, redacted, or blanked for `§` suspect) — **not zero**.")
    lines.append("")
    lines.append("Flags column legend:")
    lines.append("- `†` = eval crashed before CKA print "
                 "(see `log lines` column for the crash range).")
    lines.append(f"- `‡` = **screening flag**, NOT a hypothesis test. Rule: "
                 f"raw CKA ≥ {ANOMALY_CKA_MIN} (≈high alignment vs sweep median) "
                 f"AND |Δacc| ≥ {ANOMALY_DELTA_MIN} (≈3× the TOST equivalence "
                 f"margin) AND Δacc CI strictly excludes 0 AND CI is NOT "
                 f"`※`-degenerate. Catches both S≫T and S≪T directions. "
                 "Thresholds are ad-hoc cutoffs to surface rows for human "
                 "review, not significance levels. Row footnoted with `‡: …` "
                 "inline. If row is also in the redaction set, all stats are "
                 "blanked.")
    lines.append("- `§` = teacher in `hibou-*` family AND row matches val_loss/CKA "
                 "inversion pattern → CKA nulled in CSV; raw value preserved in "
                 "`cka_raw`. `§?` means same-family but value does NOT match pattern "
                 "(needs review but kept).")
    lines.append("- `※` = bootstrap-degenerate CI width < "
                 f"{DEGENERATE_CI_WIDTH} at n_sl=2; CI bounds are artifacts.")
    lines.append("")
    lines.append("| # | Teacher | Student | CKA | cos | T_acc | S_acc | Δacc | "
                 "CI90 | κ | TOST | n_sl | log lines | flags |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|---:|:---:|---:|---:|---|")
    for i, (r, (lstart, lend)) in enumerate(zip(rows, line_ranges), 1):
        redacted = _is_redacted(r)
        suspect = _is_suspect_cka(r)
        degenerate = _is_degenerate_ci(r)
        # Compute display values: blank stats for redacted rows; blank CKA only
        # for suspect rows; preserve everything else.
        def disp(value, redacted_blank=True, suspect_blanks=False, spec=".3f"):
            if redacted and redacted_blank:
                return "—"
            if suspect_blanks and suspect:
                return "—"
            return fmt(value, spec)

        ci_display = (
            "—" if redacted or r.c16lin_d_ci90_lo is None
            else f"[{fmt(r.c16lin_d_ci90_lo, '+.3f')}, "
                 f"{fmt(r.c16lin_d_ci90_hi, '+.3f')}]"
        )

        flags: list[str] = []
        flag_inline_notes: list[str] = []
        if r.cka is None and not redacted:
            flags.append("†")
        if redacted or _is_anomaly(r):
            flags.append("‡")
            flag_inline_notes.append(f"‡: {_anomaly_footnote(r)}")
        if r.teacher in EXTRACTION_BUG_TEACHERS:
            flags.append("§" if suspect else "§?")
        if degenerate:
            flags.append("※")
        flag_str = " ".join(flags) if flags else ""

        cka_str = disp(r.cka, suspect_blanks=True)
        lines.append(
            f"| {i} | {r.teacher} | {r.student} | "
            f"{cka_str} | {disp(r.cos)} | "
            f"{disp(r.c16lin_T_acc)} | {disp(r.c16lin_S_acc)} | "
            f"{disp(r.c16lin_d_mean, spec='+.3f')} | {ci_display} | "
            f"{disp(r.c16lin_kappa)} | "
            f"{('—' if redacted else fmt(r.c16lin_tost_equiv))} | "
            f"{disp(r.c16lin_n_slides, spec='d')} | "
            f"{lstart}–{lend} | {flag_str} |"
        )
        for note in flag_inline_notes:
            lines.append(f"| | | | | | | | | | | | | | _{note}_ |")

    # Provenance anchor
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- parsed at: `{prov['parsed_at_utc']}`")
    lines.append(f"- parser: `{prov['parser_script']}` (sha1 `{prov['parser_sha1']}`, "
                 f"git `{prov['git_rev']}`)")
    lines.append(f"- source log: `{prov['source_log']}` "
                 f"({prov['source_log_lines']} lines, sha1 `{prov['source_log_sha1']}`)")
    lines.append("- per-row `log lines` column points back to the originating "
                 "`[eval]` … `[OK]`/`[FAIL]` block in the source log.")
    lines.append("- CSV emits both blanked values (`cka`, `c16lin_*`) and `*_raw` "
                 "columns preserving the original numbers, plus derived bools "
                 "`cka_suspect`, `redacted`, `degenerate_ci`.")

    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="outputs/downstream_eval_queue.log")
    ap.add_argument("--out-csv", default="outputs/v4_full/partial_eval_summary.csv")
    ap.add_argument("--out-md", default="outputs/v4_full/partial_eval_summary.md")
    args = ap.parse_args()

    log_path = Path(args.log)
    rows, line_ranges = parse(log_path)
    write_csv(rows, Path(args.out_csv))
    write_md(rows, line_ranges, Path(args.out_md), log_path,
             Path(__file__).resolve())
    print(f"[partial-eval] {len(rows)} rows parsed from {log_path}")
    print(f"[partial-eval]   CKA filled: {sum(1 for r in rows if r.cka is not None)}")
    print(f"[partial-eval]   C16-lin filled: {sum(1 for r in rows if r.c16lin_d_mean is not None)}")
    print(f"[partial-eval] -> {args.out_csv}")
    print(f"[partial-eval] -> {args.out_md}")


if __name__ == "__main__":
    main()
