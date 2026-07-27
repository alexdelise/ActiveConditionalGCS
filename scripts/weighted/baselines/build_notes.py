#!/usr/bin/env python3
"""Build the live priority baseline queues and deferred VDHH backlog."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTES_ROOT = ROOT / "notestoself" / "weighted"
CURRENTLY_RUNNING = NOTES_ROOT / "currentlyrunning.md"
DEDICATED_NOTES = NOTES_ROOT / "sampling_baselines.md"
BEGIN_MARKER = "<!-- BEGIN WEIGHTED BASELINE PHASE -->"
END_MARKER = "<!-- END WEIGHTED BASELINE PHASE -->"

FAMILIES = ("prompt_matched", "prompt_mismatched", "out_of_range")
RECOVERIES = ("unprompted", "daytime_beach", "sunset_beach", "cat")
ACTIVE_MCS: dict[int, tuple[tuple[str, str, str], ...]] = {
    104: (
        ("last2", "out_of_range", "daytime_beach"),
    ),
    105: (
        ("first3", "prompt_matched", "daytime_beach"),
        ("last2", "prompt_mismatched", "unprompted"),
        ("first3", "prompt_mismatched", "cat"),
        ("last2", "out_of_range", "sunset_beach"),
    ),
    114: (
        ("last2", "prompt_matched", "unprompted"),
        ("first3", "prompt_matched", "cat"),
        ("last2", "prompt_mismatched", "sunset_beach"),
        ("first3", "out_of_range", "daytime_beach"),
    ),
    115: (
        ("last2", "prompt_matched", "daytime_beach"),
        ("first3", "prompt_mismatched", "unprompted"),
        ("last2", "prompt_mismatched", "cat"),
        ("first3", "out_of_range", "sunset_beach"),
    ),
}
ACTIVE_MCS_SESSIONS: dict[int, tuple[str, ...]] = {
    104: ("recon4",),
    105: ("recon", "recon2", "recon3", "recon4"),
    114: ("recon", "recon2", "recon3", "recon4"),
    115: ("recon", "recon2", "recon3", "recon4"),
}

# These four MCS jobs were formerly queued behind the K-tilde study. Class113
# completed its original 50-row MCS assignment overnight, so it can run them
# immediately without waiting for classes107 or 108 to recover.
CLASS113_MORNING_MCS: tuple[tuple[str, str, str], ...] = (
    ("first3", "prompt_mismatched", "daytime_beach"),
    ("last2", "prompt_matched", "sunset_beach"),
    ("first3", "out_of_range", "cat"),
    ("last2", "out_of_range", "unprompted"),
)

CLASS104_IMMEDIATE_INVERSE: tuple[tuple[str, str, str, str], ...] = (
    ("inverse_square", "first3", "prompt_matched", "unprompted"),
    ("inverse_square", "first3", "prompt_mismatched", "sunset_beach"),
    ("inverse_square", "first3", "prompt_matched", "daytime_beach"),
)

# These queues begin only after the five-trial K-tilde jobs finish. The MCS
# commands formerly assigned to classes107 and 108 have moved to class113.
POST_KTILDE: dict[int, tuple[tuple[str, str, str, str], ...]] = {
    108: (
        ("inverse_square", "first3", "prompt_mismatched", "cat"),
        ("inverse_square", "first3", "prompt_matched", "sunset_beach"),
        ("inverse_square", "first3", "out_of_range", "unprompted"),
    ),
    109: (
        ("inverse_square", "first3", "prompt_matched", "cat"),
        ("inverse_square", "first3", "out_of_range", "daytime_beach"),
        ("inverse_square", "last2", "prompt_matched", "cat"),
        ("inverse_square", "last2", "out_of_range", "daytime_beach"),
        ("inverse_square", "last2", "prompt_mismatched", "unprompted"),
        ("inverse_square", "last2", "out_of_range", "sunset_beach"),
    ),
    110: (
        ("inverse_square", "first3", "prompt_mismatched", "unprompted"),
        ("inverse_square", "first3", "out_of_range", "sunset_beach"),
        ("inverse_square", "last2", "prompt_mismatched", "daytime_beach"),
        ("inverse_square", "last2", "out_of_range", "cat"),
        ("inverse_square", "last2", "prompt_matched", "unprompted"),
        ("inverse_square", "last2", "prompt_mismatched", "sunset_beach"),
    ),
    111: (
        ("inverse_square", "first3", "prompt_mismatched", "daytime_beach"),
        ("inverse_square", "first3", "out_of_range", "cat"),
        ("inverse_square", "last2", "prompt_matched", "daytime_beach"),
        ("inverse_square", "last2", "prompt_mismatched", "cat"),
        ("inverse_square", "last2", "prompt_matched", "sunset_beach"),
        ("inverse_square", "last2", "out_of_range", "unprompted"),
    ),
}

LAUNCHERS = {
    "mcs": "run_mcs_split.sh",
    "inverse_square": "run_inverse_square_split.sh",
    "vdhh": "run_vdhh_split.sh",
}


def command(launcher: str, family: str, split: str, recovery: str) -> str:
    return (
        f"./scripts/weighted/baselines/{launcher} "
        f"{family} {split} {recovery}"
    )


def append_job(
    lines: list[str],
    *,
    scheme: str,
    split: str,
    family: str,
    recovery: str,
    status: str = " ",
) -> None:
    label = (
        f"{family.replace('_', ' ').title()}, "
        f"{recovery.replace('_', ' ').title()}, {split}"
    )
    lines.extend(
        [
            f"- [{status}] {label}",
            "",
            "  ```bash",
            f"  {command(LAUNCHERS[scheme], family, split, recovery)}",
            "  ```",
            "",
        ]
    )


def completed_archive_lines() -> list[str]:
    completed_mcs = (
        (104, "recon", "Prompt-matched, unprompted, first3 — 15 / 15", "prompt_matched", "first3", "unprompted"),
        (104, "recon2", "Prompt-matched, cat, last2 — 10 / 10", "prompt_matched", "last2", "cat"),
        (104, "recon3", "Prompt-mismatched, sunset beach, first3 — 15 / 15", "prompt_mismatched", "first3", "sunset_beach"),
        (113, "recon", "Prompt-matched, sunset beach, first3 — 15 / 15", "prompt_matched", "first3", "sunset_beach"),
        (113, "recon2", "Prompt-mismatched, daytime beach, last2 — 10 / 10", "prompt_mismatched", "last2", "daytime_beach"),
        (113, "recon3", "Out-of-range, unprompted, first3 — 15 / 15", "out_of_range", "first3", "unprompted"),
        (113, "recon4", "Out-of-range, cat, last2 — 10 / 10", "out_of_range", "last2", "cat"),
    )
    lines = [
        "---",
        "",
        "# Completed Runs",
        "",
        "Only commands whose full expected artifact count has been validated belong",
        "in this archive. The producing computer is recorded explicitly.",
        "",
    ]
    for computer in (104, 113):
        lines.extend([f"## class{computer} — Uniform MCS", ""])
        for saved_computer, session, label, family, split, recovery in completed_mcs:
            if saved_computer != computer:
                continue
            lines.extend(
                [
                    f"- [completed] `{session}` — {label}",
                    "",
                    "  ```bash",
                    f"  {command(LAUNCHERS['mcs'], family, split, recovery)}",
                    "  ```",
                    "",
                ]
            )
    lines.extend(["## class111 — K-Tilde Convergence Trial 5", ""])
    for session, role, label in (
        ("ktilde", "k0", "Unconditioned"),
        ("ktilde2", "k1", "Daytime-beach"),
        ("ktilde3", "k2", "Sunset-beach"),
        ("ktilde4", "k4", "Cat"),
    ):
        lines.extend(
            [
                f"- [completed] `{session}` — {label} K-tilde — 10000 / 10000",
                "",
                "  ```bash",
                f"  ./scripts/weighted/ktilde_convergence/run_trial.sh {role} 5",
                "  ```",
                "",
            ]
        )
    return lines


def queue_markdown(*, include_completed: bool = False) -> str:
    lines = [
        BEGIN_MARKER,
        "",
        "# Phase 2: Weighted Sampling Baselines",
        "",
        "Completed commands have been removed from these computer queues and",
        "placed in the completed-runs archive at the bottom of",
        "`currentlyrunning.md`. Class113 completed its original assignment and",
        "is now the morning home for the four remaining MCS commands. Three",
        "inverse-square commands move immediately to class104's completed GPU",
        "slots; the rest remain queued after K-tilde. Class112 is unavailable",
        "and has no assigned work.",
        "",
        "VDHH/half-half is deferred and appears only in the unassigned backlog at",
        "the very end of this section.",
        "",
        "## Common Setup",
        "",
        "```bash",
        "cd -P /home/ard22l/HITM/refactor/ActiveConditionalGCS",
        "test -f run_conditioning_regression.py",
        "test -x scripts/weighted/baselines/run_mcs_split.sh",
        "test -x scripts/weighted/baselines/run_inverse_square_split.sh",
        "```",
        "",
        "No project-root, prior, or Python export is required.",
        "",
        "## MCS Computer Queues",
        "",
        "Every unfinished command retains a blank checkbox. A blank checkbox",
        "means queued or requiring a process check, not confirmed running.",
        "",
    ]
    for computer, jobs in ACTIVE_MCS.items():
        lines.extend([f"## class{computer}", ""])
        if computer in {105, 114, 115}:
            lines.extend(["**STALLED**", ""])
        for session, (split, family, recovery) in zip(
            ACTIVE_MCS_SESSIONS[computer], jobs
        ):
            append_job(
                lines,
                scheme="mcs",
                split=split,
                family=family,
                recovery=recovery,
                status=(
                    f"{session} STALLED"
                    if computer in {105, 114, 115}
                    else f"{session} in progress"
                ),
            )
    lines.extend(
        [
            "## class113 — MCS In Progress",
            "",
            "The original class113 assignment is archived below at 50 / 50.",
            "These four replacement MCS sessions are now running.",
            "",
        ]
    )
    for index, (split, family, recovery) in enumerate(
        CLASS113_MORNING_MCS, start=1
    ):
        session = "recon" if index == 1 else f"recon{index}"
        append_job(
            lines,
            scheme="mcs",
            split=split,
            family=family,
            recovery=recovery,
            status=f"{session} in progress",
        )
    lines.extend(
        [
            "## class104 — Inverse-Square In Progress",
            "",
            "The prior completed MCS commands are archived below. The reused",
            "`recon`, `recon2`, and `recon3` sessions now run inverse-square;",
            "`recon4` is continuing the formerly stalled MCS command above.",
            "",
        ]
    )
    for index, (scheme, split, family, recovery) in enumerate(
        CLASS104_IMMEDIATE_INVERSE, start=1
    ):
        session = "recon" if index == 1 else f"recon{index}"
        append_job(
            lines,
            scheme=scheme,
            split=split,
            family=family,
            recovery=recovery,
            status=f"{session} in progress",
        )
    lines.extend(
        [
            "## Post-K-Tilde Priority Queues",
            "",
            "Start these only after the four K-tilde jobs on the named computer",
            "finish. All remaining jobs in these queues are pure inverse-square.",
            "Run at most four commands concurrently; start the remaining queued",
            "commands as GPU slots become free.",
            "",
        ]
    )
    for computer, jobs in POST_KTILDE.items():
        lines.extend([f"## class{computer} — after K-tilde", ""])
        if computer in {107, 108, 109, 110}:
            lines.extend(["**STALLED — K-TILDE TRIAL INCOMPLETE**", ""])
        for scheme, split, family, recovery in jobs:
            append_job(
                lines,
                scheme=scheme,
                split=split,
                family=family,
                recovery=recovery,
            )
    lines.extend(
        [
            "## class112",
            "",
            "No commands are assigned. Nothing has been started on class112.",
            "",
            "## Dry-Run Validation",
            "",
            "```bash",
            "./scripts/weighted/baselines/list_all.sh",
            "```",
            "",
            "This validates the full method grid. Its legacy per-computer summary",
            "does not describe the reorganized live queues above.",
            "",
            "## Live Computer Status",
            "",
            "| Computer | Active command | State | Started | Last saved run | Notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    mcs_status = {
        104: ("Four sessions in progress: three inverse-square and one MCS", "6 / 10 active MCS; 10 / 45 inverse-square", "2026-07-25 17:05 EDT"),
        105: ("STALLED — no route; restart did not save a new leaf", "5 / 50", "2026-07-24 08:53 EDT"),
        113: ("Four reassigned MCS sessions in progress", "17 / 50 new MCS", "2026-07-25 16:52 EDT"),
        114: ("STALLED — no route", "3 / 50", "2026-07-24 08:29 EDT"),
        115: ("STALLED — no route", "3 / 50", "2026-07-24 08:32 EDT"),
    }
    mcs_queue_labels = {
        104: "One remaining uniform-MCS command",
        105: "Four uniform-MCS commands",
        113: "Four reassigned uniform-MCS commands",
        114: "Four uniform-MCS commands",
        115: "Four uniform-MCS commands",
    }
    for computer, (state, saved, last) in mcs_status.items():
        lines.append(
            f"| class{computer} | {mcs_queue_labels[computer]} | {state} | — | {last} | {saved} rows saved |"
        )
    ktilde_status = {
        107: "STALLED — no route; inverse-square assignment moved to class104",
        108: "STALLED — no route; Trial 2 remains incomplete near 5110",
        109: "STALLED — no route; Trial 3 remains incomplete near 5100",
        110: "STALLED — no route; Trial 4 remains incomplete near 4960",
    }
    for computer, note in ktilde_status.items():
        lines.append(
            f"| class{computer} | Post-K-tilde MCS/inverse-square queue | Waiting | — | — | {note} |"
        )
    lines.append("| class112 | — | Unavailable | — | — | No work started or assigned |")
    lines.extend(
        [
            "",
            "## Baseline Progress",
            "",
            "| Priority | Method | Saved / expected | Left | Last saved run | Notes |",
            "| ---: | --- | ---: | ---: | --- | --- |",
            "| 1 | Uniform MCS | 124 / 300 | 176 | 2026-07-25 17:05 EDT | class104 and class113 replacement sessions in progress |",
            "| 1 | Pure inverse-square | 10 / 300 | 290 | 2026-07-25 15:52 EDT | 3 commands running on class104; 21 remain queued after K-tilde |",
            "| 2 | VDHH | 0 / 300 | 300 | — | Deferred; no computer assignment |",
            "|  | **All baselines** | **134 / 900** | **766** | **2026-07-25 17:05 EDT** | **MCS and inverse-square active; VDHH deferred** |",
            "",
            "## Post-Run Notebook Commands",
            "",
            "The notebooks support partial runs, so these may also be used after MCS",
            "and inverse-square finish but before the lower-priority VDHH queue.",
            "",
            "```bash",
            "python scripts/weighted/build_notebooks.py",
            "jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 analyze_results/weighted/main/prompt_matched_results.ipynb",
            "jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 analyze_results/weighted/main/prompt_mismatched_results.ipynb",
            "jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 analyze_results/weighted/main/out_of_range_results.ipynb",
            "```",
            "",
        ]
    )
    if include_completed:
        lines.extend(completed_archive_lines())
    lines.extend(
        [
            "## Deferred VDHH / Half-Half Backlog",
            "",
            "These commands are intentionally unassigned and lowest priority. Do not",
            "place them under a computer queue unless the MCS and inverse-square",
            "studies finish and time remains.",
            "",
        ]
    )
    for family in FAMILIES:
        lines.extend([f"### {family.replace('_', ' ').title()}", ""])
        for recovery in RECOVERIES:
            for split in ("first3", "last2"):
                append_job(
                    lines,
                    scheme="vdhh",
                    split=split,
                    family=family,
                    recovery=recovery,
                )
    lines.extend(
        [
            END_MARKER,
            "",
        ]
    )
    return "\n".join(lines)


def dedicated_markdown(queue: str) -> str:
    prefix = """# Weighted MCS, Pure Inverse-Square, and VDHH Baselines

This is the canonical run sheet for the 900 new main-experiment baseline
reconstructions. CFG ablations are not part of this addition.

## Sampling Definitions

- Uniform MCS uses the original law $\\mu(i)=1/n$.
- Pure inverse-square uses
  $\\mu(i)\\propto(1+u_i^2+v_i^2)^{-1}$ with no uniform mixture.
- VDHH includes a complete centered disk with at most $\\lfloor m/2\\rfloor$
  points, then samples uniformly outside. Its saved weights use exact
  first-order inclusion probabilities for this two-stratum design.
- All three use the unitary FFT and weighted least squares. Their baseline
  configs set `probability_regularization_zeta=0`.

## Legacy Port Notes

The implementation was traced to `../Old/src/config.py` and
`../Old/src/sampling.py`. Legacy method 6 supplied the centered low-frequency
disk plus a uniform outside draw, while legacy method 10 supplied the
inverse-square kernel. This suite intentionally changes two recovery laws:

- Method 10 uses the pure normalized inverse-square law instead of the old
  50/50 inverse-square/uniform mixture.
- VDHH keeps the disk/outside design, caps the largest complete disk shell at
  `floor(m/2)`, and replaces the old approximate global-mixture weights with
  exact stratum inclusion weights.

At the smallest 512-by-512 rate, $m=328$: the complete-shell rule uses 161
disk coefficients and 167 outside coefficients, so the outside stratum remains
large enough at the most extreme subsampling point.

## Result Tags

```text
results/weighted/<family>/sunset/<first3|last2>_<scheme>_recover_<recovery>/
```

The three weighted main notebooks ingest these tags together with the existing
four CS S10000 distributions.

"""
    return prefix + queue.replace(BEGIN_MARKER + "\n\n", "").replace(
        "\n" + END_MARKER + "\n", "\n"
    )


def replace_marked_section(existing: str, replacement: str) -> str:
    if BEGIN_MARKER in existing and END_MARKER in existing:
        before = existing.split(BEGIN_MARKER, 1)[0].rstrip()
        after = existing.split(END_MARKER, 1)[1].lstrip()
        return before + "\n\n" + replacement + ("\n" + after if after else "")
    return existing.rstrip() + "\n\n---\n\n" + replacement


def main() -> None:
    queue = queue_markdown(include_completed=False)
    current_queue = queue_markdown(include_completed=True)
    NOTES_ROOT.mkdir(parents=True, exist_ok=True)
    DEDICATED_NOTES.write_text(dedicated_markdown(queue), encoding="utf-8")
    existing = CURRENTLY_RUNNING.read_text(encoding="utf-8")
    CURRENTLY_RUNNING.write_text(
        replace_marked_section(existing, current_queue),
        encoding="utf-8",
    )
    print(DEDICATED_NOTES.relative_to(ROOT))
    print(CURRENTLY_RUNNING.relative_to(ROOT))


if __name__ == "__main__":
    main()
