"""Render a coverage.json report as a Markdown summary for a pull request comment.

Usage: python coverage_summary.py coverage.json [--floor 90] [--label "..."]

Writes Markdown to stdout. Keeps the table compact: files are listed worst-first so
the rows that need attention are the ones you see, and fully covered files are folded
into a single line rather than padding the table.
"""

import argparse
import json
import sys

# The comment is located by this marker on later runs so it can be updated in place
# rather than posting a new comment per push.
MARKER = "<!-- trustee-coverage-report -->"


def bar(percent, width=20):
    filled = round(percent / 100 * width)
    return "█" * filled + "░" * (width - filled)


def format_missing(missing, limit=6):
    """Condense a list of line numbers into ranges, truncated for readability."""
    if not missing:
        return ""
    ranges = []
    start = previous = missing[0]
    for line in missing[1:]:
        if line == previous + 1:
            previous = line
            continue
        ranges.append((start, previous))
        start = previous = line
    ranges.append((start, previous))

    rendered = [str(a) if a == b else f"{a}–{b}" for a, b in ranges]
    if len(rendered) > limit:
        return ", ".join(rendered[:limit]) + f", +{len(rendered) - limit} more"
    return ", ".join(rendered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--floor", type=float, default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    with open(args.report) as handle:
        data = json.load(handle)

    totals = data["totals"]
    percent = totals["percent_covered"]

    out = [MARKER, "## Coverage report", ""]

    if args.floor is not None:
        met = percent >= args.floor
        verdict = "meets" if met else "is below"
        icon = "✅" if met else "❌"
        out.append(f"{icon} **{percent:.2f}%** {verdict} the {args.floor:g}% floor.")
    else:
        out.append(f"**{percent:.2f}%** overall.")

    out += [
        "",
        f"`{bar(percent)}`  {totals['covered_lines']:,} / {totals['num_statements']:,} statements"
        f"  ·  {totals['missing_lines']:,} uncovered",
        "",
    ]

    files = []
    for path, entry in data["files"].items():
        summary = entry["summary"]
        if summary["num_statements"] == 0:
            continue  # empty __init__.py and friends carry no signal
        files.append((path, summary, entry.get("missing_lines", [])))

    partial = sorted(
        (f for f in files if f[1]["percent_covered"] < 100),
        key=lambda f: (f[1]["percent_covered"], -f[1]["missing_lines"]),
    )
    complete = [f for f in files if f[1]["percent_covered"] >= 100]

    if partial:
        out += [
            "| File | Coverage | Missed | Uncovered lines |",
            "| :--- | -------: | -----: | :-------------- |",
        ]
        for path, summary, missing in partial:
            out.append(
                f"| `{path}` | {summary['percent_covered']:.0f}% "
                f"| {summary['missing_lines']} | {format_missing(missing)} |"
            )
        out.append("")

    if complete:
        names = ", ".join(f"`{path}`" for path, _, _ in sorted(complete))
        out += [f"<details><summary>{len(complete)} file(s) at 100%</summary>", "", names, "", "</details>", ""]

    if args.label:
        out += ["", f"<sub>Measured on {args.label}.</sub>"]

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
