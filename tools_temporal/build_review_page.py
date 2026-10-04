"""Fill the dataset review page (tools_temporal/review_page_template.html) with report_dataset.py's JSON.

    python tools_temporal/build_review_page.py --report report.json --out page.html [--notes notes.json]

``--notes``: a JSON list of extra findings (strings, HTML allowed) appended to
the page's findings, after the ones computed from the numbers.
"""

import argparse
import datetime
import json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "review_page_template.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--run", type=Path, help="a training run directory (summary.json, eval_val.json)")
    parser.add_argument("--results", type=Path, help="render_results.py output")
    parser.add_argument("--eval", action="append", default=[],
                        help="TITLE=path/to/eval_temporal.json[:DESCRIPTION], shown as comparison tables in order")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    report["_stamp"] = f"Built {datetime.datetime.now():%Y-%m-%d %H:%M} from {args.report.name}"
    if args.notes and args.notes.is_file():
        report["_notes"] = json.loads(args.notes.read_text())
    if args.run:
        run = {"name": args.run.name}
        for key, file in (("summary", "summary.json"), ("eval", "eval_val.json")):
            if (args.run / file).is_file():
                run[key] = json.loads((args.run / file).read_text())
        if "summary" in run:
            run["summary"].pop("training", None)
            best = run["summary"].get("best_step")
            for line in (args.run / "log.jsonl").read_text().splitlines():
                entry = json.loads(line)
                if entry.get("step") == best and "val" in entry:
                    run["by_corpus"] = entry["val"].get("by_corpus")
        report["_run"] = run
    report["_evals"] = []
    for spec in args.eval:
        title, rest = spec.split("=", 1)
        path, _, desc = rest.partition(":")
        report["_evals"].append({"title": title, "desc": desc, "data": json.loads(Path(path).read_text())})
    if args.results and args.results.is_file():
        report["_results"] = json.loads(args.results.read_text())
    data = json.dumps(report).replace("</", "<\\/")
    args.out.write_text(TEMPLATE.read_text().replace("__DATA__", data))
    print(f"[page] {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
