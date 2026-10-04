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
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    report["_stamp"] = f"Built {datetime.datetime.now():%Y-%m-%d %H:%M} from {args.report.name}"
    if args.notes and args.notes.is_file():
        report["_notes"] = json.loads(args.notes.read_text())
    data = json.dumps(report).replace("</", "<\\/")
    args.out.write_text(TEMPLATE.read_text().replace("__DATA__", data))
    print(f"[page] {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
