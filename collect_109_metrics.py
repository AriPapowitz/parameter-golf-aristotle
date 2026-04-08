import argparse
import csv
import re
from pathlib import Path


def parse_log(path: Path) -> dict[str, str]:
    out = {
        "pre_quant_val_bpb": "",
        "final_roundtrip_val_bpb": "",
        "worst_rel_mse_top": "",
        "submission_bytes": "",
    }
    best_pre = None
    final = None
    worst = None
    sub_bytes = None

    pre_re = re.compile(r"step:\d+/\d+ val_loss:[^ ]+ val_bpb:([0-9.]+)")
    final_re = re.compile(r"final_[^ ]+_roundtrip_exact val_loss:[^ ]+ val_bpb:([0-9.]+)")
    worst_re = re.compile(r"worst_rel_mse_top10:([^ ]+)")
    bytes_re = re.compile(r"Total submission size [^:]+: (\d+) bytes")

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pre_re.search(line)
        if m:
            v = float(m.group(1))
            best_pre = v if best_pre is None else min(best_pre, v)
        m = final_re.search(line)
        if m:
            final = float(m.group(1))
        m = worst_re.search(line)
        if m:
            worst = m.group(1)
        m = bytes_re.search(line)
        if m:
            sub_bytes = int(m.group(1))

    if best_pre is not None:
        out["pre_quant_val_bpb"] = f"{best_pre:.8f}"
    if final is not None:
        out["final_roundtrip_val_bpb"] = f"{final:.8f}"
    if worst is not None:
        out["worst_rel_mse_top"] = worst
    if sub_bytes is not None:
        out["submission_bytes"] = str(sub_bytes)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to logs/<run_id>.txt")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--tracker", default="RESULTS_109_TRACKER.csv")
    args = ap.parse_args()

    log_path = Path(args.log)
    tracker_path = Path(args.tracker)
    row = {
        "run_id": args.run_id,
        "seed": str(args.seed),
        "stage": str(args.stage),
        **parse_log(log_path),
        "notes": "",
    }

    rows = []
    if tracker_path.exists():
        with tracker_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    replaced = False
    for i, r in enumerate(rows):
        if r.get("run_id") == row["run_id"]:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with tracker_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
