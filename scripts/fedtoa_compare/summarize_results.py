#!/usr/bin/env python3
"""Summarize FedToA/FedAvg logs into CSV + Markdown tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean

RUN_RE = re.compile(r"^\[RUN_CONFIG\]\s+(?P<k>[a-zA-Z0-9_]+)=(?P<v>.*)$")
MET_RE = re.compile(
    r"\[FEDTOA\]\[TRAIN_METRICS\].*task_loss=(?P<task>[-+0-9.eE]+).*topo_loss_used=(?P<topo>[-+0-9.eE]+).*scaled_topo_term=(?P<scaled>[-+0-9.eE]+).*spec_loss=(?P<spec>[-+0-9.eE]+).*active_edge_count=(?P<edges>\d+)"
)
BLUE_RE = re.compile(r"\[FEDTOA\]\[BLUEPRINT\].*retained_density=(?P<density>[-+0-9.eE]+)")
COMM_RE = re.compile(
    r"\[FEDTOA\]\[COMM\].*round_total_bytes=(?P<round>\d+).*cumulative_total_bytes=(?P<cum>\d+)"
)
RSUM_RE = re.compile(r"rsum[^0-9-+]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)

FIELDS = [
    "log_file", "dataset", "algorithm", "beta_topo", "gamma_spec", "eta_lip",
    "warmup_rounds", "warmup_start_beta", "warmup_mode", "fedtoa_prompt_only", "freeze_backbone",
    "topk_edges", "final_rsum", "best_rsum", "avg_task_loss", "avg_topo_loss_used",
    "avg_scaled_topo_term", "avg_spec_loss", "avg_active_edge_count", "retained_edge_density_summary",
    "estimated_per_round_communication", "estimated_total_communication", "status_notes",
]


def parse_log(path: Path) -> dict:
    cfg = {}
    task_losses, topo_losses, scaled_terms, spec_losses, edge_counts = [], [], [], [], []
    retained_densities, round_comms = [], []
    total_comm = 0
    rsums = []

    text = path.read_text(errors="ignore").splitlines()
    for line in text:
        m = RUN_RE.search(line)
        if m:
            cfg[m.group("k")] = m.group("v").strip()
        m = MET_RE.search(line)
        if m:
            task_losses.append(float(m.group("task")))
            topo_losses.append(float(m.group("topo")))
            scaled_terms.append(float(m.group("scaled")))
            spec_losses.append(float(m.group("spec")))
            edge_counts.append(float(m.group("edges")))
        m = BLUE_RE.search(line)
        if m:
            retained_densities.append(float(m.group("density")))
        m = COMM_RE.search(line)
        if m:
            round_comms.append(int(m.group("round")))
            total_comm = int(m.group("cum"))
        if "rsum" in line.lower():
            for val in RSUM_RE.findall(line):
                rsums.append(float(val))

    status = "stable"
    if not text:
        status = "incomplete:empty_log"
    elif cfg.get("algorithm", "").lower() == "fedtoa" and not task_losses:
        status = "suspicious:no_train_metrics"
    elif cfg.get("algorithm", "").lower() == "fedtoa" and not round_comms:
        status = "suspicious:no_comm_metrics"

    return {
        "log_file": str(path),
        "dataset": cfg.get("dataset", "unknown"),
        "algorithm": cfg.get("algorithm", "unknown"),
        "beta_topo": cfg.get("beta_topo", "n/a"),
        "gamma_spec": cfg.get("gamma_spec", "n/a"),
        "eta_lip": cfg.get("eta_lip", "n/a"),
        "warmup_rounds": cfg.get("warmup_rounds", "n/a"),
        "warmup_start_beta": cfg.get("warmup_start_beta", "n/a"),
        "warmup_mode": cfg.get("warmup_mode", "n/a"),
        "fedtoa_prompt_only": cfg.get("fedtoa_prompt_only", "n/a"),
        "freeze_backbone": cfg.get("freeze_backbone", "n/a"),
        "topk_edges": cfg.get("topk_edges", "n/a"),
        "final_rsum": rsums[-1] if rsums else "n/a",
        "best_rsum": max(rsums) if rsums else "n/a",
        "avg_task_loss": mean(task_losses) if task_losses else "n/a",
        "avg_topo_loss_used": mean(topo_losses) if topo_losses else "n/a",
        "avg_scaled_topo_term": mean(scaled_terms) if scaled_terms else "n/a",
        "avg_spec_loss": mean(spec_losses) if spec_losses else "n/a",
        "avg_active_edge_count": mean(edge_counts) if edge_counts else "n/a",
        "retained_edge_density_summary": mean(retained_densities) if retained_densities else "n/a",
        "estimated_per_round_communication": int(mean(round_comms)) if round_comms else "n/a",
        "estimated_total_communication": total_comm if total_comm else "n/a",
        "status_notes": status,
    }


def write_markdown(rows: list[dict], out_md: Path) -> None:
    headers = FIELDS
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs", help="Directory containing *.log files.")
    ap.add_argument("--glob", default="*.log", help="Glob pattern inside --log-dir.")
    ap.add_argument("--out-csv", default="outputs/fedtoa_compare/summary.csv")
    ap.add_argument("--out-md", default="outputs/fedtoa_compare/summary.md")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    logs = sorted(log_dir.rglob(args.glob))
    rows = [parse_log(path) for path in logs]

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, out_md)
    print(f"[SUMMARY] logs={len(rows)} csv={out_csv} md={out_md}")


if __name__ == "__main__":
    main()
