from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, required=True)
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for trace_path in sorted(trace_dir.glob("*_trace.csv")):
        df = pd.read_csv(trace_path)
        if df.empty:
            continue

        turn_step = int(df["turn_step"].iloc[0])
        turn_time = float(turn_step) * 0.2
        success = int(df["success"].iloc[-1])
        collision = int(df["collision"].iloc[-1])
        reaction_time = df["deviation_lateral"].where(df["deviation_lateral"] > 0.3).dropna()
        reaction_time_str = "nan" if reaction_time.empty else f"{float(df[df['deviation_lateral'] > 0.3]['time'].iloc[0] - turn_time):.2f}"
        risk_rise = df["risk_turn"].where(df["risk_turn"] >= 0.5).dropna()
        risk_rise_str = "nan" if risk_rise.empty else f"{float(df[df['risk_turn'] >= 0.5]['time'].iloc[0] - turn_time):.2f}"

        fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
        x = df["time"]

        axes[0].plot(x, df["risk_turn"], label="risk_turn")
        axes[0].axvline(turn_time, color="red", linestyle="--")
        axes[0].set_ylabel("risk_turn")

        axes[1].plot(x, df["sigma_turn_trace"], label="sigma_turn_trace", color="orange")
        axes[1].axvline(turn_time, color="red", linestyle="--")
        axes[1].set_ylabel("sigma_trace")

        axes[2].plot(x, df["R_sum"], label="R_sum")
        axes[2].plot(x, df["R_bar"], label="R_bar")
        axes[2].axvline(turn_time, color="red", linestyle="--")
        axes[2].legend()
        axes[2].set_ylabel("global risk")

        axes[3].plot(x, df["w_risk_turn"], label="w_risk_turn")
        axes[3].plot(x, df["risk_rank_turn"], label="risk_rank_turn")
        axes[3].axvline(turn_time, color="red", linestyle="--")
        axes[3].legend()
        axes[3].set_ylabel("weight/rank")

        axes[4].plot(x, df["deviation_lateral"], label="deviation_lateral")
        axes[4].plot(x, df["dist_turn"], label="dist_turn")
        axes[4].axvline(turn_time, color="red", linestyle="--")
        axes[4].legend()
        axes[4].set_ylabel("control/dist")
        axes[4].set_xlabel("time (s)")

        title = (
            f"{df['method'].iloc[0]} seed={int(df['seed'].iloc[0])} ep={int(df['episode'].iloc[0])} "
            f"success={success} collision={collision} reaction={reaction_time_str} risk0.5={risk_rise_str}"
        )
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        out_path = out_dir / trace_path.name.replace("_trace.csv", "_diagnostic.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
