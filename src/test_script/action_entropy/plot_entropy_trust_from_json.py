#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt


def to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str) and v.lower() in ("null", "none", "nan", ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_stepwise_data(json_path: str) -> Dict[str, List]:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"파일이 없습니다: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("experiment_log.json은 list 형식이어야 합니다.")

    steps = []
    H_X = []
    H_X_given_S = []
    H_X_given_LS = []
    trust_T = []

    for row in data:
        step = row.get("step")
        if step is None:
            continue

        ve = row.get("verbalized_entropy", {}) or {}

        steps.append(int(step))
        H_X.append(to_float_or_none(ve.get("H_X")))
        H_X_given_S.append(to_float_or_none(ve.get("H_X_given_S")))
        H_X_given_LS.append(to_float_or_none(ve.get("H_X_given_LS")))
        trust_T.append(to_float_or_none(ve.get("trust_T")))

    return {
        "steps": steps,
        "H_X": H_X,
        "H_X_given_S": H_X_given_S,
        "H_X_given_LS": H_X_given_LS,
        "trust_T": trust_T,
    }


def plot_trust_only(steps, trust_T, output_png=None):
    trust = np.array([
        np.nan if v is None else v for v in trust_T
    ])

    plt.figure(figsize=(12, 5))
    #plt.plot(steps, trust, "o-", linewidth=1.8)
    plt.scatter(steps, trust, s=40)
    plt.xlabel("Step #")
    plt.ylabel("Trust T")
    plt.title("Step-wise Trust")
    
    # 🔴 y축 스케일 강제
    #plt.ylim(-5, 10)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_png:
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        print(f"Trust 그래프 저장: {output_png}")
    else:
        plt.show()



def plot_entropy_only(steps, H_X, H_X_given_S, H_X_given_LS, output_png=None):
    def to_nan(arr):
        return np.array([np.nan if v is None else v for v in arr])

    plt.figure(figsize=(12, 6))
    plt.plot(steps, to_nan(H_X), "o-", label="H(X)")
    plt.plot(steps, to_nan(H_X_given_S), "o-", label="H(X|S)")
    plt.plot(steps, to_nan(H_X_given_LS), "o-", label="H(X|L,S)")

    # plt.scatter(steps, to_nan(H_X), s=40, label="H(X)")
    # plt.scatter(steps, to_nan(H_X_given_S), s=40, label="H(X|S)")
    # plt.scatter(steps, to_nan(H_X_given_LS), s=40, label="H(X|L,S)")

    plt.xlabel("Step #")
    plt.ylabel("Entropy (H)")
    plt.title("Step-wise Entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_png:
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        print(f"Entropy 그래프 저장: {output_png}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("사용법: python plot_entropy_trust_split.py <experiment_log.json 경로>")
        sys.exit(1)

    json_path = sys.argv[1]
    series = load_stepwise_data(json_path)

    steps = series["steps"]

    plot_trust_only(
        steps,
        series["trust_T"],
        output_png="trust_scaled_4.png"
    )

    plot_entropy_only(
        steps,
        series["H_X"],
        series["H_X_given_S"],
        series["H_X_given_LS"],
        output_png="entropy_only_4.png"
    )
    trust = np.array([np.nan if v is None else v for v in series["trust_T"]])
    trust_mean = np.nanmean(trust)
    trust_mean_no_outlier = np.nanmean(trust[(trust >= np.nanmean(trust) - 2*np.nanstd(trust)) & (trust <= np.nanmean(trust) + 2*np.nanstd(trust))])
    print(trust_mean)
    print(trust_mean_no_outlier)




if __name__ == "__main__":
    main()
