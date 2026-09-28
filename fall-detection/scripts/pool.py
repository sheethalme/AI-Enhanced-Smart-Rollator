"""Window both harmonized datasets, pool them, run acceptance checks (fail loudly).

-> data/processed/pooled_windows.parquet
-> data/processed/amag_hist.png
-> data/processed/pool_summary.txt
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DT, PROCESSED, SAMPLE_COLS
from window import window_dataset

DATASETS = ["sisfall", "mobifall"]
COLORS = {"sisfall": "#2a78d6", "mobifall": "#eb6834"}   # categorical slots 1, 2
ALLOWED_PLACEMENT = {"waist", "pocket"}
STILL_STD = 0.02          # g; a window this quiet is "still"
GRAVITY_TOL = 0.1         # still windows must read 1.0 +/- this
FALL_PEAK_RANGE = (2.0, 8.0)

failures, lines = [], []


def say(msg=""):
    print(msg)
    lines.append(msg)


def check(ok, msg):
    say(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


def main():
    windows = []
    say("=" * 70 + "\nPOOL + ACCEPTANCE CHECKS\n" + "=" * 70)
    for name in DATASETS:
        s = pd.read_parquet(PROCESSED / f"{name}.parquet")
        assert list(s.columns) == SAMPLE_COLS, f"{name}: schema mismatch {list(s.columns)}"

        # rate: every trial on an exact 50 Hz grid
        dt = s.groupby("trial", observed=True).t.diff().dropna()
        bad = s.loc[dt.index[(dt - DT).abs() > 1e-3], "trial"].nunique()
        check(bad == 0, f"{name}: every trial uniform 50 Hz ({bad} trials off-grid)")

        w, info = window_dataset(s)
        say(f"  {name}: {len(s):,} samples -> {info}")
        windows.append(w)

    pool = pd.concat(windows, ignore_index=True)
    for c in ["dataset", "subject", "placement", "trial"]:
        pool[c] = pool[c].astype("category")

    amag = np.stack(pool.amag.to_numpy())
    pool_mean, pool_std, pool_peak = amag.mean(1), amag.std(1), amag.max(1)

    say("\nGravity (still windows, amag std < %.2f g):" % STILL_STD)
    for name in DATASETS:
        m = (pool.dataset == name).to_numpy() & (pool_std < STILL_STD) & (pool.label == 0).to_numpy()
        g = np.median(pool_mean[m]) if m.any() else np.nan
        check(abs(g - 1) < GRAVITY_TOL, f"{name}: {m.sum():,} still windows, median amag {g:.3f} g")

    say("\nFall peaks (max amag in label=1 windows):")
    for name in DATASETS:
        m = (pool.dataset == name).to_numpy() & (pool.label == 1).to_numpy()
        pk = pool_peak[m]
        med = np.median(pk)
        check(FALL_PEAK_RANGE[0] <= med <= FALL_PEAK_RANGE[1],
              f"{name}: median {med:.2f} g (p5 {np.percentile(pk, 5):.2f}, "
              f"p95 {np.percentile(pk, 95):.2f}, max {pk.max():.2f})")

    say("\nPlacement:")
    placed = set(pool.placement.astype(str).unique())
    check(placed <= ALLOWED_PLACEMENT, f"placements present: {sorted(placed)}")

    say("\nClass balance (windows):")
    tab = pd.crosstab(pool.dataset, pool.label3, margins=True)
    tab.columns = [{0: "nonfall", 1: "prefall", 2: "fall"}.get(c, c) for c in tab.columns]
    say(tab.to_string())
    bal = pool.groupby("dataset", observed=True).label.agg(["size", "sum"])
    for name, r in bal.iterrows():
        say(f"  {name}: fall {r['sum']:,} / {r['size']:,} = {r['sum'] / r['size']:.1%}  "
            f"(subjects {pool[pool.dataset == name].subject.nunique()})")

    # distribution overlap: whole-window mean amag should sit in the same range
    med = {n: np.median(pool_mean[(pool.dataset == n).to_numpy()]) for n in DATASETS}
    say("\nDistribution check (median of window-mean amag):")
    check(abs(med["sisfall"] - med["mobifall"]) < GRAVITY_TOL,
          "  ".join(f"{n} {v:.3f} g" for n, v in med.items()))
    plot_hist(pool, pool_mean, pool_peak)

    out = PROCESSED / "pooled_windows.parquet"
    pool.to_parquet(out, index=False)
    say(f"\nWrote {out.relative_to(PROCESSED.parent.parent)}: {len(pool):,} windows x 125 samples x 7 channels")
    (PROCESSED / "pool_summary.txt").write_text("\n".join(lines) + "\n")

    if failures:
        print("\nACCEPTANCE FAILED:\n  - " + "\n  - ".join(failures), file=sys.stderr)
        sys.exit(1)
    print("\nAll acceptance checks passed.")


def plot_hist(pool, pool_mean, pool_peak):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
    panels = [(pool_mean, None, np.linspace(0.4, 2.0, 81), "Window-mean amag, all windows (g)"),
              (pool_peak, 1, np.linspace(0, 16, 81), "Window-peak amag, fall windows (g)")]
    for ax, (vals, lab, bins, title) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for name in DATASETS:
            m = (pool.dataset == name).to_numpy()
            if lab is not None:
                m &= (pool.label == lab).to_numpy()
            ax.hist(vals[m], bins=bins, density=True, histtype="step", lw=2,
                    color=COLORS[name], label=f"{name} (n={m.sum():,})")
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_ylabel("density")
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False)
    axes[0].axvline(1.0, color="#52514e", lw=1, ls="--")
    axes[0].annotate("1 g", (1.0, 0.97), xycoords=("data", "axes fraction"),
                     xytext=(4, 0), textcoords="offset points", color="#52514e", va="top")
    fig.tight_layout()
    fig.savefig(PROCESSED / "amag_hist.png", dpi=130, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
