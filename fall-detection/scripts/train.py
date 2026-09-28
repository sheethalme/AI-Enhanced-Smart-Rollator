"""RandomForest baseline, subject-independent evaluation only.

(a) within-pool: GroupKFold over subjects (each fold leaves ~1/5 of subjects out)
(b) cross-dataset: train on one dataset, test on the ENTIRE other one

Each evaluation is run for two feature sets:
  all        -- every feature from features.py
  invariant  -- orientation-invariant only (amag / gmag features + SMA). Axis-wise
                features depend on how the device sits (waist board vs pocket phone)
                and are not comparable across datasets.
Usage: python train.py [--clip2g]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from common import PROCESSED
from features import META

N_FOLDS = 5
SEED = 0


def model():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)


def feature_sets(df):
    allf = [c for c in df.columns if c not in META]
    inv = [c for c in allf if c.startswith(("amag_", "gmag_", "acc_sma", "gyro_sma"))]
    return {"all": allf, "invariant": inv}


def report(name, y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    print(f"\n  {name}")
    print(f"    sensitivity {sens:6.1%}   specificity {spec:6.1%}   false-alarm rate {1 - spec:6.2%}")
    print(f"                    pred 0   pred 1\n    true 0 (ADL) {tn:9,d} {fp:8,d}\n    true 1 (fall){fn:9,d} {tp:8,d}")
    return {"eval": name, "sensitivity": sens, "specificity": spec, "false_alarm": 1 - spec,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def within_pool(df, cols):
    pred = np.zeros(len(df), int)
    for tr, te in GroupKFold(N_FOLDS).split(df, groups=df.subject.astype(str)):
        m = model().fit(df.iloc[tr][cols], df.label.iloc[tr])
        pred[te] = m.predict(df.iloc[te][cols])
    out = [report(f"within-pool, {N_FOLDS}-fold leave-subjects-out (all windows)", df.label, pred)]
    for ds in ["sisfall", "mobifall"]:
        m = (df.dataset == ds).to_numpy()
        out.append(report(f"  ...of which {ds} windows", df.label[m], pred[m]))
    return out


def cross(df, cols, train_ds, test_ds):
    tr, te = df[df.dataset == train_ds], df[df.dataset == test_ds]
    m = model().fit(tr[cols], tr.label)
    return report(f"cross-dataset: train {train_ds} -> test ALL {test_ds}", te.label, m.predict(te[cols]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip2g", action="store_true")
    args = ap.parse_args()
    tag = "clip2g" if args.clip2g else "raw"
    df = pd.read_parquet(PROCESSED / f"features{'_clip2g' if args.clip2g else ''}.parquet")
    df["dataset"] = df.dataset.astype(str)
    print(f"Features: {tag}  |  {len(df):,} windows, {df.subject.nunique()} subjects, "
          f"fall rate {df.label.mean():.1%}")

    rows = []
    for fs_name, cols in feature_sets(df).items():
        print("\n" + "=" * 70 + f"\nFEATURE SET: {fs_name} ({len(cols)} features)\n" + "=" * 70)
        res = within_pool(df, cols)
        res += [cross(df, cols, "sisfall", "mobifall"), cross(df, cols, "mobifall", "sisfall")]
        rows += [{"features": tag, "set": fs_name, **r} for r in res]

    out = PROCESSED / f"results_{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out.name}")


if __name__ == "__main__":
    main()
