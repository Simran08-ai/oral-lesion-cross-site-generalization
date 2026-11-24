# =====================================================================
# DSC 445 — precision / recall / F1 with patient-level bootstrap CIs
#
# WHAT THIS DOES
#   Takes the predictions your pipeline already produced and reports,
#   for every experiment, the threshold metrics your professor asked
#   for alongside AUC: precision, recall (sensitivity), specificity,
#   F1, macro-F1, and the confusion matrix.
#
#   Confidence intervals resample PATIENTS, not images — the same rule
#   used everywhere else in this project, because several photographs
#   of one patient are not independent observations.
#
# WHAT YOU NEED
#   A CSV with one row per predicted image and these columns:
#       experiment   e.g. "LKA_R within-site" or "LKA_R->MYS_sample"
#       patient_id   the patient the photograph belongs to
#       y_true       1 = high-risk, 0 = low-risk
#       y_score      the model's output score, 0..1
#
#   If you do not have that CSV yet, scroll to STEP 0 at the bottom:
#   it shows the three lines to add to your existing evaluation loop
#   to write it out.
#
# HOW TO RUN
#   python metrics_precision_recall.py predictions.csv
#   (or in Colab:  !python metrics_precision_recall.py predictions.csv)
# =====================================================================

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

THRESHOLD = 0.50     # operating point; matches the confusion matrices in the report
N_BOOT    = 1000     # bootstrap resamples
SEED      = 0


# ---------------------------------------------------------------------
# point metrics at one threshold
# ---------------------------------------------------------------------
def metrics_at(y_true, y_score, thr=THRESHOLD):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= thr).astype(int)

    # labels=[0,1] forces a 2x2 matrix even if one class is absent
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "n":            int(len(y_true)),
        "n_high_risk":  int(tp + fn),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "recall":       tp / (tp + fn) if (tp + fn) else np.nan,   # sensitivity
        "precision":    tp / (tp + fp) if (tp + fp) else np.nan,   # PPV
        "specificity":  tn / (tn + fp) if (tn + fp) else np.nan,
        "npv":          tn / (tn + fn) if (tn + fn) else np.nan,
        "accuracy":     (tp + tn) / len(y_true),
    }
    p, r = out["precision"], out["recall"]
    out["f1"] = (2 * p * r / (p + r)) if (p and r and not np.isnan(p) and not np.isnan(r)) else np.nan
    out["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUC needs both classes present
    out["auc"] = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    return out


# ---------------------------------------------------------------------
# patient-level bootstrap: resample PATIENTS with replacement,
# take all of a sampled patient's images together
# ---------------------------------------------------------------------
def patient_bootstrap(df, stat_fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    patients = df["patient_id"].unique()
    by_patient = {p: g for p, g in df.groupby("patient_id")}

    draws = []
    for _ in range(n_boot):
        picked = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([by_patient[p] for p in picked], ignore_index=True)
        try:
            draws.append(stat_fn(boot))
        except Exception:
            draws.append(np.nan)          # degenerate resample: skip it

    draws = np.array(draws, dtype=float)
    draws = draws[~np.isnan(draws)]
    if len(draws) < 50:
        return (np.nan, np.nan)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def ci_for(df, key, thr=THRESHOLD):
    return patient_bootstrap(df, lambda d: metrics_at(d["y_true"], d["y_score"], thr)[key])


# ---------------------------------------------------------------------
# threshold sweep — where recall and precision trade off
# ---------------------------------------------------------------------
def sweep(df, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    rows = []
    for t in thresholds:
        m = metrics_at(df["y_true"], df["y_score"], t)
        rows.append({"threshold": t, "recall": m["recall"], "precision": m["precision"],
                     "specificity": m["specificity"], "f1": m["f1"], "accuracy": m["accuracy"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
def main(path):
    df = pd.read_csv(path)

    required = {"experiment", "patient_id", "y_true", "y_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing column(s): {sorted(missing)}\n"
                         f"Found: {list(df.columns)}")

    df["y_true"] = df["y_true"].astype(int)
    df["y_score"] = df["y_score"].astype(float)

    summary = []
    for name, g in df.groupby("experiment", sort=False):
        m = metrics_at(g["y_true"], g["y_score"])

        rec_lo, rec_hi = ci_for(g, "recall")
        pre_lo, pre_hi = ci_for(g, "precision")
        auc_lo, auc_hi = ci_for(g, "auc")

        summary.append({
            "experiment":   name,
            "n_images":     m["n"],
            "n_patients":   g["patient_id"].nunique(),
            "n_high_risk":  m["n_high_risk"],
            "AUC":          round(m["auc"], 3),
            "AUC_95CI":     f"{auc_lo:.3f}–{auc_hi:.3f}",
            "recall":       round(m["recall"], 3),
            "recall_95CI":  f"{rec_lo:.3f}–{rec_hi:.3f}",
            "precision":    round(m["precision"], 3),
            "prec_95CI":    f"{pre_lo:.3f}–{pre_hi:.3f}",
            "specificity":  round(m["specificity"], 3),
            "F1":           round(m["f1"], 3),
            "F1_macro":     round(m["f1_macro"], 3),
            "accuracy":     round(m["accuracy"], 3),
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
        })

        print(f"\n=== {name}")
        print(f"    {m['n']} images from {g['patient_id'].nunique()} patients, "
              f"{m['n_high_risk']} high-risk")
        print(f"    confusion @ {THRESHOLD}:  TP={m['TP']}  FP={m['FP']}  "
              f"FN={m['FN']}  TN={m['TN']}")
        print(f"    recall    {m['recall']:.3f}  [{rec_lo:.3f}, {rec_hi:.3f}]")
        print(f"    precision {m['precision']:.3f}  [{pre_lo:.3f}, {pre_hi:.3f}]")
        print(f"    F1 {m['f1']:.3f}   macro-F1 {m['f1_macro']:.3f}   "
              f"specificity {m['specificity']:.3f}   accuracy {m['accuracy']:.3f}")
        print(f"    AUC       {m['auc']:.3f}  [{auc_lo:.3f}, {auc_hi:.3f}]")
        print(sweep(g).round(3).to_string(index=False))

    out = pd.DataFrame(summary)
    out.to_csv("precision_recall_results.csv", index=False)
    print("\n\n================ SUMMARY (also saved to precision_recall_results.csv)")
    print(out[["experiment", "AUC", "recall", "recall_95CI",
               "precision", "prec_95CI", "F1_macro", "TP", "FP", "FN", "TN"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "predictions.csv")


# =====================================================================
# STEP 0 — if you don't have predictions.csv yet
#
# Inside the evaluation loop you already have, wherever you currently
# compute roc_auc_score(y_true, y_score), add the predictions to a list
# instead of throwing them away:
#
#     rows = []                      # once, before the loop
#
#     # ... inside the loop, after you get y_score for a test set ...
#     rows.append(pd.DataFrame({
#         "experiment": f"{train_site}->{test_site}",   # or f"{site} within-site"
#         "patient_id": test_patient_ids,               # one per image
#         "y_true":     y_true,
#         "y_score":    y_score,
#     }))
#
#     pd.concat(rows).to_csv("predictions.csv", index=False)   # once, after
#
# For the within-site 5-fold runs, append each fold's OUT-OF-FOLD
# predictions with the same experiment name — pooling them is what the
# report describes, and it is what gives one AUC per site rather than
# five.
# =====================================================================
