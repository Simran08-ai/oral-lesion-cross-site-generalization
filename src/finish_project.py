"""
finish_project.py — completes the remaining analysis for the DSC 445 oral lesion project.

Place in {ROOT}/src/ alongside pipeline.py, then in Colab:

    import sys; sys.path.insert(0, '/content/drive/MyDrive/oral_cancer_crosssite/src')
    import finish_project as F
    F.step0_load()                 # manifest + sanity
    F.step1_rq3_leakage_check(adapt_df, eval_df)   # run FIRST — can invalidate RQ3
    F.step2_rerun_rq1()            # slow: retrains CNN
    F.step3_rerun_rq2()            # slow: retrains CNN, 12 pairs
    F.step4_permutation()
    F.step5_budget_direction_check()
    F.step6_figure()
    F.step7_audit()

Each step prints what it found and saves to {ROOT}/outputs/.
"""

import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix, accuracy_score)

# ----------------------------------------------------------------- config
ROOT    = '/content/drive/MyDrive/oral_cancer_crosssite'
TAB     = f'{ROOT}/outputs/tables'
FIG     = f'{ROOT}/outputs/figures'
# Primary design, matching the submitted proposal: four sites, twelve ordered pairs.
# LKA_C is excluded because it contains no high-risk cases (AUC undefined).
SITES     = ['LKA_R', 'LKA_N', 'LKA_S', 'MYS_sample']
# Robustness design: adds LKA_K (103 images, 31 patients, both classes present).
# Five sites -> 20 ordered pairs, 12 within-country and 8 cross-country.
SITES_EXT = ['LKA_R', 'LKA_N', 'LKA_S', 'LKA_K', 'MYS_sample']
EXCLUDED  = ['LKA_C']              # single class
N_TRAIN   = 640
N_BOOT  = 2000

BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#898781'

_df = None


# ------------------------------------------------------------------ step 0
def step0_load():
    """Load the manifest and confirm sites, patients and class balance."""
    global _df
    _df = pd.read_csv(f'{TAB}/manifest_all.csv')
    need = {'site', 'patient_id', 'risk_label', 'country'}
    missing = need - set(_df.columns)
    assert not missing, f'manifest is missing columns: {missing}'

    prof = (_df.groupby('site')
                .agg(images=('risk_label', 'size'),
                     patients=('patient_id', 'nunique'),
                     high_risk=('risk_label', 'sum'))
                .assign(low_risk=lambda d: d.images - d.high_risk))
    print(prof.to_string())

    both = prof[(prof.high_risk > 0) & (prof.low_risk > 0)].index.tolist()
    print('\nsites with both classes:', both)
    print('primary SITES          :', SITES)
    print('robustness SITES_EXT   :', SITES_EXT)
    print('excluded (single class):', EXCLUDED)
    bad = [x for x in SITES_EXT if x not in both]
    if bad:
        print(f'WARNING: {bad} lack both risk classes — remove them')

    spanning = (_df.groupby('patient_id').site.nunique() > 1).sum()
    print(f'\npatients appearing in more than one site: {spanning}')
    assert spanning == 0, 'a patient spans sites — site-based splitting would leak'
    print('OK: no patient spans sites')
    return _df


def _df_or_die():
    assert _df is not None, 'run step0_load() first'
    return _df


# ------------------------------------------------------------------ step 1
def step1_rq3_leakage_check(adapt_df, eval_df):
    """RQ3 promises adaptation and evaluation patients are disjoint. Verify it."""
    a = set(adapt_df.patient_id.unique())
    e = set(eval_df.patient_id.unique())
    ov = a & e
    print(f'adaptation patients : {len(a)}')
    print(f'evaluation patients : {len(e)}')
    print(f'overlap             : {len(ov)}')
    if ov:
        print('LEAK — shared patients:', sorted(ov)[:15])
        raise AssertionError('RQ3 adaptation and evaluation share patients; '
                             'the fine-tuning numbers are inflated')
    print('OK: RQ3 patient separation holds')


# --------------------------------------------------- patient-level scoring
def score(y, p, groups=None, n_boot=N_BOOT, seed=0, thr=0.5):
    """All metrics the proposal promises. groups=patient_id -> patient-level bootstrap."""
    y, p = np.asarray(y), np.asarray(p, float)
    yhat = (p >= thr).astype(int)

    pr,  rc,  f1,  _ = precision_recall_fscore_support(y, yhat, average='binary',
                                                       zero_division=0)
    mpr, mrc, mf1, _ = precision_recall_fscore_support(y, yhat, average='macro',
                                                       zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()

    rng, boots = np.random.default_rng(seed), []
    if groups is None:
        allidx = np.arange(len(y))
        draw = lambda: rng.choice(allidx, len(allidx), replace=True)
    else:
        groups = np.asarray(groups)
        pats = np.unique(groups)
        by   = {q: np.where(groups == q)[0] for q in pats}
        draw = lambda: np.concatenate([by[q] for q in
                                       rng.choice(pats, len(pats), replace=True)])
    for _ in range(n_boot):
        idx = draw()
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)

    return dict(auc=roc_auc_score(y, p), lo=lo, hi=hi,
                accuracy=accuracy_score(y, yhat),
                precision=pr, recall=rc, f1=f1,
                macro_precision=mpr, macro_recall=mrc, macro_f1=mf1,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                n=len(y),
                n_patients=None if groups is None else len(np.unique(groups)),
                n_boot_valid=len(boots))


def within_site_cv(df, site, fit_predict, n_splits=5, seed=0):
    g = df[df.site == site].reset_index(drop=True)
    y, groups = g.risk_label.values, g.patient_id.values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in sgkf.split(g, y, groups):
        assert not (set(groups[tr]) & set(groups[te])), 'patient leaked across folds'
        folds.append(score(y[te], fit_predict(g.iloc[tr], g.iloc[te]),
                           groups=groups[te]))
    F = pd.DataFrame(folds)
    out = F.mean(numeric_only=True).to_dict()
    out['n_folds'] = len(F)
    return out, F


def leave_one_site_out(df, train_sites, test_site, fit_predict):
    tr = df[df.site.isin(train_sites)].reset_index(drop=True)
    te = df[df.site == test_site].reset_index(drop=True)
    assert not (set(tr.patient_id) & set(te.patient_id)), 'patient in both sites'
    return score(te.risk_label.values, fit_predict(tr, te),
                 groups=te.patient_id.values)


def subsample_patients(df, n_img, seed=0):
    """Trim to n_img images without splitting any patient."""
    if len(df) <= n_img:
        return df
    rng = np.random.default_rng(seed)
    keep, tot = [], 0
    for p in rng.permutation(df.patient_id.unique()):
        k = int((df.patient_id == p).sum())
        if tot + k > n_img:
            break
        keep.append(p); tot += k
    return df[df.patient_id.isin(keep)].reset_index(drop=True)


def _country(site):
    d = _df_or_die()
    vals = d.loc[d.site == site, 'country'].unique()
    assert len(vals) == 1, f'site {site} maps to several countries: {vals}'
    return vals[0]


def _cnn():
    from pipeline import make_cnn_predictor
    return make_cnn_predictor(epochs=8, bs=32, lr=3e-4, wd=1e-4)


# ------------------------------------------------------------------ step 2
def step2_rerun_rq1():
    """RQ1 with patient-level intervals and the full metric set."""
    df, cnn, rows, folds = _df_or_die(), _cnn(), [], []
    for s in SITES:
        if (df.site == s).sum() < 40:
            print(f'skip {s}: fewer than 40 images'); continue
        m, F = within_site_cv(df, s, cnn, n_splits=5, seed=0)
        rows.append(dict(site=s, **m)); F['site'] = s; folds.append(F)
    rq1 = pd.DataFrame(rows)
    rq1.to_csv(f'{TAB}/rq1_within_site_patientboot.csv', index=False)
    pd.concat(folds).to_csv(f'{TAB}/rq1_folds_patientboot.csv', index=False)
    print(rq1[['site', 'auc', 'lo', 'hi', 'accuracy', 'macro_f1', 'n_patients']]
          .round(3).to_string(index=False))
    return rq1


# ------------------------------------------------------------------ step 3
def step3_rerun_rq2(sites=None, out=None):
    """All ordered site pairs, fixed budget, patient-level intervals.

    sites=None  -> SITES (four sites, twelve pairs, matches the proposal)
    sites=SITES_EXT -> five sites, twenty pairs, used as a robustness check
    """
    sites = sites or SITES
    out = out or (f'{TAB}/rq2_matrix_patientboot.csv' if sites == SITES
                  else f'{TAB}/rq2_matrix_patientboot_5site.csv')
    df, cnn, rows = _df_or_die(), _cnn(), []
    for a in sites:
        for b in sites:
            if a == b:
                continue
            tr  = subsample_patients(df[df.site == a].reset_index(drop=True),
                                     N_TRAIN, seed=0)
            sub = pd.concat([tr, df[df.site == b]]).reset_index(drop=True)
            r = leave_one_site_out(sub, [a], b, cnn)
            rows.append(dict(train=a, test=b,
                             cross=_country(a) != _country(b),
                             n_train_img=len(tr), **r))
    rq2 = pd.DataFrame(rows)
    rq2.to_csv(out, index=False)
    print(rq2[['train', 'test', 'cross', 'n_train_img', 'auc', 'lo', 'hi']]
          .round(3).to_string(index=False))
    n, nc = len(rq2), int(rq2.cross.sum())
    print(f'\npairs {n} | cross {nc} | within {n - nc}')
    print(f'permutation space: C({n},{nc}) assignments')
    print('saved', out)
    return rq2


# ------------------------------------------------------------------ step 4
def perm_pair_test(scores, is_cross):
    s, c = np.asarray(scores, float), np.asarray(is_cross, bool)
    n, k = len(s), int(c.sum())
    obs = s[~c].mean() - s[c].mean()
    stats = []
    for comb in combinations(range(n), k):
        m = np.zeros(n, bool); m[list(comb)] = True
        stats.append(s[~m].mean() - s[m].mean())
    stats = np.asarray(stats)
    return obs, float((stats >= obs).mean()), len(stats)


def step4_permutation(path=None):
    rq2 = pd.read_csv(path or f'{TAB}/rq2_matrix_patientboot.csv')
    obs, p, nperm = perm_pair_test(rq2.auc, rq2.cross)
    print(f'within mean {rq2.loc[~rq2.cross, "auc"].mean():.3f}')
    print(f'cross  mean {rq2.loc[ rq2.cross, "auc"].mean():.3f}')
    print(f'gap {obs:.3f} | p {p:.4f} | {nperm} permutations | floor {1/nperm:.5f}')
    if abs(p - 1 / nperm) < 1e-9:
        print('NOTE: p is at the floor of this test — report it as the smallest '
              'value the test can return, not as a precise p-value.')
    spans = ((rq2.cross) & (rq2.lo <= 0.5) & (rq2.hi >= 0.5)).sum()
    print(f'cross-country intervals including 0.5: {spans} of {int(rq2.cross.sum())}')
    return obs, p, nperm


# ------------------------------------------------------------------ step 5
def step5_budget_direction_check(path=None):  # noqa: C901
    """Malaysia cannot meet the 640-image budget, so MYS-source pairs train on less.
    Split the cross-country pairs by direction to see whether that drives the result."""
    rq2 = pd.read_csv(path or f'{TAB}/rq2_matrix_patientboot.csv')
    rq2['direction'] = np.where(~rq2.cross, 'within-country',
                        np.where(rq2.n_train_img < N_TRAIN, 'MYS-source (small train)',
                                                    'MYS-target (full budget)'))
    summ = (rq2.groupby('direction')
                .agg(n=('auc', 'size'), mean_auc=('auc', 'mean'),
                     min_auc=('auc', 'min'), max_auc=('auc', 'max'))
                .round(3))
    print(summ.to_string())
    print('\ncross-country pairs in detail:')
    print(rq2[rq2.cross][['train', 'test', 'n_train_img', 'auc', 'lo', 'hi', 'direction']]
          .round(3).to_string(index=False))
    print('\nREAD: if the full-budget direction also sits near 0.5 with intervals\n'
          'spanning chance, training-set size does not explain the cross-country\n'
          'collapse. If only the small-train direction fails, it partly does —\n'
          'and the results write-up needs changing.')
    rq2.to_csv(f'{TAB}/rq2_matrix_patientboot.csv', index=False)
    return summ


# ------------------------------------------------------------------ step 6
def step6_figure(path=None):
    import matplotlib.pyplot as plt
    rq2 = pd.read_csv(path or f'{TAB}/rq2_matrix_patientboot.csv')
    d = rq2.sort_values(['cross', 'auc'], ascending=[True, False]).reset_index(drop=True)
    y = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    for i, r in d.iterrows():
        col = ORANGE if r.cross else BLUE
        ax.plot([r.lo, r.hi], [y[i], y[i]], color=col, lw=2.2, solid_capstyle='round')
        ax.plot(r.auc, y[i], 'o', color=col, ms=6)
        ax.text(r.hi + 0.012, y[i], f'{r.auc:.3f}', va='center', fontsize=8, color=col)

    ax.axvline(0.5, color=GREY, ls='--', lw=1)
    ax.set_yticks(y); ax.set_yticklabels(d.train + '→' + d.test, fontsize=9)
    ax.set_xlabel('Transfer AUC (95% patient-level bootstrap CI)')
    ax.set_xlim(0.30, 1.0)
    ax.text(0.315, y[0] + 0.6, 'within country', color=BLUE, fontsize=9, weight='bold')
    n_within = int((~d.cross).sum())
    if n_within < len(d):
        ax.text(0.315, y[n_within] + 0.6, 'across countries',
                color=ORANGE, fontsize=9, weight='bold')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = f'{FIG}/rq2_transfer_patientboot.png'
    plt.savefig(out, bbox_inches='tight'); plt.show()
    print('saved', out)


# ------------------------------------------------------------------ step 7
def step7_audit():
    """Confirm every metric the proposal promises is present in the saved tables."""
    promised = ['auc', 'lo', 'hi', 'accuracy', 'precision', 'recall', 'f1',
                'macro_f1', 'tn', 'fp', 'fn', 'tp']
    files = ['rq1_within_site_patientboot.csv', 'rq2_matrix_patientboot.csv',
             'rq3_fewshot_curve.csv', 'rq4_risk_coverage.csv']
    ok = True
    for fn in files:
        p = f'{TAB}/{fn}'
        if not os.path.exists(p):
            print(f'{fn:38s} MISSING'); ok = False; continue
        cols = pd.read_csv(p).columns.tolist()
        gaps = [c for c in promised if c not in cols]
        print(f'{fn:38s} {"complete" if not gaps else "missing " + str(gaps)}')
        ok &= not gaps
    print('\nall promised metrics present' if ok else
          '\nsome metrics still missing — re-run the step that writes that table')
    return ok


def run_all(adapt_df=None, eval_df=None):
    step0_load()
    if adapt_df is not None and eval_df is not None:
        step1_rq3_leakage_check(adapt_df, eval_df)
    step2_rerun_rq1()
    step3_rerun_rq2()
    step4_permutation()
    step5_budget_direction_check()
    step6_figure()
    step7_audit()
