"""
finish_rq34.py — re-runs RQ3 and RQ4 on current footing, with patient-level
bootstrap intervals, matching the design described in the proposal.

Place beside finish_project.py in {ROOT}/src/, then in Colab:

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'finish_rq34', f'{ROOT}/src/finish_rq34.py')
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    R.bind(F, cnn)                 # F = finish_project module, cnn = F._cnn()

    rq3 = R.run_rq3()              # slow: trains 2 seeds x 5 k-values per target
    rq4 = R.run_rq4()              # slow: trains 1 model per site pair, then sweeps
"""

import numpy as np
import pandas as pd

_F = None
_CNN = None


def bind(finish_project_module, cnn_predictor):
    """Hand over the loaded finish_project module and a fit_predict callable."""
    global _F, _CNN
    _F, _CNN = finish_project_module, cnn_predictor
    print('bound. SITES =', _F.SITES)


def _need():
    assert _F is not None, 'call bind(F, cnn) first'
    return _F, _CNN


# --------------------------------------------------------------- RQ3
def _split_target_patients(tgt, seed):
    """Halve a site at the patient level: adaptation pool vs evaluation set."""
    pats = tgt.patient_id.unique()
    perm = np.random.default_rng(seed).permutation(pats)
    half = len(perm) // 2
    a, t = set(perm[:half]), set(perm[half:])
    assert not (a & t), 'adaptation and evaluation share patients'
    return (tgt[tgt.patient_id.isin(a)].reset_index(drop=True),
            tgt[tgt.patient_id.isin(t)].reset_index(drop=True))


def run_rq3(source='LKA_R', targets=None, ks=(0, 10, 25, 50, 100),
            seeds=(0, 1), budget=640, out=None):
    """Augment a source-trained model with k target-site images and re-evaluate.

    Adaptation images and evaluation images come from disjoint target patients,
    enforced by assertion. k is a budget in images, filled whole patients at a
    time, so k_actual <= k.
    """
    F, cnn = _need()
    df = F._df_or_die()
    targets = targets or [s for s in F.SITES if s != source]
    src_full = df[df.site == source].reset_index(drop=True)
    src = F.subsample_patients(src_full, budget, seed=0)

    rows = []
    for target in targets:
        tgt = df[df.site == target].reset_index(drop=True)
        for seed in seeds:
            adapt, test = _split_target_patients(tgt, seed)
            assert not (set(src.patient_id) & set(test.patient_id))
            for k in ks:
                extra = F.subsample_patients(adapt, k, seed=seed) if k else adapt.iloc[:0]
                assert not (set(extra.patient_id) & set(test.patient_id)), \
                    'adaptation images leaked into the evaluation set'
                tr = pd.concat([src, extra]).reset_index(drop=True)
                r = F.score(test.risk_label.values, cnn(tr, test),
                            groups=test.patient_id.values)
                rows.append(dict(source=source, target=target, seed=seed, k=k,
                                 k_actual=len(extra),
                                 k_patients=extra.patient_id.nunique(),
                                 n_train=len(tr), n_test=len(test),
                                 n_test_patients=test.patient_id.nunique(), **r))
                print(f'{source}->{target} seed{seed} k={k:>3} '
                      f'(actual {len(extra):>3}) auc={r["auc"]:.3f} '
                      f'[{r["lo"]:.3f}, {r["hi"]:.3f}]')
    rq3 = pd.DataFrame(rows)
    out = out or f'{F.TAB}/rq3_fewshot_patientboot.csv'
    rq3.to_csv(out, index=False)

    curve = (rq3.groupby(['target', 'k'])
                .agg(mean_auc=('auc', 'mean'), sd_auc=('auc', 'std'),
                     n_seeds=('auc', 'size')).round(3).reset_index())
    curve.to_csv(f'{F.TAB}/rq3_fewshot_curve_patientboot.csv', index=False)
    print('\n' + curve.to_string(index=False))
    print('\nsaved', out)
    return rq3, curve


# --------------------------------------------------------------- RQ4
def _predict_pair(df, train_site, test_site, budget=640):
    """Train on one site, predict on another. Returns the test frame + scores."""
    F, cnn = _need()
    tr = F.subsample_patients(df[df.site == train_site].reset_index(drop=True),
                              budget, seed=0)
    te = df[df.site == test_site].reset_index(drop=True)
    assert not (set(tr.patient_id) & set(te.patient_id))
    return te, np.asarray(cnn(tr, te), float)


def risk_coverage(y, p, groups, grid=np.round(np.arange(0.25, 1.001, 0.05), 2)):
    """Accuracy and high-risk recall among retained cases, by coverage.

    Confidence is distance from the decision boundary; the least confident
    predictions are referred first.
    """
    y, p, groups = np.asarray(y), np.asarray(p, float), np.asarray(groups)
    order = np.argsort(-np.abs(p - 0.5))          # most confident first
    rows = []
    for cov in grid:
        k = max(1, int(round(cov * len(y))))
        idx = order[:k]
        yh = (p[idx] >= 0.5).astype(int)
        pos = y[idx] == 1
        rows.append(dict(
            coverage=float(cov), referral_rate=round(1 - cov, 2), n_retained=k,
            accuracy=float((yh == y[idx]).mean()),
            high_risk_recall=float(yh[pos].mean()) if pos.any() else np.nan,
            n_high_risk_retained=int(pos.sum()),
            missed_high_risk=int((pos & (yh == 0)).sum()),
            n_patients_retained=int(len(np.unique(groups[idx])))))
    return pd.DataFrame(rows)


def run_rq4(source='LKA_R', within='LKA_N', across='MYS_sample', out=None):
    """Risk-coverage for one within-country and one cross-country pair."""
    F, _ = _need()
    df = F._df_or_die()
    frames = []
    for label, test_site in [('within-country', within), ('cross-country', across)]:
        te, p = _predict_pair(df, source, test_site)
        rc = risk_coverage(te.risk_label.values, p, te.patient_id.values)
        rc.insert(0, 'comparison', label)
        rc.insert(1, 'train', source); rc.insert(2, 'test', test_site)
        frames.append(rc)
        pd.DataFrame({'site': test_site, 'patient_id': te.patient_id,
                      'y': te.risk_label.values, 'p': p}).to_csv(
            f'{F.TAB}/predictions_{source}_to_{test_site}.csv', index=False)
    rq4 = pd.concat(frames, ignore_index=True)
    out = out or f'{F.TAB}/rq4_risk_coverage_patientboot.csv'
    rq4.to_csv(out, index=False)

    show = ['comparison', 'test', 'coverage', 'referral_rate', 'accuracy',
            'high_risk_recall', 'missed_high_risk', 'n_high_risk_retained']
    print(rq4[show].round(3).to_string(index=False))
    print('\nsaved', out)
    return rq4
