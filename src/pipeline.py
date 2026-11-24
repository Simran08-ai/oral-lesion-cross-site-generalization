"""Cross-site oral lesion classification — shared pipeline."""
import os, numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv

HIGH_RISK = {"Cancer", "OPMD"}

# ---------------- metrics ----------------

def score(y_true, y_prob, thresh=0.5):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    out = {
        "n": len(y_true),
        "pos_rate": round(float(y_true.mean()), 3),
        "acc":      round(accuracy_score(y_true, y_pred), 3),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
        "prec_hr":  round(precision_score(y_true, y_pred, zero_division=0), 3),
        "rec_hr":   round(recall_score(y_true, y_pred, zero_division=0), 3),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"]    = round(roc_auc_score(y_true, y_prob), 3)
        out["pr_auc"] = round(average_precision_score(y_true, y_prob), 3)
    else:
        out["auc"] = out["pr_auc"] = None
    out["cm"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out

# ---------------- evaluation ----------------

def within_site_cv(df, site, fit_predict, n_splits=5, seed=0):
    g = df[df.site == site].reset_index(drop=True)
    y, groups = g.risk_label.values, g.patient_id.values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in sgkf.split(g, y, groups):
        assert not (set(groups[tr]) & set(groups[te])), "patient leaked across folds"
        folds.append(score(y[te], fit_predict(g.iloc[tr], g.iloc[te])))
    agg = {}
    for k in ("acc", "f1_macro", "prec_hr", "rec_hr", "auc", "pr_auc"):
        vals = [f[k] for f in folds if f[k] is not None]
        agg[k] = round(float(np.mean(vals)), 3) if vals else None
    agg.update(eval_type="within", site=site, n=len(g),
               n_patients=g.patient_id.nunique())
    return agg, folds


def leave_one_site_out(df, train_sites, test_site, fit_predict):
    tr = df[df.site.isin(train_sites)].reset_index(drop=True)
    te = df[df.site == test_site].reset_index(drop=True)
    assert not (set(tr.patient_id) & set(te.patient_id)), "patient in both sites"
    r = score(te.risk_label.values, fit_predict(tr, te))
    r.update(eval_type="loso", train_sites="+".join(train_sites), site=test_site)
    return r

# ---------------- handcrafted features ----------------

def image_features(path, size=256):
    im = Image.open(path).convert("RGB").resize((size, size))
    a = np.asarray(im, dtype=np.float32) / 255.0
    feats = []
    for c in range(3):
        ch = a[:, :, c]
        feats += [ch.mean(), ch.std(), np.percentile(ch, 25), np.percentile(ch, 75)]
    for c in range(3):
        h, _ = np.histogram(a[:, :, c], bins=16, range=(0, 1), density=True)
        feats += list(h)
    hsv = rgb2hsv(a)
    for c in range(3):
        h, _ = np.histogram(hsv[:, :, c], bins=16, range=(0, 1), density=True)
        feats += list(h)
    grey = a @ [0.299, 0.587, 0.114]
    lbp = local_binary_pattern((grey * 255).astype(np.uint8), P=8, R=1, method="uniform")
    h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    feats += list(h)
    return np.array(feats, dtype=np.float32)


def build_feature_cache(paths, cache=None):
    cache = {} if cache is None else cache
    for p in paths:
        if p not in cache:
            cache[p] = image_features(p)
    return cache

# ---------------- baseline models ----------------

def make_X(df, cache):
    return np.stack([cache[p] for p in df.filepath])


def fit_predict_majority(train_df, test_df):
    return np.full(len(test_df), train_df.risk_label.mean())


def fit_predict_random(train_df, test_df, seed=0):
    return np.random.default_rng(seed).random(len(test_df))


def make_rf_predictor(cache, seed=0):
    def f(train_df, test_df):
        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                   class_weight="balanced_subsample",
                                   random_state=seed, n_jobs=-1),
        )
        clf.fit(make_X(train_df, cache), train_df.risk_label.values)
        return clf.predict_proba(make_X(test_df, cache))[:, 1]
    return f

# ---------------- CNN ----------------

def make_cnn_predictor(arch="resnet18", epochs=8, bs=16, lr=3e-4,
                       img=224, seed=0, verbose=False):
    import torch, torch.nn as nn, timm
    from torch.utils.data import Dataset, DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(img, img), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),
        A.Normalize(), ToTensorV2(),
    ])
    eval_tf = A.Compose([A.Resize(img, img), A.Normalize(), ToTensorV2()])

    class OralDS(Dataset):
        def __init__(self, df, tf):
            self.paths = df.filepath.tolist()
            self.y = df.risk_label.values.astype(np.float32)
            self.tf = tf
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            im = np.array(Image.open(self.paths[i]).convert("RGB"))
            return self.tf(image=im)["image"], self.y[i]

    def f(train_df, test_df):
        torch.manual_seed(seed)
        model = timm.create_model(arch, pretrained=True, num_classes=1).to(DEVICE)
        rate = max(train_df.risk_label.mean(), 1e-6)
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(1 - rate) / rate], device=DEVICE))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        tl = DataLoader(OralDS(train_df, train_tf), batch_size=bs,
                        shuffle=True, num_workers=2)
        el = DataLoader(OralDS(test_df, eval_tf), batch_size=bs,
                        shuffle=False, num_workers=2)
        model.train()
        for ep in range(epochs):
            tot = 0.0
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(x).squeeze(1), y)
                loss.backward(); opt.step()
                tot += loss.item() * len(y)
            if verbose:
                print(f"  epoch {ep+1}: loss {tot/len(train_df):.4f}")
        model.eval()
        probs = []
        with torch.no_grad():
            for x, _ in el:
                probs.append(torch.sigmoid(model(x.to(DEVICE)).squeeze(1)).cpu().numpy())
        return np.concatenate(probs)
    return f
