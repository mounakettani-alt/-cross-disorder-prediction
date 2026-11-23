"""
run_analysis.py
===============
Runs every experiment in "Cross-Disorder Prediction of Depression and PTSD Using
Machine Learning" and writes the tables, figures and logs reported in the paper.

Two pipelines are run side by side on identical features, splits, search spaces,
candidate counts and seeds:

  ORIGINAL     : RFECV and SMOTE fitted once on the full training split, then
                 RandomizedSearchCV. Synthetic SMOTE points therefore reach the CV
                 validation folds, which inflates CV scores. The held-out test set is
                 never resampled, so its metrics are valid.
  LEAKAGE-FREE : RFECV -> SMOTE -> StandardScaler -> classifier inside one imblearn
                 Pipeline, so every data-dependent step is refit within each CV
                 training fold.

Usage
-----
    pip install -r requirements.txt
    python src/run_analysis.py --data data/DAIC_dataset.csv --out results
    python src/run_analysis.py --data data/DAIC_dataset.csv --out results --jobs 1      # if parallel jobs hang
    python src/run_analysis.py --data data/DAIC_dataset.csv --out results --budget 250  # stop & resume in chunks
    python src/run_analysis.py --data data/DAIC_dataset.csv --out smoke_test --smoke    # code check only

Outputs (in --out, default results/)
------------------------------------
    figures/*.pdf (+ .png previews)   tables/*.tex (LaTeX table rows)   tables/*.csv
    results.json                      run_log.txt (incl. reproduction check of the original pipeline)
"""
import argparse
import json
import os
import sys
import time
import warnings
from math import sqrt

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
T0 = time.time()
SEED = 42
N_ITER, RFE_TREES, RFE_STEP = 20, 100, 1  # paper settings; --smoke lowers them for a code check only
JOBS = -1  # set with --jobs; use 1 if joblib/loky hangs (e.g. some Windows setups)
MODELS = ["Logistic Regression", "Random Forest", "SVM", "Neural Network"]
SHORT = {"Logistic Regression": "LR", "Random Forest": "RF", "SVM": "SVM", "Neural Network": "MLP"}

# Values reported for the original pipeline (used only for the reproduction check)
PAPER = {
    "depression": {"cv_f1": [0.8935, 0.9131, 0.9543, 0.8975], "test_f1": [0.7586, 0.8000, 0.4211, 0.6452]},
    "ptsd": {"cv_f1": [0.8536, 0.8848, 0.8528, 0.8479], "test_f1": [0.8108, 0.9091, 0.8000, 0.7895]},
}


# =============================================================== data & features
def load(path):
    df = pd.read_csv(path)
    df["gender"] = df["gender"].replace({"male": 0, "female": 1, "unknown": 2}).astype(float)
    return df


def pcl_cols(df):
    return [c for c in df.columns if c.startswith("PCL-C_")]


def phq_cols(df):
    return [c for c in df.columns if c.startswith("PHQ8_")]


def features_depression(df, drop=()):
    """Features for predicting the depression label from PCL-C items (26 candidates)."""
    items = [c for c in pcl_cols(df) if c not in drop]
    X = df[items + ["gender", "age"]].copy()
    keep = lambda cols: [c for c in cols if c not in drop]
    X["re_experiencing"] = df[keep(["PCL-C_1_Memories", "PCL-C_2_Dreams", "PCL-C_3_Reliving",
                                    "PCL-C_4_Upset", "PCL-C_5_Physical"])].sum(axis=1)
    X["avoidance"] = df[keep(["PCL-C_6_ThoughtAvoidance", "PCL-C_7_ActivityAvoidance"])].sum(axis=1)
    X["cognition_mood"] = df[keep(["PCL-C_8_TroubleRemembering", "PCL-C_9_NoInterest",
                                   "PCL-C_10_FeelDistant", "PCL-C_11_FeelNumb",
                                   "PCL-C_12_Future_Cut_Short"])].sum(axis=1)
    X["arousal_reactivity"] = df[keep(["PCL-C_13_Sleep", "PCL-C_14_Irritability",
                                       "PCL-C_15_Concentration", "PCL-C_16_HyperAlert",
                                       "PCL-C_17_Jumpy"])].sum(axis=1)
    X["mood_to_arousal_ratio"] = X["cognition_mood"] / (X["arousal_reactivity"] + 1)
    X["symptom_variance"] = df[items].var(axis=1)
    X["total_pclc_score"] = df[items].sum(axis=1)
    return X, df["Depression_label"].astype(int)


def features_ptsd(df, drop=()):
    """Features for predicting the PTSD label from PHQ-8 items (17 candidates)."""
    items = [c for c in phq_cols(df) if c not in drop]
    X = df[items + ["gender", "age"]].copy()
    keep = lambda cols: [c for c in cols if c not in drop]
    if "PHQ8_1_NoInterest" not in drop:
        X["anhedonia"] = df["PHQ8_1_NoInterest"]
    X["depressed_mood"] = df["PHQ8_2_Depressed"]
    X["somatic_symptoms"] = df[keep(["PHQ8_3_Sleep", "PHQ8_4_Tired", "PHQ8_5_Appetite"])].sum(axis=1)
    X["cognitive_symptoms"] = df[keep(["PHQ8_6_Failure", "PHQ8_7_Concentration"])].sum(axis=1)
    X["psychomotor"] = df["PHQ8_8_Psychomotor"]
    X["cognitive_to_somatic_ratio"] = X["cognitive_symptoms"] / (X["somatic_symptoms"] + 1)
    X["symptom_variance"] = df[items].var(axis=1)
    return X, df["PTSD_label"].astype(int)


TASK = None
TASKS = {
    "depression": dict(label=r"Depr.\,$\leftarrow$\,PTSD", title="Depression from PCL-C items",
                       feats=features_depression,
                       overlap=["PCL-C_9_NoInterest", "PCL-C_13_Sleep", "PCL-C_15_Concentration"]),
    "ptsd": dict(label=r"PTSD\,$\leftarrow$\,Depr.", title="PTSD from PHQ-8 items",
                 feats=features_ptsd,
                 overlap=["PHQ8_1_NoInterest", "PHQ8_3_Sleep", "PHQ8_7_Concentration"]),
}


# =============================================================== models (search spaces)
def search_space():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=SEED), {
            "C": np.logspace(-4, 4, 20), "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"], "class_weight": [None, "balanced"]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED), {
            "n_estimators": [50, 100, 200, 300], "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False], "class_weight": [None, "balanced", "balanced_subsample"]}),
        "SVM": (SVC(probability=True, random_state=SEED, max_iter=10_000_000), {  # cap: see run_log
            "C": np.logspace(-3, 3, 10), "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"] + list(np.logspace(-3, 3, 5)),
            "class_weight": [None, "balanced"]}),
        "Neural Network": (MLPClassifier(max_iter=1000, random_state=SEED), {
            "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
            "activation": ["relu", "tanh"], "solver": ["adam", "sgd"],
            "alpha": np.logspace(-5, 1, 10), "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": np.logspace(-4, -1, 5), "early_stopping": [True]}),
    }


def rfecv():
    return RFECV(estimator=RandomForestClassifier(n_estimators=RFE_TREES, random_state=SEED),
                 step=RFE_STEP, cv=5, scoring="f1", n_jobs=JOBS)


def inner_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def search(est, grid, X, y):
    rs = RandomizedSearchCV(est, param_distributions=grid, n_iter=N_ITER, cv=inner_cv(),
                            scoring="f1", random_state=SEED, n_jobs=JOBS)
    return rs.fit(X, y)


# =============================================================== protocols
def run_original(Xtr, ytr, Xte):
    """Original pipeline: RFECV and SMOTE fitted once on the training split, then tuned."""
    sel = rfecv().fit(Xtr, ytr)
    cols = Xtr.columns[sel.support_]
    Xr, yr = SMOTE(random_state=SEED).fit_resample(Xtr[cols], ytr)
    sc = StandardScaler().fit(Xr)
    Xr_s, Xte_s = sc.transform(Xr), sc.transform(Xte[cols])
    out = {"selected": list(cols), "rfecv_curve": list(sel.cv_results_["mean_test_score"])}
    for name, (est, grid) in search_space().items():
        rs = search(est, grid, Xr_s, yr)
        out[name] = dict(cv_f1=rs.best_score_, params=rs.best_params_,
                         prob=rs.best_estimator_.predict_proba(Xte_s)[:, 1],
                         pred=rs.best_estimator_.predict(Xte_s))
    return out


def run_corrected(Xtr, ytr, Xte, cachedir):
    """Same steps and grids, but every step is refit inside each CV training fold."""
    out = {}
    for name, (est, grid) in search_space().items():
        out[name] = cached(f"{TASK}_corrected_{SHORT[name]}",
                           lambda: _corrected_one(name, est, grid, Xtr, ytr, Xte, cachedir))
    return out


def _corrected_one(name, est, grid, Xtr, ytr, Xte, cachedir):
    if True:
        t = time.time()
        pipe = Pipeline([("select", rfecv()), ("smote", SMOTE(random_state=SEED)),
                         ("scale", StandardScaler()), ("clf", est)], memory=cachedir)
        rs = search(pipe, {f"clf__{k}": v for k, v in grid.items()}, Xtr, ytr)
        best = rs.best_estimator_
        out = {}
        out[name] = dict(cv_f1=rs.best_score_,
                         params={k.replace("clf__", ""): v for k, v in rs.best_params_.items()},
                         prob=best.predict_proba(Xte)[:, 1], pred=best.predict(Xte),
                         selected=list(Xtr.columns[best.named_steps["select"].support_]))
        print(f"    {name}: {time.time() - t:.0f}s", flush=True)
        return out[name]


def tuned_estimator(name, params):
    est, _ = search_space()[name]
    return est.set_params(**params)


def repeated_cv(X, y, params, repeats, models=MODELS, tag=None):
    """10x5 repeated stratified CV. RFECV is fit once per outer training fold (it does not
    depend on the classifier) and shared by all models; SMOTE and scaling are refit per fold."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    rows = []
    for k, (tr, va) in enumerate(rskf.split(X, y)):
        fpath = os.path.join(CKPT, f"{tag}_fold{k:03d}.pkl") if tag else None
        if fpath and os.path.exists(fpath):
            rows.extend(pickle.load(open(fpath, "rb")))
            continue
        check_time()
        n0 = len(rows)
        Xtr, ytr, Xva, yva = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
        sel = rfecv().fit(Xtr, ytr)
        cols = Xtr.columns[sel.support_]
        Xr, yr = SMOTE(random_state=SEED).fit_resample(Xtr[cols], ytr)
        sc = StandardScaler().fit(Xr)
        for name in models:
            clf = tuned_estimator(name, params[name]).fit(sc.transform(Xr), yr)
            p = clf.predict_proba(sc.transform(Xva[cols]))[:, 1]
            yhat = clf.predict(sc.transform(Xva[cols]))
            rows.append(dict(fold=k, model=name, n_features=len(cols),
                             f1=f1_score(yva, yhat, zero_division=0),
                             precision=precision_score(yva, yhat, zero_division=0),
                             recall=recall_score(yva, yhat, zero_division=0),
                             auc=roc_auc_score(yva, p)))
        if fpath:
            pickle.dump(rows[n0:], open(fpath, "wb"))
        print(f"    fold {k + 1}/{5 * repeats}  t={time.time() - T0:.0f}s", flush=True)
    print()
    return pd.DataFrame(rows)


# =============================================================== checkpointing
import pickle
CKPT = None
DEADLINE = float("inf")


class OutOfTime(Exception):
    pass


def check_time():
    if time.time() > DEADLINE:
        raise OutOfTime


def cached(name, fn):
    """Run fn() once and pickle the result, so an interrupted run can resume."""
    path = os.path.join(CKPT, name + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    check_time()
    out = fn()
    pickle.dump(out, open(path, "wb"))
    return out


# =============================================================== statistics
def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _auc(y, p):
    """Mann-Whitney AUC (ties count 1/2); identical to roc_auc_score."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def bootstrap(y, pred, prob, B=10000):
    """Percentile bootstrap (resampling test cases) for F1 and ROC-AUC."""
    rng = np.random.default_rng(SEED)
    y, pred, prob = (np.asarray(v) for v in (y, pred, prob))
    idx = rng.integers(0, len(y), (B, len(y)))
    Y, P = y[idx], pred[idx]
    tp = ((Y == 1) & (P == 1)).sum(1); fp = ((Y == 0) & (P == 1)).sum(1); fn = ((Y == 1) & (P == 0)).sum(1)
    d = 2 * tp + fp + fn
    f1s = np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)
    aucs = np.array([_auc(y[i], prob[i]) for i in idx])
    return np.percentile(f1s, [2.5, 97.5]), np.nanpercentile(aucs, [2.5, 97.5])


def test_metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    f1_ci, auc_ci = bootstrap(y, pred, prob)
    return dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                accuracy=(tp + tn) / n, acc_ci=wilson(tp + tn, n),
                precision=tp / (tp + fp) if tp + fp else 0.0, prec_ci=wilson(tp, tp + fp),
                recall=tp / (tp + fn), rec_ci=wilson(tp, tp + fn),
                f1=2 * tp / (2 * tp + fp + fn), f1_ci=tuple(f1_ci),
                auc=roc_auc_score(y, prob), auc_ci=tuple(auc_ci),
                spec=tn / (tn + fp), npv=tn / (tn + fn) if tn + fn else 0.0)


def mcnemar(y, a, b):
    y, a, b = map(np.asarray, (y, a, b))
    n01 = int(np.sum((a == y) & (b != y)))
    n10 = int(np.sum((a != y) & (b == y)))
    p = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    return n01, n10, p


# =============================================================== figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
    return plt


W = 4.8  # LNCS text width in inches (12.2 cm)
COLORS = {"Logistic Regression": "#1f77b4", "Random Forest": "#d62728",
          "SVM": "#2ca02c", "Neural Network": "#9467bd"}


def save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    fig.savefig(os.path.join(out, f"{name}.png"))


def fig_confusion(plt, res, ytest, protocol, out):
    fig, axes = plt.subplots(2, 4, figsize=(W, 2.75))
    for r, task in enumerate(TASKS):
        for c, name in enumerate(MODELS):
            ax = axes[r, c]
            cm = confusion_matrix(ytest[task], res[task][protocol][name]["pred"], labels=[0, 1])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > cm.sum() * 0.45 else "black")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
            if r == 1:
                ax.set_xlabel("Predicted")
            if c == 0:
                ax.set_ylabel(f"{'Depr.' if task == 'depression' else 'PTSD'}\nTrue")
            if r == 0:
                ax.set_title(SHORT[name])
            for s in ax.spines.values():
                s.set_visible(True)
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    save(fig, out, f"fig_confusion_matrices_{protocol}")
    plt.close(fig)


def fig_roc(plt, res, ytest, protocol, out):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), sharey=True)
    for ax, task in zip(axes, TASKS):
        for name in MODELS:
            p = res[task][protocol][name]["prob"]
            fpr, tpr, _ = roc_curve(ytest[task], p)
            ax.plot(fpr, tpr, color=COLORS[name], lw=1.1,
                    label=f"{SHORT[name]} (AUC = {roc_auc_score(ytest[task], p):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.6)
        ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="False positive rate",
               title=TASKS[task]["title"])
        ax.legend(loc="lower right", frameon=False)
        ax.set_aspect("equal")
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout()
    save(fig, out, f"fig_roc_curves_{protocol}")
    plt.close(fig)


def fig_cv_vs_test(plt, res, out):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
    x = np.arange(len(MODELS)); w = 0.26
    for ax, task in zip(axes, TASKS):
        o, c = res[task]["original"], res[task]["corrected"]
        bars = [([o[m]["cv_f1"] for m in MODELS], "CV, original (SMOTE before CV)", "#bdbdbd"),
                ([c[m]["cv_f1"] for m in MODELS], "CV, corrected (SMOTE in folds)", "#6baed6"),
                ([res[task]["test_original"][m]["f1"] for m in MODELS], "Held-out test", "#08306b")]
        for k, (vals, lab, col) in enumerate(bars):
            b = ax.bar(x + (k - 1) * w, vals, w, label=lab, color=col, edgecolor="black", lw=0.4)
            for rect, v in zip(b, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.5, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in MODELS])
        ax.set_ylim(0, 1.18); ax.set_title(TASKS[task]["title"])
    axes[0].set_ylabel("$F_1$ (positive class)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "fig_cv_vs_test")
    plt.close(fig)


def fig_repcv(plt, rep, out):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), sharey=True)
    for ax, task in zip(axes, TASKS):
        d = rep[task]
        data = [d.loc[d.model == m, "f1"].values for m in MODELS]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops=dict(color="black"),
                        flierprops=dict(markersize=2))
        for patch, m in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.55)
        ax.set_xticklabels([SHORT[m] for m in MODELS]); ax.set_title(TASKS[task]["title"])
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("$F_1$ per validation fold")
    fig.tight_layout()
    save(fig, out, "fig_repeated_cv")
    plt.close(fig)


def fig_rfecv(plt, res, out):
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.9))
    for ax, task in zip(axes, TASKS):
        curve = res[task]["original"]["rfecv_curve"]
        n = len(res[task]["original"]["selected"])
        ax.plot(range(1, len(curve) + 1), curve, "o-", ms=2, lw=0.9, color="#08306b")
        ax.axvline(n, ls="--", lw=0.6, color="gray")
        ax.set(xlabel="Number of features", title=f"{TASKS[task]['title']} ({n} selected)")
    axes[0].set_ylabel("CV $F_1$")
    fig.tight_layout()
    save(fig, out, "fig_rfecv")
    plt.close(fig)


def fig_anova(plt, anova, out):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9))
    for ax, task in zip(axes, TASKS):
        s = anova[task].head(12).iloc[::-1]
        ax.barh(s.index, s.values, color="#4292c6", edgecolor="black", lw=0.3)
        ax.set(xlabel="ANOVA $F$", title=TASKS[task]["title"])
        ax.tick_params(axis="y", labelsize=5.8)
    fig.tight_layout()
    save(fig, out, "fig_anova_importance")
    plt.close(fig)


# =============================================================== LaTeX tables
def f3(v):
    return f"{v:.3f}"


def ci(t):
    return f"[{t[0]:.3f}, {t[1]:.3f}]"


def tex_escape(s):
    return s.replace("_", r"\_")


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def tables(res, rep, abl, mcn, out):
    # Table 1: hyperparameters + CV F1 under both protocols
    L = []
    for task in TASKS:
        L.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{TASKS[task]['title']}}}}}\\")
        for m in MODELS:
            p = res[task]["original"][m]["params"]
            ps = ", ".join(f"{tex_escape(k)} = {v:.4g}" if isinstance(v, float)
                           else f"{tex_escape(k)} = {v}" for k, v in sorted(p.items()))
            L.append(f"{SHORT[m]} & {ps} & {f3(res[task]['original'][m]['cv_f1'])} & "
                     f"{f3(res[task]['corrected'][m]['cv_f1'])} \\\\")
        L.append(r"\midrule")
    write(os.path.join(out, "tab_hyperparams.tex"), "\n".join(L[:-1]) + "\n")

    # Table 2 / 4: test performance (counts, metrics, CIs) for both protocols
    for proto in ["original", "corrected"]:
        A, B = [], []
        for task in TASKS:
            hdr = rf"\multicolumn{{11}}{{@{{}}l}}{{\textit{{{TASKS[task]['title']}}}}}\\"
            A.append(hdr); B.append(hdr.replace("{11}", "{6}"))
            best = max(MODELS, key=lambda m: res[task][f"test_{proto}"][m]["f1"])
            for m in MODELS:
                t = res[task][f"test_{proto}"][m]
                b = (lambda s: rf"\textbf{{{s}}}") if m == best else (lambda s: s)
                A.append(f"{SHORT[m]} & {t['TN']} & {t['FP']} & {t['FN']} & {t['TP']} & "
                         f"{b(f3(t['accuracy']))} & {b(f3(t['precision']))} & {b(f3(t['recall']))} & "
                         f"{b(f3(t['spec']))} & {b(f3(t['f1']))} & {b(f3(t['auc']))} \\\\")
                B.append(f"{SHORT[m]} & {ci(t['acc_ci'])} & {ci(t['prec_ci'])} & {ci(t['rec_ci'])} & "
                         f"{ci(t['f1_ci'])} & {ci(t['auc_ci'])} \\\\")
            A.append(r"\midrule"); B.append(r"\midrule")
        write(os.path.join(out, f"tab_test_{proto}.tex"), "\n".join(A[:-1]) + "\n")
        write(os.path.join(out, f"tab_ci_{proto}.tex"), "\n".join(B[:-1]) + "\n")

    # Table 5: repeated CV
    L = []
    for task in TASKS:
        L.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{TASKS[task]['title']}}}}}\\")
        d = rep[task]
        for m in MODELS:
            s = d[d.model == m]
            L.append(f"{SHORT[m]} & " + " & ".join(
                f"{s[c].mean():.3f} $\\pm$ {s[c].std(ddof=1):.3f}"
                for c in ["f1", "recall", "precision", "auc"]) + r" \\")
        L.append(r"\midrule")
    write(os.path.join(out, "tab_repeated_cv.tex"), "\n".join(L[:-1]) + "\n")

    # Table 6: ablation
    L = []
    for task in TASKS:
        full, ab = abl[task]["full"], abl[task]["ablated"]
        dF1 = ab.f1.values - full.f1.values
        dAUC = ab.auc.values - full.auc.values
        items = ", ".join(i.split("_")[1] for i in TASKS[task]["overlap"])
        L.append(rf"\multirow{{2}}{{*}}{{{TASKS[task]['label']}}} & All items & "
                 f"{full.f1.mean():.3f} $\\pm$ {full.f1.std(ddof=1):.3f} & "
                 f"{full.auc.mean():.3f} $\\pm$ {full.auc.std(ddof=1):.3f} & -- \\\\")
        L.append(f" & Without items {items} & {ab.f1.mean():.3f} $\\pm$ {ab.f1.std(ddof=1):.3f} & "
                 f"{ab.auc.mean():.3f} $\\pm$ {ab.auc.std(ddof=1):.3f} & "
                 f"{dF1.mean():+.3f} / {dAUC.mean():+.3f} \\\\")
        L.append(r"\midrule")
    write(os.path.join(out, "tab_ablation.tex"), "\n".join(L[:-1]) + "\n")

    # Table 7: McNemar
    L = []
    for task in TASKS:
        for m, (n01, n10, p) in mcn[task].items():
            L.append(f"{TASKS[task]['label']} & RF vs {SHORT[m]} & {n01} & {n10} & {p:.3f} \\\\")
        L.append(r"\midrule")
    write(os.path.join(out, "tab_mcnemar.tex"), "\n".join(L[:-1]) + "\n")


# =============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--repeats", type=int, default=5,
                    help="repeats of stratified 5-fold CV (paper: 5)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--budget", type=float, default=0,
                    help="seconds before checkpointing and exiting (rerun to resume); 0 = no limit")
    ap.add_argument("--smoke", action="store_true", help="tiny settings to test the code; results NOT reportable")
    a = ap.parse_args()
    global JOBS, N_ITER, RFE_TREES, RFE_STEP
    JOBS = a.jobs
    if a.smoke:
        N_ITER, RFE_TREES, RFE_STEP, a.repeats = 2, 10, 4, 1
    figdir, tabdir = os.path.join(a.out, "figures"), os.path.join(a.out, "tables")
    os.makedirs(figdir, exist_ok=True); os.makedirs(tabdir, exist_ok=True)
    log = open(os.path.join(a.out, "run_log.txt"), "a")
    say = lambda *s: (print(*s), print(*s, file=log), log.flush())

    df = load(a.data)
    say(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    na = int(df.isna().sum().sum())
    say(f"Missing values in dataset: {na}" + ("" if na == 0 else "  <-- the pipelines do NOT impute"))

    res, ytest, rep, abl, mcn, anova = {}, {}, {}, {}, {}, {}
    global CKPT, DEADLINE
    CKPT = os.path.join(a.out, "checkpoints"); os.makedirs(CKPT, exist_ok=True)
    if a.budget:
        DEADLINE = time.time() + a.budget
    cachedir = os.path.join(a.out, "pipeline_cache")
    t0 = time.time()
    global TASK
    for task, cfg in TASKS.items():
        TASK = task
        say(f"\n=== {task.upper()} ===")
        X, y = cfg["feats"](df)
        anova[task] = pd.Series(SelectKBest(f_classif, k="all").fit(X, y).scores_,
                                index=X.columns).sort_values(ascending=False)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        ytest[task] = yte.values
        say(f"train {len(ytr)} (pos {ytr.mean():.3f}), test {len(yte)} (pos {int(yte.sum())})")

        say("  original protocol (reproduction) ...")
        o = cached(f"{task}_original", lambda: run_original(Xtr, ytr, Xte))
        say(f"  RFECV selected {len(o['selected'])}: {o['selected']}")
        say("  corrected protocol (SMOTE + RFECV inside folds) ...")
        c = cached(f"{task}_corrected", lambda: run_corrected(Xtr, ytr, Xte, cachedir))
        res[task] = {"original": o, "corrected": c,
                     "test_original": {m: test_metrics(yte, o[m]["pred"], o[m]["prob"]) for m in MODELS},
                     "test_corrected": {m: test_metrics(yte, c[m]["pred"], c[m]["prob"]) for m in MODELS}}

        say("  reproduction check against the paper (original protocol):")
        for i, m in enumerate(MODELS):
            cvp, tp = PAPER[task]["cv_f1"][i], PAPER[task]["test_f1"][i]
            cvo, to = o[m]["cv_f1"], res[task]["test_original"][m]["f1"]
            flag = "OK" if abs(cvo - cvp) < 5e-4 and abs(to - tp) < 5e-4 else "DIFFERS"
            say(f"    {m:20s} CV {cvo:.4f} (paper {cvp:.4f})  test F1 {to:.4f} (paper {tp:.4f})  {flag}")
        say("  CV F1 original -> corrected:  " +
            ", ".join(f"{SHORT[m]} {o[m]['cv_f1']:.3f}->{c[m]['cv_f1']:.3f}" for m in MODELS))

        mcn[task] = {m: mcnemar(yte, o["Random Forest"]["pred"], o[m]["pred"])
                     for m in MODELS if m != "Random Forest"}

        params = {m: c[m]["params"] for m in MODELS}
        say(f"  repeated stratified CV ({a.repeats}x5), corrected pipeline ...")
        rep[task] = repeated_cv(X, y, params, a.repeats, tag=f"{task}_rep")
        say("  overlap ablation (Random Forest) ...")
        Xa, _ = cfg["feats"](df, drop=cfg["overlap"])
        abl[task] = {"full": rep[task][rep[task].model == "Random Forest"].reset_index(drop=True),
                     "ablated": repeated_cv(Xa, y, params, a.repeats, models=["Random Forest"],
                                                tag=f"{task}_abl")}
        say(f"  elapsed {time.time() - t0:.0f}s")


    plt = setup_mpl()
    for proto in ["original", "corrected"]:
        fig_confusion(plt, res, ytest, proto, figdir)
        fig_roc(plt, res, ytest, proto, figdir)
    fig_cv_vs_test(plt, res, figdir)
    fig_repcv(plt, rep, figdir)
    fig_rfecv(plt, res, figdir)
    fig_anova(plt, anova, figdir)
    tables(res, rep, abl, mcn, tabdir)

    for task in TASKS:
        rep[task].to_csv(os.path.join(tabdir, f"repeated_cv_folds_{task}.csv"), index=False)
        anova[task].to_csv(os.path.join(tabdir, f"anova_{task}.csv"), header=["F"])

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if k not in ("prob", "pred")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    json.dump(clean({"results": res, "mcnemar": mcn,
                     "repeated_cv_summary": {t: rep[t].groupby("model")[["f1", "recall", "precision", "auc"]]
                                             .agg(["mean", "std"]).round(4).to_dict() for t in TASKS}}),
              open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    say(f"\nDone in {time.time() - t0:.0f}s. Figures -> {figdir}, tables -> {tabdir}")


if __name__ == "__main__":
    try:
        main()
    except OutOfTime:
        print("\nTime budget reached; progress saved. Re-run the same command to resume.")
        sys.exit(3)
