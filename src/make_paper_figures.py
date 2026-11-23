"""
make_paper_figures.py
Rebuilds Figs. 1-3 of the paper at their final printed size (LNCS text width,
7 pt lettering) from the outputs of run_analysis.py.

The leakage-free models are refitted on the training split from the hyperparameters
saved in results/results.json, using the same pipeline as the paper. Every confusion
matrix and AUC is checked against the saved results before plotting.

    python src/run_analysis.py --data data/DAIC_dataset.csv --out results
    python src/make_paper_figures.py --data data/DAIC_dataset.csv --results results
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_analysis as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--results', default='results')
ap.add_argument('--jobs', type=int, default=-1)
a = ap.parse_args()
R.JOBS = a.jobs
res = json.load(open(os.path.join(a.results, 'results.json')))['results']
out = os.path.join(a.results, 'figures'); os.makedirs(out, exist_ok=True)
df = R.load(a.data); M = R.MODELS; SH = R.SHORT
COL = {'Logistic Regression': '#1f77b4', 'Random Forest': '#d62728', 'SVM': '#2ca02c', 'Neural Network': '#9467bd'}
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

# refit leakage-free models and collect test predictions
P = {}
for t in ['depression', 'ptsd']:
    X, y = R.TASKS[t]['feats'](df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=R.SEED, stratify=y)
    for m in M:
        p = dict(res[t]['corrected'][m]['params'])
        if 'hidden_layer_sizes' in p: p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
        est, _ = R.search_space()[m]
        pipe = Pipeline([('select', R.rfecv()), ('smote', SMOTE(random_state=R.SEED)), ('scale', StandardScaler()), ('clf', est)])
        pipe.set_params(**{f'clf__{k}': v for k, v in p.items()}).fit(Xtr, ytr)
        pr, pb = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        s = res[t]['test_corrected'][m]
        cm = confusion_matrix(yte, pr, labels=[0, 1]).ravel().tolist()
        assert cm == [s['TN'], s['FP'], s['FN'], s['TP']], f'{t} {m}: refit {cm} differs from saved results'
        assert abs(roc_auc_score(yte, pb) - s['auc']) < 1e-9, f'{t} {m}: AUC differs from saved results'
        P[(t, m)] = (yte.values, pr, pb)
        print(f'{t:10s} {SH[m]:4s} reproduced: CM {cm}, AUC {s["auc"]:.3f}', flush=True)

# Fig. 1: confusion matrices
fig, axes = plt.subplots(1, 8, figsize=(4.8, 1.2))
for k, t in enumerate(['depression', 'ptsd']):
    for j, m in enumerate(M):
        ax = axes[4 * k + j]; y, pr, _ = P[(t, m)]; cm = confusion_matrix(y, pr, labels=[0, 1])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.sum())
        for r in range(2):
            for c in range(2):
                ax.text(c, r, cm[r, c], ha='center', va='center', fontsize=7, color='white' if cm[r, c] > 25 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['N', 'P']); ax.set_yticks([0, 1])
        ax.set_yticklabels(['N', 'P'] if (j == 0 and k == 0) else ['', '']); ax.tick_params(length=0, pad=1)
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)
        ax.set_title(SH[m], pad=2)
        if j == 0 and k == 0: ax.set_ylabel('True', labelpad=1)
fig.text(0.29, 0.92, 'Depression (from PCL-C items)', ha='center'); fig.text(0.755, 0.92, 'PTSD (from PHQ-8 items)', ha='center')
fig.text(0.5, 0.01, 'Predicted', ha='center')
fig.subplots_adjust(left=0.05, right=0.995, top=0.74, bottom=0.18, wspace=0.12)
fig.savefig(os.path.join(out, 'fig1_confusion_leakage_free.pdf')); plt.close(fig)

# Fig. 2: ROC curves
fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.5), sharey=True)
for ax, (t, lab) in zip(axes, [('depression', 'Depression'), ('ptsd', 'PTSD')]):
    for m in M:
        y, _, pb = P[(t, m)]; f, tp, _ = roc_curve(y, pb)
        ax.plot(f, tp, color=COL[m], lw=0.9, label=f'{SH[m]} (AUC {roc_auc_score(y, pb):.3f})')
    ax.plot([0, 1], [0, 1], 'k:', lw=0.6); ax.set(xlim=(0, 1), ylim=(0, 1.02)); ax.set_title(lab, pad=2)
    ax.set_xlabel('False positive rate', labelpad=1)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=1, handlelength=1.4, borderpad=0.3)
axes[0].set_ylabel('True positive rate', labelpad=1)
fig.subplots_adjust(left=0.115, right=0.975, top=0.88, bottom=0.22, wspace=0.12)
fig.savefig(os.path.join(out, 'fig2_roc_leakage_free.pdf')); plt.close(fig)

# Fig. 3: cross-validated vs test F1, both pipelines
fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.32), sharey=True); x = np.arange(4); w = 0.2
for ax, (t, lab) in zip(axes, [('depression', 'Depression'), ('ptsd', 'PTSD')]):
    r = res[t]
    series = [([r['original'][m]['cv_f1'] for m in M], 'Original CV', '#c6dbef'), ([r['test_original'][m]['f1'] for m in M], 'Original test', '#2171b5'),
              ([r['corrected'][m]['cv_f1'] for m in M], 'Leakage-free CV', '#fdae6b'), ([r['test_corrected'][m]['f1'] for m in M], 'Leakage-free test', '#d94801')]
    for k, (v, l, c) in enumerate(series): ax.bar(x + (k - 1.5) * w, v, w, color=c, edgecolor='black', lw=0.3, label=l)
    ax.set_xticks(x); ax.set_xticklabels([SH[m] for m in M]); ax.set_ylim(0, 1); ax.set_title(lab, pad=2)
axes[0].set_ylabel('$F_1$', labelpad=1)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=1.2, columnspacing=1.0)
fig.subplots_adjust(left=0.1, right=0.985, top=0.88, bottom=0.3, wspace=0.08)
fig.savefig(os.path.join(out, 'fig3_cv_vs_test.pdf')); plt.close(fig)
print('Figures 1-3 written to', out)
