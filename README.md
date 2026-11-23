# Cross-Disorder Prediction of Depression and PTSD

Code and results for:

> M. Kettani, S. Samine, F. Abbou. *Cross-Disorder Prediction of Depression and PTSD Using Machine Learning: A Comparative Study with Statistical Robustness Analysis.* To appear in the proceedings of CAISN'26 (Springer). The DOI will be added here on publication.

We test whether item responses on one questionnaire can flag probable caseness on the other: depression (PHQ-8-based label) from PCL-C items, and PTSD (PCL-C-based label) from PHQ-8 items. The data are 275 participants from the E-DAIC corpus. Logistic regression, random forest, SVM and an MLP are compared, with RFECV feature selection and SMOTE oversampling.

## Repository layout

```
├── src/
│   ├── run_analysis.py        # runs all experiments; writes tables, figures and the run log
│   └── make_paper_figures.py  # builds Figs. 1–3 of the paper at print size
├── results/
│   ├── figures/               # Figs. 1–3 and supplementary figures (PDF; PNG previews)
│   ├── tables/                # LaTeX table rows and CSVs (aggregate results only)
│   ├── results.json           # all metrics and selected hyperparameters
│   └── run_log.txt            # log of the run used in the paper
├── data/README.md             # how to obtain the data (data are NOT included)
├── requirements.txt           # exact package versions
├── CITATION.cff
└── LICENSE
```

## Two pipelines

`src/run_analysis.py` runs two pipelines on identical features, splits, search spaces and seeds:

| Pipeline | What it does | Used for |
|---|---|---|
| **Original** | RFECV and SMOTE fitted once on the whole training split before cross-validated tuning. Synthetic samples derived from validation-fold participants then enter the training folds, which inflates cross-validated scores. The test set is never resampled, so test results are valid. | Original CV and test F1 (Fig. 3); original McNemar tests |
| **Leakage-free** | RFECV → SMOTE → scaling → classifier inside one `imblearn` pipeline, refit in every CV training fold | Headline results: test performance (Table 1, Figs. 1–2), leakage-free CV (Fig. 3), repeated CV and item ablation (Table 2) |

## Reproducing the results

1. **Get the data.** E-DAIC is licensed and cannot be shared here; see [`data/README.md`](data/README.md). Place the file at `data/DAIC_dataset.csv`.

2. **Install** (Python 3.12) with the pinned versions. The leakage-free random-forest choice depends on the scikit-learn version, because two candidates tie within 0.0014 in CV F1 (see the paper).
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run:**
   ```bash
   python src/run_analysis.py --data data/DAIC_dataset.csv --out results
   python src/make_paper_figures.py --data data/DAIC_dataset.csv --results results
   ```
   The second command refits the leakage-free models from `results/results.json`, checks every confusion matrix and AUC against the saved results, and draws Figs. 1–3.

   Useful options for `run_analysis.py`:
   - `--jobs 1` if parallel jobs hang (seen on some Windows setups);
   - `--budget 250` saves progress and exits after about 250 s, so you can re-run the same command to resume;
   - `--repeats 10` for 10×5 repeated CV (the paper uses 5×5, the default);
   - `--smoke` runs tiny settings to test the installation (results not reportable).

   On a single CPU core the full run takes roughly 1–1.5 hours; with several cores it is much faster.

4. **Check reproduction.** For each model, `results/run_log.txt` prints the original pipeline's cross-validated and test F1 next to the values reported in the paper, marked `OK` when they match to four decimals.

## Data and citation requirements

The data come from E-DAIC (http://dcapswoz.ict.usc.edu/) and were used under its End User License Agreement: research use only, no redistribution, and no release of identifiable information. This repository therefore contains **no data and no participant-level outputs**, only aggregate results (model metrics, fold-level scores and feature statistics). Anyone using this code needs their own E-DAIC licence.

As the licence requires, any publication using these data must cite:

- J. Gratch, R. Artstein, G. Lucas, G. Stratou, S. Scherer, A. Nazarian, R. Wood, J. Boberg, D. DeVault, S. Marsella, D. Traum, S. Rizzo, L.-P. Morency. *The Distress Analysis Interview Corpus of human and computer interviews.* Proc. LREC 2014, pp. 3123–3128.
- F. Ringeval, B. Schuller, M. Valstar, N. Cummins, R. Cowie, L. Tavabi, M. Schmitt, et al. *AVEC 2019 workshop and challenge: State-of-mind, detecting depression with AI, and cross-cultural affect recognition.* Proc. 9th Int. Audio/Visual Emotion Challenge and Workshop (AVEC '19), pp. 3–12. ACM, 2019. https://doi.org/10.1145/3347320.3357688

## Use of generative AI

As stated in the paper's Methods section and AI Use Declaration, the authors used Claude (Anthropic), a large language model, to:
- review the original analysis code, which identified the data leakage;
- suggest and draft the code in `src/`;
- draft and edit parts of the manuscript;
- search for and verify references;
- copy-edit and format the paper.

The figures are plots generated from the data, not generative images. The authors reviewed all code, text and references, ran the analyses on the licensed E-DAIC data, and verified every reported number. The AI tool is not an author.

## License

Code: MIT (see `LICENSE`). The E-DAIC data are not part of this repository and remain subject to the E-DAIC End User License Agreement.
