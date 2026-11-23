# Data

This folder is intentionally empty. **Do not commit the dataset.**

The analyses use item-level PHQ-8 and PCL-C responses from the Extended Distress
Analysis Interview Corpus (E-DAIC), distributed by the University of Southern
California Institute for Creative Technologies under an End User License Agreement
(research use only; no redistribution of the database or any part of it). Request access through the DAIC-WOZ / E-DAIC
data request process, then place the file here as:

    data/DAIC_dataset.csv

Expected format: 275 rows × 33 columns:
`Participant, PHQ8_1_NoInterest … PHQ8_8_Psychomotor, Depression_severity,
PCL-C_1_Memories … PCL-C_17_Jumpy, PTSD_severity, gender, age,
Depression_label, PTSD_label, split`.

`.gitignore` excludes every CSV in this folder, so the file cannot be committed by accident.

Publications using the data must cite Gratch et al. (LREC 2014) and Ringeval et al.
(AVEC 2019); see the main README for full references.
