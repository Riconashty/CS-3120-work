# Near-Earth Object Hazard Classification

Binary classification of NASA Near-Earth Objects (NEOs) as potentially hazardous, using only measurable orbital and physical characteristics.

**Stack:** Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter
**Course:** CS 3120 — Machine Learning, MSU Denver (Spring 2026)

---

## Dataset

2,000 near-Earth objects pulled from NASA's [NeoWs](https://api.nasa.gov/) (Near Earth Object Web Service) API, 39 raw fields per record.

**Target:** `potentially_hazardous` — a boolean flag indicating a theoretical collision risk with Earth. Class balance is 533 hazardous / 1,467 not hazardous (26.7% positive).

**Key features:** minimum orbit intersection distance, perihelion and aphelion distance, orbital inclination, orbital period, diameter bounds, absolute magnitude, orbit uncertainty, and orbit class (Apollo / Amor / Aten / Interior Earth Object).

## Pipeline

1. **Cleaning** — dropped identifier, URL, free-text, and constant-valued columns, plus two near-empty fields (`sentry_data` missing in 1,997 of 2,000 rows).
2. **Feature engineering** — parsed observation dates into a `first_obs_year` feature; one-hot encoded `orbit_class_type` with `drop_first` to avoid the dummy variable trap.
3. **EDA** — class balance, grouped means and standard deviations by hazard status, per-feature histograms and boxplots, a full correlation matrix, and hazard rate by orbit class.
4. **Modeling** — stratified 80/20 train/test split; class-weighted Random Forest (200 trees) and standardized Logistic Regression.
5. **Evaluation** — precision, recall, F1, ROC-AUC, and confusion matrices, with recall on the hazardous class treated as the primary metric.

Final shape after preprocessing: **2,000 × 24**.

## Hypotheses and results

**H1 — Smaller perihelion distance, minimum orbit intersection, and inclination are each associated with the hazardous designation.**

Supported. Random Forest feature importances ranked minimum orbit intersection distance first, perihelion distance second, and inclination fourth. Group means separate cleanly: mean minimum orbit intersection is 0.197 AU for non-hazardous objects versus 0.022 AU for hazardous ones — an 89% relative difference, the largest of any feature.

**H2 — A classifier trained on these features achieves high recall on hazardous objects, because the PHO designation is itself derived from these same measurable quantities.**

Supported, and this is the interesting part.

| Model | Hazardous precision | Hazardous recall | Accuracy | ROC-AUC |
|---|---|---|---|---|
| Random Forest (balanced, 200 trees) | 1.00 | 0.99 | 1.00 | 1.0000 |
| Logistic Regression (balanced, scaled) | 0.91 | 1.00 | 0.97 | 0.9983 |

## Interpretation

**These scores are not evidence of a strong model.** They are evidence that the label is a deterministic function of the features. NASA assigns the potentially-hazardous designation from minimum orbit intersection distance and absolute magnitude — both of which sit in the feature matrix. The models are recovering a threshold rule, not discovering a pattern, and reporting ROC-AUC of 1.000 as predictive skill would misrepresent what happened.

That framing is the actual finding: asteroid hazard classification is not a black box, and the models confirm it is driven by a small number of well-understood orbital quantities.

The genuine modeling question left over is the precision/recall tradeoff. Logistic regression caught every hazardous object at the cost of some false positives; Random Forest missed at least one unless `max_depth` was constrained below 5. In planetary defense the cost is asymmetric — a missed hazardous object is far worse than a false alarm — so recall should drive tuning, and logistic regression is the better choice here despite its lower headline accuracy.

## Files

- `asteroid_analysis.ipynb` — full analysis, from data load through model evaluation
- `RNH_Project_Presentation_Final.pdf` — slide deck presented to faculty and peers

## Notes on AI tool usage

Claude and Gemini were used for exploratory data analysis guidance, feature definition lookups, model training strategy, and debugging inside Google Colab. All modeling decisions, hypotheses, and interpretation are my own.
