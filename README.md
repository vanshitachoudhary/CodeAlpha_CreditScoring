# 💳 Credit Scoring AI

![CI](https://github.com/<your-username>/CodeAlpha_CreditScoring/actions/workflows/ci.yml/badge.svg)

End-to-end credit risk system: **training pipeline → business-aware decisioning → explainable Streamlit app → Docker**.
Built for the CodeAlpha ML Internship (Task 1) on the German Credit dataset (1,000 applicants, 30% defaults).

## Highlights
- **Leak-free sklearn `Pipeline`** (scaling + one-hot inside CV) comparing Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Imbalance handling** via class weights; evaluated with Precision, Recall, F1, ROC-AUC, 5-fold CV and **Brier score**
- **Calibration check**: are predicted probabilities trustworthy?
- **Cost-sensitive threshold**: a missed defaulter costs 5x a wrongly rejected customer, so the decision threshold is optimised on business cost, not 0.5
- **Explainability**: per-applicant "why this result?" via model-agnostic what-if analysis
- **App**: single-applicant scoring with credit-score gauge, batch CSV scoring, model performance dashboard
- **Engineering**: shared feature module, unit tests, GitHub Actions CI, Dockerfile

## Results
> After `python train.py`, paste `outputs/model_comparison.csv` here.

| Model | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |
| Gradient Boosting | | | | | |

Cost at default threshold 0.5 vs optimal threshold: *(copy the two numbers printed by train.py)*

## Quick start
```bash
pip install -r requirements.txt
python train.py            # train, evaluate, save model + plots
streamlit run app.py       # live demo  ->  http://localhost:8501
pip install -r requirements-dev.txt && pytest   # tests
```
Docker: `docker build -t credit-scoring . && docker run -p 8501:8501 credit-scoring`

## Project structure
```
features.py        shared feature engineering (used by training and app)
train.py           pipeline, model comparison, calibration, cost-optimal threshold
app.py             Streamlit app
tests/             unit tests
Dockerfile, .github/workflows/ci.yml
```

## Limitations & responsible use
- Small dataset (1,000 rows): metrics have high variance; treat them as indicative, not production-grade.
- Dataset contains sensitive attributes (age, personal status, foreign worker). A production system needs a fairness audit and legal review before using such features.
- The 300-850 score is an illustrative linear rescaling of default probability, not a regulated scorecard.
- Explanations are what-if sensitivities, not Shapley values.

## Next steps
SHAP explanations, hyper-parameter search, fairness metrics by group, FastAPI serving layer.
