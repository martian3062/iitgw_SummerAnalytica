# iitgw_SummerAnalytica

Script-first solution for the Summer Analytics Week 2 e-commerce conversion prediction hackathon.

Current best measured public validation result:

- Model: blend of tuned Logistic Regression and TabICL
- Public F1: 0.562281
- Submission: `seb-cheneb/submission.csv`

Notable challengers checked:

- Tuned Logistic Regression: 0.560847 public F1
- TabICL: 0.557732 public F1
- AutoGluon: 0.554609 public F1
- TabPFN full train, 2 estimators CPU: 0.554327 public F1
- XGBoost: 0.548318 public F1
- LightGBM: 0.547009 public F1

Run from `seb-cheneb`:

```powershell
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe run_advanced_experiments.py
.\.venv\Scripts\python.exe run_blend_experiments.py
```
