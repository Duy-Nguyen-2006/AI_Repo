## 2026-01-22 - [Vectorized Feature Engineering]
**Learning:** Python loops for feature engineering are slow and error-prone. Vectorizing using `groupby` + `rolling` is cleaner and faster.
**Action:** Always prefer vectorized Pandas operations over `iterrows`. Ensure `joblib` is used consistently for model persistence.
