## 2026-01-24 - Vectorization over Iteration
**Learning:** The codebase relied on `iterrows()` for rolling feature calculation, which is significantly slower than vectorized `groupby().rolling()` operations.
**Action:** Always check for `iterrows` loops in feature engineering pipelines and replace with vectorized Pandas operations.
