## 2025-05-15 - Sort Stability in Match Processing
**Learning:** When vectorizing sequential processing of matches (e.g. for rolling stats), sorting by `date` alone is insufficient if multiple matches occur on the same day (or if data has duplicate dates). The original row order (index) must be preserved and used as a secondary sort key to match the iterative processing order exactly.
**Action:** When converting iterative loops to vectorized operations on time-series data, always `reset_index` and include the index in the sort columns: `df.sort_values(['date', 'index'])`.
