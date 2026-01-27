# Bolt's Journal

## 2024-05-22 - [Initial Journal Creation]
**Learning:** Always check for the existence of the journal file before reading it to avoid errors.
**Action:** Create the file if it doesn't exist.

## 2024-05-22 - [Broken Model File]
**Learning:** `model.py` was found broken (missing function definition). It seems to have been a copy-paste error. This prevented any optimization verification.
**Action:** Reconstructed `prepare_features` using vectorized operations instead of the original iterative loop, thus fixing the bug and optimizing training performance simultaneously.

## 2024-05-22 - [Vectorized Feature Engineering]
**Learning:** Reconstructing the logic using Pandas vectorized operations (groupby, rolling, shift) is much more efficient than `iterrows`, but requires careful handling of the "long" format and merging back.
**Action:** Always prefer vectorized operations for feature engineering on DataFrames.
