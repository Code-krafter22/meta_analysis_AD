# Meta-analysis AD External Validation

This repository contains analysis scripts for evaluating an Alzheimer's disease (AD) gene signature on an external cohort (GSE125583), using both **fixed-signature validation** and **retrained elastic-net benchmarks**.

## Repository contents

- `external_validation.py`  
  Main reproducible Python script for external validation with repeated stratified CV, fold-aggregated out-of-fold predictions, bootstrap confidence intervals, calibration checks, and result export.
- `external_validation.qmd`  
  Quarto notebook that walks through data loading, fixed-weight signature scoring, bootstrap uncertainty estimation, calibration plotting, and a nested-CV elastic-net baseline.
- `Elastic_net_ext-val.py`  
  Standalone script to compare multiple model-evaluation strategies for the 37-gene panel (train/test split, CV, LOOCV, stratified K-fold, full-cohort fit).
