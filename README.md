# Datathon 2026

## Trang doc nhanh

Neu can hieu repo nhanh, doc:

- `docs/CODEBASE_OVERVIEW.md`
- `docs/FORECASTING_PIPELINE.md`
- `docs/REPO_HANDOFF.md`

## Entry point hien tai

Pipeline forecasting nen chay hien tai:

- `notebooks/06_forecasting_v2_recursive.ipynb`
- `scripts/run_forecasting_v2.py`

Notebook forecasting cu nam o:

- `notebooks/05_forecasting_model.ipynb`

Script/notebook import logic tu:

- `src/utils/feature_engineering.py`
- `src/utils/models.py`
- `src/utils/utils.py`
- `src/utils/forecasting_v2.py`

## Data path thuc te

Notebook modeling hien dang doc CSV tu:

- `dataset/`

Va xuat submission tai:

- `dataset/submission.csv`
- `dataset/submission_v2.csv`

## Cach chay v2

Trong notebook:

- mo `notebooks/06_forecasting_v2_recursive.ipynb`
- Run All

Bang script:

```bash
python scripts/run_forecasting_v2.py
```

## Ghi chu

- Repo hien van theo huong `notebook-first`.
- `scripts/run_forecasting_v2.py` la pipeline sach hon cho backtest va submission.
- README cu da khong con khop hoan toan voi codebase, nen cac file trong `docs/` moi la mo ta thuc te hon.
