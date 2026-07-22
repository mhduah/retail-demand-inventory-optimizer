# Data

This project uses the M5 Forecasting Accuracy dataset.

## Directory structure

- `raw/` — original downloaded files; never edited manually
- `interim/` — cleaned or partially transformed data
- `processed/` — modelling-ready datasets

The raw and processed datasets are excluded from Git because of their size.

## Required source files

The project initially uses:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`

The M5 dataset contains daily Walmart sales data organised by product, department, category, store and US state.