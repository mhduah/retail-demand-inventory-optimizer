# Retail Demand Forecasting and Inventory Optimisation

## Project Overview

This project develops an end-to-end decision-support system for retail demand forecasting and inventory replenishment.

Historical sales, calendar and pricing data are used to forecast future product demand. The forecasts are then incorporated into an optimisation model that recommends replenishment quantities subject to operational constraints such as purchasing budget, warehouse capacity and target service levels.

## Business Question

How much of each product should a retailer order for each store over the next 28 days while reducing stockouts, excess inventory and total operating cost?

## Initial Scope

* Dataset: M5 retail forecasting dataset
* Product group: FOODS_3
* Stores: CA_1, TX_1 and WI_1
* Forecast horizon: 28 days
* Initial forecasting model: seasonal naïve
* Advanced model: gradient-boosted decision trees
* Inventory model: constrained replenishment optimisation
* Reporting: Python, SQL and an interactive dashboard

## Project Workflow

1. Data acquisition and validation
2. Exploratory data analysis
3. SQL data modelling
4. Baseline forecasting
5. Machine-learning forecasting
6. Inventory optimisation
7. Scenario analysis
8. Dashboard development
9. Business impact evaluation

## Planned Evaluation

Forecasts will be evaluated using time-based validation and metrics such as MAE, RMSE and forecast bias.

Inventory recommendations will be evaluated using estimated stockout rate, service level, holding cost, shortage cost and total operating cost.

## Repository Structure

* `data/` — raw, interim and processed datasets
* `notebooks/` — exploration and model-development notebooks
* `sql/` — database definitions and analytical queries
* `src/` — reusable Python modules
* `tests/` — automated validation tests
* `dashboard/` — dashboard files and application code
* `reports/` — figures, summaries and business reports
