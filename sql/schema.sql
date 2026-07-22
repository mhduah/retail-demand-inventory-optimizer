PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS fact_prices;
DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_calendar;
DROP TABLE IF EXISTS dim_stores;
DROP TABLE IF EXISTS dim_items;

CREATE TABLE dim_items (
    item_id TEXT PRIMARY KEY,
    dept_id TEXT NOT NULL,
    cat_id TEXT NOT NULL
);

CREATE TABLE dim_stores (
    store_id TEXT PRIMARY KEY,
    state_id TEXT NOT NULL
);

CREATE TABLE dim_calendar (
    day_id TEXT PRIMARY KEY,
    date TEXT NOT NULL UNIQUE,
    wm_yr_wk INTEGER NOT NULL,
    weekday TEXT NOT NULL,
    wday INTEGER NOT NULL,
    month INTEGER NOT NULL,
    year INTEGER NOT NULL,
    event_name_1 TEXT,
    event_type_1 TEXT,
    event_name_2 TEXT,
    event_type_2 TEXT,
    snap_CA INTEGER NOT NULL CHECK (snap_CA IN (0, 1)),
    snap_TX INTEGER NOT NULL CHECK (snap_TX IN (0, 1)),
    snap_WI INTEGER NOT NULL CHECK (snap_WI IN (0, 1))
);

CREATE TABLE fact_sales (
    item_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    day_id TEXT NOT NULL,
    units_sold INTEGER NOT NULL CHECK (units_sold >= 0),

    PRIMARY KEY (item_id, store_id, day_id),

    FOREIGN KEY (item_id)
        REFERENCES dim_items (item_id),

    FOREIGN KEY (store_id)
        REFERENCES dim_stores (store_id),

    FOREIGN KEY (day_id)
        REFERENCES dim_calendar (day_id)
);

CREATE TABLE fact_prices (
    store_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    wm_yr_wk INTEGER NOT NULL,
    sell_price REAL NOT NULL CHECK (sell_price > 0),

    PRIMARY KEY (store_id, item_id, wm_yr_wk),

    FOREIGN KEY (store_id)
        REFERENCES dim_stores (store_id),

    FOREIGN KEY (item_id)
        REFERENCES dim_items (item_id)
);

CREATE INDEX idx_sales_day
    ON fact_sales (day_id);

CREATE INDEX idx_sales_store_day
    ON fact_sales (store_id, day_id);

CREATE INDEX idx_sales_item_day
    ON fact_sales (item_id, day_id);

CREATE INDEX idx_prices_week
    ON fact_prices (wm_yr_wk);