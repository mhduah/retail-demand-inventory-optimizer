-- ============================================================
-- Retail Demand and Inventory Optimisation
-- Initial business analysis queries
-- ============================================================


-- 1. Overall dataset coverage
SELECT
    COUNT(DISTINCT item_id) AS number_of_items,
    COUNT(DISTINCT store_id) AS number_of_stores,
    COUNT(DISTINCT day_id) AS number_of_days,
    COUNT(*) AS number_of_sales_records,
    SUM(units_sold) AS total_units_sold
FROM fact_sales;


-- 2. Sales performance by store
SELECT
    s.store_id,
    st.state_id,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(AVG(s.units_sold), 3) AS average_units_per_product_day,
    SUM(
        CASE
            WHEN s.units_sold = 0 THEN 1
            ELSE 0
        END
    ) AS zero_sales_records,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN s.units_sold = 0 THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS zero_sales_percentage
FROM fact_sales AS s
INNER JOIN dim_stores AS st
    ON s.store_id = st.store_id
GROUP BY
    s.store_id,
    st.state_id
ORDER BY
    total_units_sold DESC;


-- 3. Estimated revenue by store
SELECT
    s.store_id,
    ROUND(
        SUM(s.units_sold * p.sell_price),
        2
    ) AS estimated_revenue,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(
        SUM(s.units_sold * p.sell_price)
        / NULLIF(SUM(s.units_sold), 0),
        2
    ) AS average_revenue_per_unit
FROM fact_sales AS s
INNER JOIN dim_calendar AS c
    ON s.day_id = c.day_id
INNER JOIN fact_prices AS p
    ON s.store_id = p.store_id
    AND s.item_id = p.item_id
    AND c.wm_yr_wk = p.wm_yr_wk
GROUP BY
    s.store_id
ORDER BY
    estimated_revenue DESC;


-- 4. Ten products with the highest total demand
SELECT
    s.item_id,
    i.dept_id,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(AVG(s.units_sold), 3) AS average_daily_units,
    COUNT(DISTINCT s.store_id) AS stores_represented
FROM fact_sales AS s
INNER JOIN dim_items AS i
    ON s.item_id = i.item_id
GROUP BY
    s.item_id,
    i.dept_id
ORDER BY
    total_units_sold DESC
LIMIT 10;


-- 5. Ten products with the highest estimated revenue
SELECT
    s.item_id,
    ROUND(
        SUM(s.units_sold * p.sell_price),
        2
    ) AS estimated_revenue,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(AVG(p.sell_price), 2) AS average_sell_price
FROM fact_sales AS s
INNER JOIN dim_calendar AS c
    ON s.day_id = c.day_id
INNER JOIN fact_prices AS p
    ON s.store_id = p.store_id
    AND s.item_id = p.item_id
    AND c.wm_yr_wk = p.wm_yr_wk
GROUP BY
    s.item_id
ORDER BY
    estimated_revenue DESC
LIMIT 10;


-- 6. Monthly demand trend
SELECT
    c.year,
    c.month,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(AVG(s.units_sold), 3) AS average_units_per_record
FROM fact_sales AS s
INNER JOIN dim_calendar AS c
    ON s.day_id = c.day_id
GROUP BY
    c.year,
    c.month
ORDER BY
    c.year,
    c.month;


-- 7. Demand by weekday
SELECT
    c.wday,
    c.weekday,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(AVG(s.units_sold), 3) AS average_units_per_record
FROM fact_sales AS s
INNER JOIN dim_calendar AS c
    ON s.day_id = c.day_id
GROUP BY
    c.wday,
    c.weekday
ORDER BY
    c.wday;


-- 8. Price-data coverage for positive-sales records
SELECT
    COUNT(*) AS positive_sales_records,
    SUM(
        CASE
            WHEN p.sell_price IS NOT NULL THEN 1
            ELSE 0
        END
    ) AS records_with_price,
    SUM(
        CASE
            WHEN p.sell_price IS NULL THEN 1
            ELSE 0
        END
    ) AS records_without_price,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN p.sell_price IS NOT NULL THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS price_coverage_percentage
FROM fact_sales AS s
INNER JOIN dim_calendar AS c
    ON s.day_id = c.day_id
LEFT JOIN fact_prices AS p
    ON s.store_id = p.store_id
    AND s.item_id = p.item_id
    AND c.wm_yr_wk = p.wm_yr_wk
WHERE
    s.units_sold > 0;