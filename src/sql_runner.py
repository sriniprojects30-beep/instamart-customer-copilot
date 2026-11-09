"""
src/sql_runner.py -- Runs analytical SQL queries against data/instamart.db and performs pandas EDA cross-checks.
"""
import os
import re
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.path.join("data", "instamart.db")
SQL_FILE = os.path.join("sql", "queries.sql")


def load_queries():
    """Parses sql/queries.sql into individual queries by @query_id tag."""
    with open(SQL_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = text.split("-- @query_id:")
    queries = {}
    for block in blocks[1:]:
        lines = block.strip().split("\n")
        qid = lines[0].strip()
        sql = "\n".join(lines[1:]).strip()
        queries[qid] = sql
    return queries


def run_sql_queries():
    conn = sqlite3.connect(DB_PATH)
    queries = load_queries()
    results = {}

    print("=== EXECUTING SQL QUERIES ===")
    for qid, query in queries.items():
        df = pd.read_sql_query(query, conn)
        results[qid] = df
        print(f"\n--- {qid} ---")
        if qid == "Q1":
            print(f"Qualifying customers count: {len(df)} (Reference: 30)")
        elif qid == "Q2":
            print(f"Rows: {len(df)} (Reference: 30)")
            print(f"Total Revenue: INR {df['revenue'].sum():,.2f} (Reference: INR 2,998,668.80)")
            top_cell = df.sort_values("revenue", ascending=False).iloc[0]
            print(f"Top cell: {top_cell['city']} / {top_cell['category']} = INR {top_cell['revenue']:,.2f}")
        elif qid == "Q3":
            print(f"Products count: {len(df)} (Reference: 11)")
        elif qid == "Q4":
            print(f"Customers with 0 completed orders: {len(df)} (Reference: 25)")
        elif qid == "Q5":
            print(f"Customers above average spend: {len(df)} (Reference: 550)")
            print(f"Benchmark average: INR {df['benchmark_avg'].iloc[0]:,.2f} (Reference: INR 2,552.06)")
        elif qid == "Q6":
            top_per_city = df[df["spend_rank"] == 1].sort_values("city")
            print("Top customer per city:")
            for _, r in top_per_city.iterrows():
                print(f"  {r['city']}: {r['customer_id']} (INR {r['total_spend']:,.2f})")
            diff_count = (df["spend_rank"] != df["spend_row_number"]).sum()
            print(f"Rows where RANK() != ROW_NUMBER(): {diff_count} (Reference: 8)")
        elif qid == "Q7":
            print(f"Daily rows: {len(df)} (Reference: 183)")
            print(f"Final cumulative revenue: INR {df['cumulative_revenue'].iloc[-1]:,.2f} (Reference: INR 2,998,668.80)")

    conn.close()
    return results


def run_pandas_crosscheck():
    """Performs pandas cross check and exploratory statistical analysis."""
    orders = pd.read_csv(os.path.join("data", "clean", "orders.csv"))
    customers = pd.read_csv(os.path.join("data", "clean", "customers.csv"))
    products = pd.read_csv(os.path.join("data", "clean", "products.csv"))

    # Join tables
    completed_orders = orders[orders["status"] == "completed"].copy()
    merged = completed_orders.merge(customers, on="customer_id").merge(products, on="product_id")

    # (a) Pivot table cross-check for Q2
    pivot = pd.pivot_table(
        merged,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total"
    )
    grand_total = pivot.loc["Total", "Total"]
    print(f"\n--- PANDAS PIVOT CROSS-CHECK ---")
    print(f"Pivot Grand Total: INR {grand_total:,.2f} (Matches SQL: {grand_total == 2998668.80})")

    # Groupby with named aggregation per city
    city_agg = completed_orders.merge(customers, on="customer_id").groupby("city").agg(
        total_sum=("order_amount", "sum"),
        mean_amount=("order_amount", "mean"),
        order_count=("order_amount", "count")
    ).reset_index()

    print("\nCity Aggregations (Completed Orders):")
    for _, row in city_agg.iterrows():
        print(f"  {row['city']}: Sum = INR {row['total_sum']:,.2f}, Mean = INR {row['mean_amount']:.2f}, Count = {row['order_count']}")

    # (b) Statistical distribution of completed order_amount
    amounts = completed_orders["order_amount"]
    mean_val = amounts.mean()
    median_val = amounts.median()
    skew_val = amounts.skew()
    q1 = amounts.quantile(0.25)
    q3 = amounts.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_below = int((amounts < lower_fence).sum())
    outliers_above = int((amounts > upper_fence).sum())

    print("\n--- COMPLETED ORDER AMOUNT EDA ---")
    print(f"Mean: INR {mean_val:.2f}")
    print(f"Median: INR {median_val:.2f}")
    print(f"Skewness: {skew_val:.2f} (Right-skewed)")
    print(f"Q1: INR {q1:.2f}, Q3: INR {q3:.2f}, IQR: INR {iqr:.2f}")
    print(f"Lower Fence: INR {lower_fence:.2f} (Outliers below: {outliers_below})")
    print(f"Upper Fence: INR {upper_fence:.2f} (Outliers above: {outliers_above})")

    # Correlation matrix
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_mat = completed_orders[corr_cols].corr()
    print("\nCorrelation Matrix:")
    print(corr_mat.round(4))

    return pivot, city_agg, corr_mat


if __name__ == "__main__":
    run_sql_queries()
    run_pandas_crosscheck()
