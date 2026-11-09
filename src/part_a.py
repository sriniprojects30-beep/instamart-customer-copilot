"""
src/part_a.py -- Part A: Data Foundation: Clean, Store, Query
Audits dirty raw data, cleans customers and orders with documented rules,
persists clean tables to CSV and SQLite, runs seven SQL analytical queries,
and performs Pandas EDA cross-checks.
"""
import os
import re
import sqlite3
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "instamart.db")
SQL_PATH = os.path.join("sql", "queries.sql")


def audit_raw_data():
    """
    Loads raw tables and calculates audit metrics matching Part A task 2.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    customers_raw = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    products_raw = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
    orders_raw = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
    tickets_raw = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"), keep_default_na=False)

    audit = {}
    audit["raw_orders_rows"] = len(orders_raw)
    audit["exact_duplicates"] = int(orders_raw.duplicated().sum())
    audit["pending_rows"] = int((orders_raw["status"] == "pending").sum())
    audit["distinct_order_ids"] = int(orders_raw["order_id"].nunique())
    audit["non_numeric_amounts"] = int(pd.to_numeric(orders_raw["order_amount"], errors="coerce").isna().sum())
    
    deliv_numeric = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
    audit["negative_delivery_times"] = int((deliv_numeric < 0).sum())
    audit["blank_delivery_times"] = int(orders_raw["delivery_minutes"].isna().sum())

    age_numeric = pd.to_numeric(customers_raw["age"], errors="coerce")
    audit["blank_ages"] = int(customers_raw["age"].isna().sum())
    audit["ages_outside_range"] = int(((age_numeric < 10) | (age_numeric > 100)).sum())
    audit["distinct_city_spellings"] = int(customers_raw["city"].nunique())
    
    non_blank_gender = customers_raw["gender"].dropna()
    non_blank_gender = non_blank_gender[non_blank_gender.astype(str).str.strip() != ""]
    audit["distinct_gender_spellings"] = int(non_blank_gender.nunique())
    audit["blank_genders"] = int((customers_raw["gender"].isna() | (customers_raw["gender"].astype(str).str.strip() == "")).sum())

    return audit, customers_raw, products_raw, orders_raw, tickets_raw


def clean_customers(customers_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans customer table:
    (a) Standardise city: remove leading/trailing spaces and apply title case (exactly 6 cities).
    (b) Map every gender spelling to male, female, or unknown.
    (c) Treat ages outside 10-100 as missing, fill missing with median of customer's city.
    (d) Convert signup_date to datetime format.
    """
    cust = customers_raw.copy()
    cust["city"] = cust["city"].astype(str).str.strip().str.title()

    def map_gender(g):
        if pd.isna(g) or str(g).strip() == "":
            return "unknown"
        s = str(g).strip().lower()
        if s in ["m", "male"]:
            return "male"
        elif s in ["f", "female"]:
            return "female"
        return "unknown"

    cust["gender"] = cust["gender"].apply(map_gender)

    age_num = pd.to_numeric(cust["age"], errors="coerce")
    age_num.loc[(age_num < 10) | (age_num > 100)] = np.nan
    cust["age"] = age_num
    cust["age"] = cust.groupby("city")["age"].transform(lambda s: s.fillna(s.median())).astype(int)
    cust["signup_date"] = pd.to_datetime(cust["signup_date"]).dt.strftime("%Y-%m-%d")

    return cust


def clean_orders(orders_raw: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans orders strictly following the 5 documented steps:
    1. Convert order_ts to datetime.
    2. Remove exact duplicate rows.
    3. Keep only the record with latest order_ts per order_id.
    4. Convert order_amount to numeric, impute missing via product price formula.
    5. Set negative delivery_minutes to missing, impute completed missing with category median, leave cancelled missing.
    """
    ord_df = orders_raw.copy()

    # Step 1: Convert order_ts to datetime
    ord_df["order_ts"] = pd.to_datetime(ord_df["order_ts"])

    # Step 2: Remove exact duplicate rows
    ord_df = ord_df.drop_duplicates()

    # Step 3: Keep record with latest order_ts per order_id
    ord_df = ord_df.sort_values("order_ts").groupby("order_id").last().reset_index()

    # Step 4: Convert order_amount to numeric and impute missing
    ord_df["order_amount"] = pd.to_numeric(ord_df["order_amount"], errors="coerce")
    ord_merged = ord_df.merge(products_df[["product_id", "category", "price_inr"]], on="product_id", how="left")

    missing_amt_mask = ord_merged["order_amount"].isna()
    imputed_amounts = (ord_merged["price_inr"] * ord_merged["quantity"] * (1 - ord_merged["discount_pct"] / 100.0)).round(2)
    ord_merged.loc[missing_amt_mask, "order_amount"] = imputed_amounts.loc[missing_amt_mask]

    # Step 5: Set negative delivery_minutes to missing and impute completed only
    deliv = pd.to_numeric(ord_merged["delivery_minutes"], errors="coerce")
    deliv.loc[deliv < 0] = np.nan
    ord_merged["delivery_minutes"] = deliv

    completed_mask = ord_merged["status"] == "completed"
    cat_deliv_median = ord_merged[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    ord_merged.loc[completed_mask & ord_merged["delivery_minutes"].isna(), "delivery_minutes"] = cat_deliv_median

    # Convert order_ts to standard string representation for storage
    ord_merged["order_ts"] = ord_merged["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    clean_orders_df = ord_merged[[
        "order_id", "customer_id", "product_id", "quantity",
        "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"
    ]]

    return clean_orders_df


def save_data(customers_clean: pd.DataFrame, products_clean: pd.DataFrame, orders_clean: pd.DataFrame):
    """Saves clean data to CSV files and SQLite database."""
    os.makedirs(CLEAN_DIR, exist_ok=True)
    customers_clean.to_csv(os.path.join(CLEAN_DIR, "customers.csv"), index=False)
    products_clean.to_csv(os.path.join(CLEAN_DIR, "products.csv"), index=False)
    orders_clean.to_csv(os.path.join(CLEAN_DIR, "orders.csv"), index=False)

    conn = sqlite3.connect(DB_PATH)
    customers_clean.to_sql("customers", conn, if_exists="replace", index=False)
    products_clean.to_sql("products", conn, if_exists="replace", index=False)
    orders_clean.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()


def run_sql_queries():
    """Runs analytical SQL queries from sql/queries.sql against data/instamart.db."""
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = text.split("-- @query_id:")
    queries = {}
    for block in blocks[1:]:
        lines = block.strip().split("\n")
        qid = lines[0].strip()
        sql = "\n".join(lines[1:]).strip()
        queries[qid] = sql

    conn = sqlite3.connect(DB_PATH)
    sql_results = {}
    for qid, sql in queries.items():
        df = pd.read_sql_query(sql, conn)
        sql_results[qid] = df
    conn.close()
    return sql_results


def run_pandas_crosscheck():
    """Performs pandas EDA cross-checks."""
    orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))
    customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
    products = pd.read_csv(os.path.join(CLEAN_DIR, "products.csv"))

    completed = orders[orders["status"] == "completed"].copy()
    merged = completed.merge(customers, on="customer_id").merge(products, on="product_id")

    pivot = pd.pivot_table(
        merged,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total"
    )

    city_agg = completed.merge(customers, on="customer_id").groupby("city").agg(
        total_sum=("order_amount", "sum"),
        mean_amount=("order_amount", "mean"),
        order_count=("order_amount", "count")
    ).reset_index()

    amounts = completed["order_amount"]
    q1 = float(amounts.quantile(0.25))
    q3 = float(amounts.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_above = int((amounts > upper_fence).sum())
    outliers_below = int((amounts < lower_fence).sum())

    stats = {
        "mean": float(amounts.mean()),
        "median": float(amounts.median()),
        "skewness": float(amounts.skew()),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outliers_above": outliers_above,
        "outliers_below": outliers_below
    }

    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_mat = completed[corr_cols].corr()

    return pivot, city_agg, stats, corr_mat


def main():
    print("=== PART A: DATA FOUNDATION PIPELINE ===")
    audit, cust_raw, prod_raw, ord_raw, tick_raw = audit_raw_data()
    print("Audited raw data successfully.")
    
    cust_clean = clean_customers(cust_raw)
    prod_clean = prod_raw.copy()
    ord_clean = clean_orders(ord_raw, prod_clean)
    
    save_data(cust_clean, prod_clean, ord_clean)
    print("Cleaned tables saved to data/clean/ and data/instamart.db.")

    sql_results = run_sql_queries()
    print("Executed all 7 SQL queries from sql/queries.sql.")
    print(f"  Q1 qualifying customers: {len(sql_results['Q1'])}")
    print(f"  Q2 revenue rows: {len(sql_results['Q2'])}, total rev: INR {sql_results['Q2']['revenue'].sum():,.2f}")
    print(f"  Q3 products priced above avg: {len(sql_results['Q3'])}")
    print(f"  Q4 customers with 0 completed orders: {len(sql_results['Q4'])}")
    print(f"  Q5 customers above avg spend: {len(sql_results['Q5'])}")
    print(f"  Q6 rank!=row_number count: {(sql_results['Q6']['spend_rank'] != sql_results['Q6']['spend_row_number']).sum()}")
    print(f"  Q7 cumulative rev rows: {len(sql_results['Q7'])}, final: INR {sql_results['Q7']['cumulative_revenue'].iloc[-1]:,.2f}")

    pivot, city_agg, stats, corr_mat = run_pandas_crosscheck()
    print(f"Pandas Pivot Total: INR {pivot.loc['Total', 'Total']:,.2f}")
    print(f"Completed order_amount Stats: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Skew={stats['skewness']:.2f}")


if __name__ == "__main__":
    main()
