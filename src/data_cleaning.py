"""
src/data_cleaning.py -- Part A: Clean, store, and validate raw Instamart datasets.
"""
import os
import sqlite3
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "instamart.db")


def audit_raw_data():
    """Performs raw data audit and returns audit dictionary."""
    os.makedirs(RAW_DIR, exist_ok=True)
    customers_raw = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    products_raw = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
    orders_raw = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
    tickets_raw = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"), keep_default_na=False)

    # Auditing orders
    raw_orders_rows = len(orders_raw)
    exact_duplicates = int(orders_raw.duplicated().sum())
    pending_rows = int((orders_raw["status"] == "pending").sum())
    distinct_order_ids = int(orders_raw["order_id"].nunique())
    non_numeric_amounts = int(pd.to_numeric(orders_raw["order_amount"], errors="coerce").isna().sum())
    deliv_numeric = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
    negative_delivery_times = int((deliv_numeric < 0).sum())
    blank_delivery_times = int(orders_raw["delivery_minutes"].isna().sum())

    # Auditing customers
    age_numeric = pd.to_numeric(customers_raw["age"], errors="coerce")
    blank_ages = int(customers_raw["age"].isna().sum())
    ages_outside_range = int(((age_numeric < 10) | (age_numeric > 100)).sum())
    distinct_city_spellings = int(customers_raw["city"].nunique())
    non_blank_gender = customers_raw["gender"].dropna()
    non_blank_gender = non_blank_gender[non_blank_gender.str.strip() != ""]
    distinct_gender_spellings = int(non_blank_gender.nunique())
    blank_genders = int((customers_raw["gender"].isna() | (customers_raw["gender"].str.strip() == "")).sum())

    audit_stats = {
        "orders_shape": orders_raw.shape,
        "customers_shape": customers_raw.shape,
        "products_shape": products_raw.shape,
        "tickets_shape": tickets_raw.shape,
        "raw_orders_rows": raw_orders_rows,
        "exact_duplicates": exact_duplicates,
        "pending_rows": pending_rows,
        "distinct_order_ids": distinct_order_ids,
        "non_numeric_amounts": non_numeric_amounts,
        "negative_delivery_times": negative_delivery_times,
        "blank_delivery_times": blank_delivery_times,
        "blank_ages": blank_ages,
        "ages_outside_range": ages_outside_range,
        "distinct_city_spellings": distinct_city_spellings,
        "distinct_gender_spellings": distinct_gender_spellings,
        "blank_genders": blank_genders,
    }
    return audit_stats, customers_raw, products_raw, orders_raw, tickets_raw


def clean_customers(customers_raw: pd.DataFrame) -> pd.DataFrame:
    """Standardizes city, gender, age, and signup_date."""
    cust = customers_raw.copy()
    # (a) Standardize city: remove leading/trailing spaces, title case
    cust["city"] = cust["city"].astype(str).str.strip().str.title()

    # (b) Standardize gender: male, female, unknown
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

    # (c) Treat ages outside 10-100 as missing, fill missing with median of customer's city
    age_num = pd.to_numeric(cust["age"], errors="coerce")
    age_num.loc[(age_num < 10) | (age_num > 100)] = np.nan
    cust["age"] = age_num
    cust["age"] = cust.groupby("city")["age"].transform(lambda s: s.fillna(s.median())).astype(int)

    # (d) Convert signup_date to datetime
    cust["signup_date"] = pd.to_datetime(cust["signup_date"]).dt.strftime("%Y-%m-%d")

    return cust


def clean_orders(orders_raw: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans orders strictly following the 5 documented steps:
    1. Convert order_ts to datetime
    2. Remove exact duplicates
    3. Keep only record with latest order_ts per order_id
    4. Numeric order_amount, impute missing via product price formula
    5. Set negative delivery_minutes to missing, impute completed missing with category median, leave cancelled missing
    """
    ord_df = orders_raw.copy()

    # Step 1: Convert order_ts to datetime
    ord_df["order_ts"] = pd.to_datetime(ord_df["order_ts"])

    # Step 2: Remove exact duplicate rows
    ord_df = ord_df.drop_duplicates()

    # Step 3: For each order_id keep only record with latest order_ts
    ord_df = ord_df.sort_values("order_ts").groupby("order_id").last().reset_index()

    # Step 4: Convert order_amount to numeric, impute missing
    ord_df["order_amount"] = pd.to_numeric(ord_df["order_amount"], errors="coerce")
    ord_merged = ord_df.merge(products_df[["product_id", "category", "price_inr"]], on="product_id", how="left")

    missing_amt_mask = ord_merged["order_amount"].isna()
    imputed_amounts = (ord_merged["price_inr"] * ord_merged["quantity"] * (1 - ord_merged["discount_pct"] / 100.0)).round(2)
    ord_merged.loc[missing_amt_mask, "order_amount"] = imputed_amounts.loc[missing_amt_mask]

    # Step 5: Delivery minutes handling
    deliv = pd.to_numeric(ord_merged["delivery_minutes"], errors="coerce")
    deliv.loc[deliv < 0] = np.nan
    ord_merged["delivery_minutes"] = deliv

    completed_mask = ord_merged["status"] == "completed"
    cat_deliv_median = ord_merged[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    ord_merged.loc[completed_mask & ord_merged["delivery_minutes"].isna(), "delivery_minutes"] = cat_deliv_median

    # Convert order_ts back to string format for CSV and SQLite storage
    ord_merged["order_ts"] = ord_merged["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Keep exact original orders schema
    clean_orders_df = ord_merged[[
        "order_id", "customer_id", "product_id", "quantity",
        "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"
    ]]

    return clean_orders_df


def save_clean_data(customers_clean: pd.DataFrame, products_clean: pd.DataFrame, orders_clean: pd.DataFrame):
    """Writes clean tables to CSV and SQLite database."""
    os.makedirs(CLEAN_DIR, exist_ok=True)
    customers_clean.to_csv(os.path.join(CLEAN_DIR, "customers.csv"), index=False)
    products_clean.to_csv(os.path.join(CLEAN_DIR, "products.csv"), index=False)
    orders_clean.to_csv(os.path.join(CLEAN_DIR, "orders.csv"), index=False)

    conn = sqlite3.connect(DB_PATH)
    customers_clean.to_sql("customers", conn, if_exists="replace", index=False)
    products_clean.to_sql("products", conn, if_exists="replace", index=False)
    orders_clean.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()


def run_part_a_pipeline():
    audit_stats, customers_raw, products_raw, orders_raw, tickets_raw = audit_raw_data()
    print("--- RAW DATA AUDIT ---")
    for k, v in audit_stats.items():
        print(f"  {k}: {v}")

    customers_clean = clean_customers(customers_raw)
    products_clean = products_raw.copy()
    orders_clean = clean_orders(orders_raw, products_clean)

    save_clean_data(customers_clean, products_clean, orders_clean)
    print("\n--- CLEANED DATA SUMMARY ---")
    print(f"Customers shape: {customers_clean.shape}")
    print(f"Orders shape: {orders_clean.shape}")
    print(f"Completed orders: {(orders_clean['status'] == 'completed').sum()}")
    print(f"Cancelled orders: {(orders_clean['status'] == 'cancelled').sum()}")
    completed_rev = orders_clean.loc[orders_clean['status'] == 'completed', 'order_amount'].sum()
    print(f"Completed revenue: INR {completed_rev:,.2f}")
    return audit_stats, customers_clean, products_clean, orders_clean


if __name__ == "__main__":
    run_part_a_pipeline()
