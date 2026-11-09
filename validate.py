"""
validate.py -- Asserts reference checkpoints for Part A cleaned data and analytical queries.
Prints exactly 'ALL CHECKPOINTS PASSED' when all hold.
"""
import os
import sqlite3
import pandas as pd
import numpy as np

def validate_all():
    customers_path = os.path.join("data", "clean", "customers.csv")
    orders_path = os.path.join("data", "clean", "orders.csv")
    products_path = os.path.join("data", "clean", "products.csv")
    db_path = os.path.join("data", "instamart.db")

    assert os.path.exists(customers_path), f"Missing {customers_path}"
    assert os.path.exists(orders_path), f"Missing {orders_path}"
    assert os.path.exists(products_path), f"Missing {products_path}"
    assert os.path.exists(db_path), f"Missing {db_path}"

    customers = pd.read_csv(customers_path)
    orders = pd.read_csv(orders_path)
    products = pd.read_csv(products_path)

    # Checkpoint 1: City distribution
    city_counts = customers["city"].value_counts().to_dict()
    expected_cities = {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197,
        "Chennai": 191, "Hyderabad": 191, "Mumbai": 184
    }
    assert city_counts == expected_cities, f"City counts mismatch: {city_counts} != {expected_cities}"

    # Checkpoint 2: Gender distribution
    gender_counts = customers["gender"].value_counts().to_dict()
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    assert gender_counts == expected_genders, f"Gender counts mismatch: {gender_counts} != {expected_genders}"

    # Checkpoint 3: Missing ages
    assert customers["age"].isna().sum() == 0, "Missing ages found in clean customers"

    # Checkpoint 4: Age range 18 to 60
    assert customers["age"].min() >= 18 and customers["age"].max() <= 60, f"Ages outside 18-60: min={customers['age'].min()}, max={customers['age'].max()}"

    # Checkpoint 5: Orders row count exactly 12309
    assert len(orders) == 12309, f"Orders count mismatch: {len(orders)} != 12309"
    assert orders["order_id"].nunique() == 12309, "Duplicate order_id found in clean orders"

    # Checkpoint 6: Order status counts
    status_counts = orders["status"].value_counts().to_dict()
    assert status_counts.get("completed", 0) == 11110, f"Completed orders mismatch: {status_counts.get('completed')}"
    assert status_counts.get("cancelled", 0) == 1199, f"Cancelled orders mismatch: {status_counts.get('cancelled')}"

    # Checkpoint 7: No pending rows
    assert "pending" not in status_counts, "Pending orders found in clean data"

    # Checkpoint 8: 0 missing order_amount
    assert orders["order_amount"].isna().sum() == 0, "Missing order_amount found"

    # Checkpoint 9: 0 missing delivery_minutes for completed, 1199 for cancelled
    completed_orders = orders[orders["status"] == "completed"]
    cancelled_orders = orders[orders["status"] == "cancelled"]
    assert completed_orders["delivery_minutes"].isna().sum() == 0, "Missing delivery_minutes in completed orders"
    assert cancelled_orders["delivery_minutes"].isna().sum() == 1199, "Non-missing delivery_minutes in cancelled orders"

    # Checkpoint 10: Sum of completed order_amount is INR 2,998,668.80
    completed_rev = round(float(completed_orders["order_amount"].sum()), 2)
    assert abs(completed_rev - 2998668.80) < 0.01, f"Revenue sum mismatch: {completed_rev} != 2998668.80"

    # Checkpoint 11: SQLite Database Tables and Content
    conn = sqlite3.connect(db_path)
    db_tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"].tolist()
    assert "customers" in db_tables and "products" in db_tables and "orders" in db_tables, f"Tables missing in DB: {db_tables}"
    db_rev = pd.read_sql_query("SELECT ROUND(SUM(order_amount), 2) AS rev FROM orders WHERE status='completed';", conn)["rev"].iloc[0]
    conn.close()
    assert abs(db_rev - 2998668.80) < 0.01, f"DB revenue mismatch: {db_rev} != 2998668.80"

    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate_all()
