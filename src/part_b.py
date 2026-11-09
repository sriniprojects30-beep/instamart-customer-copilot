"""
src/part_b.py -- Part B: Customer Segments and Lapse Prediction
Computes RFM metrics, performs K-Means clustering & PCA, profiles customer segments,
defines the churn/lapse prediction problem strictly without data leakage, tunes
Logistic Regression and Random Forest models, performs class-balancing experiments,
persists the selected model, and exports the illustrative retention list.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def build_rfm_table():
    """
    Builds RFM table as of snapshot 2025-10-01 00:00:00 for completed orders.
    """
    orders = pd.read_csv(os.path.join("data", "clean", "orders.csv"))
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    completed = orders[orders["status"] == "completed"].copy()

    snapshot = pd.to_datetime("2025-10-01 00:00:00")

    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

    return rfm


def cluster_customers(rfm_df: pd.DataFrame):
    """
    Standardises RFM columns, evaluates K=2..8, clusters with chosen K=4,
    profiles segments, applies PCA, and writes outputs/customer_segments.csv.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_df[["recency", "frequency", "monetary"]])

    kmeans_records = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        kmeans_records.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(sil)
        })

    # Chosen K=4
    km4 = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm_df["segment_id"] = km4.fit_predict(X_scaled)

    # Determine highest monetary cluster dynamically
    cluster_means = rfm_df.groupby("segment_id").agg(
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
        size=("customer_id", "count")
    ).reset_index()

    premium_cluster_id = int(cluster_means.sort_values("monetary_mean", ascending=False).iloc[0]["segment_id"])

    # Segment naming based on profile characteristics
    segment_names = {
        0: "Occasional Small-Basket",
        1: "High-Value Shoppers",
        2: "Loyal Daily Shoppers",
        3: "Dormant / At-Risk"
    }

    rfm_df["segment_name"] = rfm_df["segment_id"].map(segment_names)
    rfm_df["is_premium"] = rfm_df["segment_id"] == premium_cluster_id

    # PCA
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    pca_ratios = pca.explained_variance_ratio_

    segments_out_path = os.path.join(OUTPUTS_DIR, "customer_segments.csv")
    rfm_df[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
        segments_out_path, index=False
    )

    return kmeans_records, cluster_means, pca_ratios, rfm_df


def build_lapse_dataset():
    """
    Builds the leakage-free lapse cohort and feature table using cutoff 2025-09-01.
    """
    orders = pd.read_csv(os.path.join("data", "clean", "orders.csv"))
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers = pd.read_csv(os.path.join("data", "clean", "customers.csv"))
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    products = pd.read_csv(os.path.join("data", "clean", "products.csv"))

    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    end_window = pd.to_datetime("2025-09-30 23:59:59")

    # Only data strictly before cutoff
    pre_cutoff_completed = orders[(orders["order_ts"] < cutoff) & (orders["status"] == "completed")].copy()
    pre_cutoff_all = orders[orders["order_ts"] < cutoff].copy()

    cohort_cids = pre_cutoff_completed["customer_id"].unique()

    # Active in September 2025
    sept_completed = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] <= end_window) & (orders["status"] == "completed")]
    sept_active_cids = set(sept_completed["customer_id"].unique())

    features = []
    for cid in cohort_cids:
        c_orders_comp = pre_cutoff_completed[pre_cutoff_completed["customer_id"] == cid]
        c_orders_all = pre_cutoff_all[pre_cutoff_all["customer_id"] == cid]
        c_info = customers[customers["customer_id"] == cid].iloc[0]

        last_ts = c_orders_comp["order_ts"].max()
        recency = (cutoff - last_ts).days
        frequency = len(c_orders_comp)
        monetary = c_orders_comp["order_amount"].sum()
        avg_discount = c_orders_comp["discount_pct"].mean()

        n_cancelled = (c_orders_all["status"] == "cancelled").sum()
        n_all = len(c_orders_all)
        canc_rate = n_cancelled / n_all if n_all > 0 else 0.0

        avg_deliv = c_orders_comp["delivery_minutes"].mean()
        fav_cat = c_orders_comp["category"].mode()[0]
        tenure_days = (cutoff - c_info["signup_date"]).days

        lapsed = 1 if cid not in sept_active_cids else 0

        features.append({
            "customer_id": cid,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_discount_pct": avg_discount,
            "cancellation_rate": canc_rate,
            "avg_delivery_minutes": avg_deliv,
            "favourite_category": fav_cat,
            "city": c_info["city"],
            "age": c_info["age"],
            "gender": c_info["gender"],
            "tenure_days": tenure_days,
            "lapsed": lapsed
        })

    df_feat = pd.DataFrame(features)
    return df_feat


def train_and_evaluate_models(df_feat: pd.DataFrame, rfm_df: pd.DataFrame):
    """
    Trains and tunes Logistic Regression and Random Forest models with GridSearchCV,
    performs class-balance experiment, saves best model, and outputs retention list.
    """
    X = df_feat.drop(columns=["customer_id", "lapsed"])
    y = df_feat["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Baseline evaluation
    y_pred_base = np.zeros_like(y_test)
    base_acc = float(accuracy_score(y_test, y_pred_base))
    base_f1 = float(f1_score(y_test, y_pred_base, zero_division=0))

    # Preprocessing pipeline
    num_cols = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
    cat_cols = ["favourite_category", "city", "gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
        ]
    )

    # Learned mean vs full cohort mean
    scaler_alone = StandardScaler().fit(X_train[["frequency"]])
    train_freq_mean = float(scaler_alone.mean_[0])
    cohort_freq_mean = float(X["frequency"].mean())

    # 1. Logistic Regression Tuning
    pipe_lr = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    grid_lr = GridSearchCV(
        pipe_lr,
        param_grid={"clf__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1"
    )
    grid_lr.fit(X_train, y_train)

    # 2. Random Forest Tuning
    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    grid_rf = GridSearchCV(
        pipe_rf,
        param_grid={
            "clf__max_depth": [3, 5, 8],
            "clf__min_samples_leaf": [5, 10, 20]
        },
        cv=5,
        scoring="f1"
    )
    grid_rf.fit(X_train, y_train)

    # Test evaluations
    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    lr_metrics = {
        "best_params": grid_lr.best_params_,
        "best_cv_f1": float(grid_lr.best_score_),
        "test_cm": confusion_matrix(y_test, y_pred_lr).tolist(),
        "test_precision": float(precision_score(y_test, y_pred_lr)),
        "test_recall": float(recall_score(y_test, y_pred_lr)),
        "test_f1": float(f1_score(y_test, y_pred_lr))
    }

    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    rf_metrics = {
        "best_params": grid_rf.best_params_,
        "best_cv_f1": float(grid_rf.best_score_),
        "test_cm": confusion_matrix(y_test, y_pred_rf).tolist(),
        "test_precision": float(precision_score(y_test, y_pred_rf)),
        "test_recall": float(recall_score(y_test, y_pred_rf)),
        "test_f1": float(f1_score(y_test, y_pred_rf))
    }

    # Task 7: Class-balance experiment
    train_df = X_train.copy()
    train_df["lapsed"] = y_train
    majority = train_df[train_df["lapsed"] == 0]
    minority = train_df[train_df["lapsed"] == 1]

    minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)
    balanced_train = pd.concat([majority, minority_upsampled])
    X_train_bal = balanced_train.drop(columns=["lapsed"])
    y_train_bal = balanced_train["lapsed"]

    pipe_lr_bal = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, C=grid_lr.best_params_["clf__C"], random_state=42))
    ])
    pipe_lr_bal.fit(X_train_bal, y_train_bal)
    y_pred_lr_bal = pipe_lr_bal.predict(X_test)
    bal_metrics = {
        "train_majority": len(majority),
        "train_minority": len(minority),
        "balanced_total": len(balanced_train),
        "test_precision": float(precision_score(y_test, y_pred_lr_bal)),
        "test_recall": float(recall_score(y_test, y_pred_lr_bal)),
        "test_f1": float(f1_score(y_test, y_pred_lr_bal))
    }

    # Interpretability
    rf_clf = best_rf.named_steps["clf"]
    prep = best_rf.named_steps["prep"]
    cat_feature_names = prep.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names

    importances = rf_clf.feature_importances_
    top_5_idx = np.argsort(importances)[::-1][:5]
    top_5_features = [(all_feature_names[i], float(importances[i])) for i in top_5_idx]

    lr_clf = best_lr.named_steps["clf"]
    lr_coefs = lr_clf.coef_[0]
    sample_coefs = [
        ("recency", float(lr_coefs[all_feature_names.index("recency")])),
        ("frequency", float(lr_coefs[all_feature_names.index("frequency")])),
        ("monetary", float(lr_coefs[all_feature_names.index("monetary")]))
    ]

    # Select Random Forest as final production model
    selected_model = best_rf
    model_path = os.path.join(OUTPUTS_DIR, "lapse_model.joblib")
    joblib.dump(selected_model, model_path)

    # Verify reload exact reproduction
    reloaded_model = joblib.load(model_path)
    reloaded_preds = reloaded_model.predict(X_test)
    assert np.array_equal(y_pred_rf, reloaded_preds), "Reloaded model failed to reproduce test predictions exactly."

    # Task 8(c): Retention List for Test-set Premium customers
    test_indices = X_test.index
    test_cids = df_feat.loc[test_indices, "customer_id"].values
    test_lapsed = df_feat.loc[test_indices, "lapsed"].values
    test_probs = selected_model.predict_proba(X_test)[:, 1]

    test_eval_df = pd.DataFrame({
        "customer_id": test_cids,
        "lapse_probability": test_probs,
        "actually_lapsed": test_lapsed
    })

    # Join with Premium flag from rfm_df
    prem_cids = set(rfm_df[rfm_df["is_premium"]]["customer_id"])
    test_prem_df = test_eval_df[test_eval_df["customer_id"].isin(prem_cids)].copy()
    retention_list = test_prem_df.sort_values("lapse_probability", ascending=False).head(10)

    retention_path = os.path.join(OUTPUTS_DIR, "retention_list.csv")
    retention_list.to_csv(retention_path, index=False)

    return {
        "base_acc": base_acc,
        "base_f1": base_f1,
        "train_freq_mean": train_freq_mean,
        "cohort_freq_mean": cohort_freq_mean,
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "bal_metrics": bal_metrics,
        "top_5_features": top_5_features,
        "sample_coefs": sample_coefs,
        "retention_list": retention_list
    }


def main():
    print("=== PART B: CUSTOMER SEGMENTS AND LAPSE PREDICTION ===")
    rfm_df = build_rfm_table()
    print(f"RFM table computed for {len(rfm_df)} customers.")

    kmeans_records, cluster_means, pca_ratios, rfm_df = cluster_customers(rfm_df)
    print("K-Means clustering completed. Outputs saved to outputs/customer_segments.csv.")

    df_feat = build_lapse_dataset()
    print(f"Lapse dataset created: {len(df_feat)} cohort rows, {df_feat['lapsed'].sum()} lapsed.")

    res = train_and_evaluate_models(df_feat, rfm_df)
    print(f"Models trained and tuned. Random Forest saved to outputs/lapse_model.joblib.")
    print(f"Top 10 Premium retention list saved to outputs/retention_list.csv.")
    print(f"Lapsed among top retention list: {res['retention_list']['actually_lapsed'].sum()} / {len(res['retention_list'])}")


if __name__ == "__main__":
    main()
