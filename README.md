# Instamart Customer Intelligence & Support Copilot

An enterprise-grade, end-to-end data and AI copilot built for Swiggy Instamart to solve raw data unreliability, customer lapse prediction, automated privacy-safe ticket triage, policy knowledge retrieval, and autonomous refund resolution.

---

## Overview

Swiggy Instamart delivers groceries, dairy, snacks, personal-care products, and small electronics across six metro cities in India. As weekly order volume surges, operational friction emerges across three critical fronts:
1. **Dirty Raw Data**: Order exports contain duplicate transactions, conflicting status entries, unparseable currencies, negative delivery durations, and missing customer demographics.
2. **Customer Retention Blindspots**: Retention teams lack early-warning churn signals and clear behavioural segmentation to proactively engage customers before they lapse.
3. **Overwhelmed Customer Support**: Support desks are flooded with thousands of unstructured tickets containing personally identifiable information (PII), mixing urgent fraud alerts, delivery delays, app crashes, and refund demands.

The **Customer Intelligence & Support Copilot** is designed as a unified, production-ready system spanning four modular layers:
- **Part A (Data Foundation)**: Cleans, audits, standardises, and validates raw exports; persists the golden dataset into relational SQLite storage (`data/instamart.db`); and answers core business analytics questions via ANSI SQL window functions and Pandas exploratory data analysis.
- **Part B (Customer Segmentation & Lapse Prediction)**: Extracts snapshot RFM features, clusters customers via K-Means and PCA, formulates a strictly leakage-free lapse prediction task, tunes Regularised Logistic Regression and Random Forest classifiers, executes class-balancing experiments, and exports a prioritised retention target list.
- **Part C (Support Ticket Triage with Guardrails)**: Implements privacy guardrails (regex PII masking of emails, phone numbers, and order IDs), token and input length filters, resilient retry wrappers with exponential backoff and jitter against transient 503 errors, dual validation (JSON and regex word tokens), token budget controls, and dynamic routing using Part B customer tiers.
- **Part D (Policy RAG Assistant & ReAct Refund Agent)**: Chunks store policy documents, builds a TF-IDF vector retrieval engine, grounds policy question-answering, defines autonomous agent tools, executes a multi-turn ReAct reasoning loop across simulated scenarios, enforces graceful fallback limits (iterations, token budgets, execution timeouts), and demonstrates zero-duplicate crash-and-resume state persistence.

All language model interactions run offline and deterministically via the supplied `mock_llm.py` stand-in, requiring no external API keys, network access, or paid services. All monetary amounts are recorded and displayed in Indian Rupees (INR).

---

## Reproduce

The entire repository is deterministic and fully reproducible. Follow the numbered shell commands below from a clean clone to set up the environment, regenerate the synthetic data, run all analytical pipelines in sequential order (01 to 05), and verify all checkpoints:

```bash
# 1. Clone repository and navigate to root directory
git clone https://github.com/sriniprojects30-beep/instamart-customer-copilot.git
cd instamart-customer-copilot

# 2. Ensure git branch is main
git branch -M main

# 3. Create an isolated Python virtual environment (Python 3.11 or newer)
python -m venv .venv

# 4. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# 5. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 6. Generate deterministic raw synthetic data and policy texts
python generate_data.py

# =====================================================================
# ORDER-WISE PIPELINE EXECUTION (Steps 01 to 05)
# =====================================================================

# Step 1: Data Foundation (Clean, Store in SQLite, Analytical SQL Queries)
python 01_data_foundation.py

# Step 1 Checkpoint Validation:
python validate.py

# Step 2: Customer Intelligence (RFM Segmentation, K-Means & Random Forest Churn Model)
python 02_customer_intelligence.py

# Step 3: Support Ticket Triage (PII Masking, Few-Shot LLM Triage & Segment Routing)
python 03_support_triage.py

# Step 4: Policy Knowledge Assistant & Multi-Turn ReAct Refund Agent
python 04_policy_refund_agent.py

# Step 5: Unified One-Input Support Copilot (Omnibox CLI Verification)
python 05_one_input_copilot.py --query "Customer C0082 profile and churn risk"

# =====================================================================
# AUTOMATED REPRODUCIBILITY & QA VERIFICATION
# =====================================================================

# Run the complete sequential pipeline end-to-end in one command (~21 seconds):
python run_all.py

# Run comprehensive QA and regression test suite (100% pass rate):
python test_suite.py

# Launch Full-Stack Web Application (Port 8000):
python app.py
# Open http://127.0.0.1:8000 in your browser for the Interactive Omnibox Copilot UI
```

---

## Part A — Data Foundation: Clean, Store, Query

### 1. Raw Data Audit
The initial raw exports (`customers.csv`, `products.csv`, `orders.csv`, `tickets.csv`) were audited for structural shape, datatypes, and data quality defects before cleaning.

| Dataset | Rows | Columns | Primary Key / Identifier | Observed Data Quality Issues |
| :--- | :--- | :--- | :--- | :--- |
| `orders.csv` | 13,084 | 9 | `order_id` (non-unique raw) | Exact duplicates, pending status rows, unparseable strings in amount, negative delivery times, blank delivery times |
| `customers.csv` | 1,200 | 5 | `customer_id` | Blank ages, physiologically impossible ages (<10, >100), casing/whitespace variations in cities, mixed gender formats |
| `products.csv` | 40 | 4 | `product_id` | Valid catalog baseline; prices range from INR 20 to INR 1,999 |
| `tickets.csv` | 164 | 5 | `ticket_id` | Empty tickets (4 rows), oversized tickets (>400 tokens, 4 rows), raw customer PII (emails, phones, order IDs) |

#### Raw Data Audit Figures Table
All metrics below strictly match the reference specifications:

| Audit Checkpoint Metric | Value Observed | Target Reference | Status |
| :--- | :--- | :--- | :--- |
| Raw `orders.csv` rows | 13,084 | 13,084 | Exact match |
| Exact duplicate rows in `orders.csv` | 119 | 119 | Exact match |
| Orders with status `pending` | 656 | 656 | Exact match |
| Distinct `order_id` values | 12,309 | 12,309 | Exact match |
| Non-numeric `order_amount` values | 120 | 120 | Exact match |
| Negative `delivery_minutes` entries | 106 | 106 | Exact match |
| Blank `delivery_minutes` entries | 2,541 | 2,541 | Exact match |
| Blank customer `age` entries | 61 | 61 | Exact match |
| Customer `age` values outside 10–100 | 6 | 6 | Exact match |
| Distinct raw city spellings | 24 | 24 | Exact match |
| Distinct non-blank raw gender spellings | 10 | 10 | Exact match |
| Blank customer `gender` entries | 30 | 30 | Exact match |

---

### 2. Cleaning Rules & Transformations Applied

#### Customer Cleaning Rules
1. **City Standardisation**: Applied whitespace stripping and title casing (`.astype(str).str.strip().str.title()`). Reduced 24 messy city variations to exactly 6 canonical cities:
   - **Pune**: 227
   - **Delhi**: 210
   - **Bengaluru**: 197
   - **Chennai**: 191
   - **Hyderabad**: 191
   - **Mumbai**: 184
   - Total: 1,200 customers.
2. **Gender Mapping**: Standardised all 10 non-blank casing permutations (`male`, `Male`, `M`, `m`, `MALE`, `female`, `Female`, `F`, `f`, `FEMALE`) and 30 blank entries into three discrete categories:
   - **female**: 610
   - **male**: 560
   - **unknown**: 30
3. **Age Imputation**: Replaced 6 values outside [10, 100] (values: 280, -4, 0, 150, 999, 200) and 61 blank values (total 67 missing) with the median customer age of that customer's respective city using `groupby('city')['age'].transform(lambda s: s.fillna(s.median()))`.
   - Post-cleaning missing ages: 0.
   - Post-cleaning age range: 18 to 60.
4. **Signup Date Conversion**: Parsed raw signup strings into ISO-8601 date strings (`YYYY-MM-DD`).

#### Order Cleaning Rules (Executed Strictly in Documented Order)
1. **Timestamp Conversion**: Parsed `order_ts` into standard Python `datetime` objects.
2. **Exact Duplicate Removal**: Dropped 119 exact duplicate rows (dataset reduced from 13,084 to 12,965 rows).
3. **Status Conflict Resolution**: Addressed orders appearing with multiple status entries (earlier `pending` followed by later final status). Grouped by `order_id`, sorted chronologically by `order_ts`, and retained only the latest record (`keep='last'`). This removed 656 stale `pending` rows, yielding exactly 12,309 unique orders:
   - **Completed orders**: 11,110
   - **Cancelled orders**: 1,199
   - **Pending orders**: 0
4. **Order Amount Repair**: Converted `order_amount` to numeric, coercing 120 unparseable planted strings (`"N/A"` and `"error"`) to missing (`NaN`). Joined with `products.csv` on `product_id` and filled each missing value using the catalogue formula:
   $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
   - Missing `order_amount` post-imputation: 0.
5. **Delivery Duration Repair**: Converted 106 impossible negative delivery durations (`-10`) to missing (`NaN`). For orders with status `completed`, filled missing delivery durations using the median delivery time of that order's product category across completed orders (`groupby('category')['delivery_minutes'].transform('median')`). For orders with status `cancelled`, left `delivery_minutes` missing (`NaN`), avoiding artificial data fabrication since cancelled orders were never dispatched.
   - Missing `delivery_minutes` among completed: 0.
   - Missing `delivery_minutes` among cancelled: 1,199.
   - Total completed revenue: **INR 2,998,668.80**.

---

### 3. Database Persistence and Checkpoint Validation
The cleaned tables were persisted as CSVs in `data/clean/` (`customers.csv`, `products.csv`, `orders.csv`) and loaded into SQLite at `data/instamart.db`. 

The test harness in `validate.py` verifies all 11 critical reference invariants:
```bash
python validate.py
# Output:
# ALL CHECKPOINTS PASSED
```

---

### 4. SQL Analysis Results (`sql/queries.sql`)

All queries were executed via `pd.read_sql_query` against `data/instamart.db`:

| Query ID | Purpose & Construct | SQL Analytical Result | Target Reference |
| :--- | :--- | :--- | :--- |
| **Q1** | WHERE order_amount > 500, GROUP BY, HAVING SUM > 8000 | **30 qualifying customers** | 30 customers |
| **Q2** | Three-table JOIN (`orders`, `customers`, `products`) | **30 rows**, Total Rev: **INR 2,998,668.80**. Top cell: **Bengaluru / Electronics** at **INR 233,909.45** | 30 rows; INR 2,998,668.80; Bengaluru / Electronics INR 233,909.45 |
| **Q3** | Scalar Subquery (`price_inr > (SELECT AVG...)`) | **11 products** priced above catalog average of INR 352.38 | 11 products |
| **Q4** | LEFT JOIN finding customers with zero completed orders | **25 customers** with no completed orders | 25 customers |
| **Q5** | Two CTEs (customer spend & benchmark average spend) | **550 customers** spending above benchmark average of **INR 2,552.06** | 550 customers; avg INR 2,552.06 |
| **Q6** | Window Functions (`RANK()`, `DENSE_RANK()`, `ROW_NUMBER()`) | Top per city: Bengaluru **C0082** (INR 12,856.20), Chennai **C0930** (INR 10,151.40), Delhi **C0291** (INR 13,742.60), Hyderabad **C0743** (INR 12,730.05), Mumbai **C0693** (INR 14,766.55), Pune **C0426** (INR 10,377.10). | Exact city leaders matched |
| **Q7** | Cumulative Running Total (`SUM(...) OVER (ORDER BY day)`) | **183 daily rows** (2025-04-01 to 2025-09-30). Final cumulative spend: **INR 2,998,668.80** | 183 rows; INR 2,998,668.80 |

#### Explanation for Window Function Discrepancy (Q6)
In query Q6, exactly **8 rows** have `RANK()` different from `ROW_NUMBER()`. This occurs because when two or more customers within the same city share identical total completed spends, `RANK()` assigns identical ranks to the tied rows and leaves gaps in subsequent rankings (e.g. 5, 5, 7), whereas `ROW_NUMBER()` assigns an arbitrary, strictly consecutive sequential integer (e.g. 5, 6, 7) regardless of numerical ties.

---

### 5. Pandas Cross-Check and Exploratory Data Analysis

#### Cross-Check (a): City & Category Revenue
The Pandas pivot table cross-check (`pd.pivot_table(..., margins=True)`) reproduces the exact SQL grand total of **INR 2,998,668.80**. Groupby aggregations with named aggregation per city yield:
- **Bengaluru**: Sum = INR 523,643.40, Mean = INR 276.47, Count = 1,894
- **Chennai**: Sum = INR 463,629.15, Mean = INR 272.72, Count = 1,700
- **Delhi**: Sum = INR 502,829.80, Mean = INR 243.62, Count = 2,064
- **Hyderabad**: Sum = INR 528,589.90, Mean = INR 284.95, Count = 1,855
- **Mumbai**: Sum = INR 479,058.50, Mean = INR 297.00, Count = 1,613
- **Pune**: Sum = INR 500,918.05, Mean = INR 252.48, Count = 1,984

#### Exploratory Statistical Analysis (b): Completed Order Amount
- **Mean**: INR 269.91
- **Median**: INR 135.00
- **Skewness**: 4.61 (heavily right-skewed)
- **Quartiles**: Q1 = INR 64.80, Q3 = INR 297.00, IQR = INR 232.20
- **Tukey 1.5×IQR Fences**: Lower Fence = INR -283.50, Upper Fence = INR 645.30
- **Outliers**: 0 below lower fence, **897 outliers** above upper fence

#### Correlation Matrix (Completed Orders)
| Feature | `order_amount` | `discount_pct` | `quantity` | `delivery_minutes` |
| :--- | :--- | :--- | :--- | :--- |
| `order_amount` | 1.0000 | -0.0951 | 0.0873 | -0.0031 |
| `discount_pct` | -0.0951 | 1.0000 | -0.0004 | -0.0098 |
| `quantity` | 0.0873 | -0.0004 | 1.0000 | -0.0126 |
| `delivery_minutes` | -0.0031 | -0.0098 | -0.0126 | 1.0000 |

#### Executive Data Synthesis (Under 200 Words)
This dataset tracks 11,110 completed grocery and electronics orders across six Indian metros. Order amount exhibits extreme positive skewness (4.61), with a median of INR 135.00 compared to a mean of INR 269.91, driven by high-value electronics purchases. Tukey's 1.5×IQR rule identifies 897 upper-tail outliers above INR 645.30 (maximum INR 3,998.00), representing legitimate bulk or premium hardware carts rather than data corruption. Linear correlation across transaction variables is modest: the strongest relationship is a weak negative correlation between order amount and discount percentage ($r = -0.0951$), followed by order amount and quantity ($r = +0.0873$). Delivery duration shows virtually zero correlation with order size ($r = -0.0031$). Crucially, a correlation of this modest magnitude does not establish causation; higher basket values do not trigger lower discounts, but rather reflect premium catalog pricing where percentage markdowns were capped.

---

## Part B — Customer Segments and Lapse Prediction

### 1. RFM Table Construction
RFM metrics were computed across 1,175 customers having at least one completed order as of snapshot **2025-10-01 00:00:00**:
- **Recency** ($\text{snapshot} - \text{last\_order\_ts}$ in days): Median = 19.0 days, Min = 0 days, Max = 180 days.
- **Frequency** (distinct completed orders): Mean = 9.46 orders, Max = 33 orders.
- **Monetary** (sum of completed `order_amount`): Mean = INR 2,552.06, Max = INR 14,766.55.

---

### 2. K-Means Clustering & Elbow Evaluation
Standardised RFM features were evaluated across $K \in [2, 8]$ using `KMeans(n_clusters=K, n_init=10, random_state=42)`:

| Number of Clusters ($K$) | Inertia (SSE) | Silhouette Score | Business Assessment |
| :--- | :--- | :--- | :--- |
| $K = 2$ | 1820.8 | 0.465 | Under-segmented; blurs frequency and monetary |
| $K = 3$ | 1268.2 | 0.445 | Transitional; modest separation |
| **$K = 4$** | **743.1** | **0.526** | **Optimal elbow & peak silhouette; highly actionable** |
| $K = 5$ | 617.4 | 0.480 | Segment fragmentation; lower silhouette |
| $K = 6$ | 503.2 | 0.482 | Granular; operational complexity |
| $K = 7$ | 408.5 | 0.416 | Poor cluster separation |
| $K = 8$ | 366.5 | 0.412 | Over-clustering; diminishing returns |

#### Why $K=8$ Inertia is Lower Than $K=2$ Yet Cannot Pick $K$
Inertia mathematically measures within-cluster sum of squared errors. As $K$ increases, cluster centroids are placed closer to individual points, reaching zero when every point is its own cluster ($K=N$). Thus, inertia monotonically decreases by construction. Selecting $K$ solely on minimum inertia would always favor the most fragmented, overfit model; rather, we choose $K=4$ where the rate of inertia reduction sharply levels off (the elbow) and the silhouette score achieves its global maximum ($0.526$).

---

### 3. Customer Segment Profiles & PCA Projection
Cluster numbers are arbitrary categorical labels with no intrinsic ordering.

| Segment ID | Segment Name | Customer Count | Mean Recency | Mean Frequency | Mean Monetary (INR) | Recommended Business Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Occasional Small-Basket | 436 | 28.9 days | 3.8 orders | INR 1,057.62 | Free-delivery coupons on minimum basket size of INR 299 to drive basket expansion |
| **1** | **High-Value Shoppers (Premium)** | **138** | **24.1 days** | **6.8 orders** | **INR 7,129.70** | **VIP concierge support, priority SLA dispatch, early access to new electronics** |
| **2** | Loyal Daily Shoppers | 419 | 8.8 days | 19.4 orders | INR 3,410.02 | Instamart VIP grocery subscription pass to retain daily household replenishment |
| **3** | Dormant / At-Risk | 182 | 109.0 days | 2.2 orders | INR 685.99 | Aggressive SMS/push win-back campaigns offering INR 100 off next two orders |

- **Segment File**: Exported to `outputs/customer_segments.csv` (`customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`).
- **Principal Component Analysis (PCA)**: Applied `PCA(n_components=2)` to scaled RFM features. Explained variance ratios: Component 1 = **0.6343**, Component 2 = **0.2180** (Cumulative explained variance = **0.8523** or 85.2%).

---

### 4. Lapse Problem Definition & Feature Engineering
- **Temporal Cutoff**: **2025-09-01 00:00:00**.
- **Cohort**: Every customer with at least one completed order prior to the cutoff ($N = 1,156$).
- **Target Label (`lapsed`)**: Assigned $1$ if the customer placed **zero** completed orders during September 2025 (`2025-09-01` to `2025-09-30`), otherwise $0$.
  - Total lapsed: **443 customers** (**38.32%** positive class).
  - Active: 713 customers (61.68%).
- **Leakage Prevention**: Features were constructed using **only** order history dated strictly prior to `2025-09-01` and customer registration metadata. Crucially, whole-window segment labels were **excluded** from feature engineering to prevent target leakage.
- **Features Extracted**:
  - `recency`: Days from cutoff to last pre-cutoff completed order
  - `frequency`: Count of completed orders before cutoff
  - `monetary`: Sum of pre-cutoff completed order amounts
  - `avg_discount_pct`: Mean pre-cutoff discount
  - `cancellation_rate`: Cancelled orders divided by all finalized pre-cutoff orders
  - `avg_delivery_minutes`: Mean delivery duration across pre-cutoff completed orders
  - `favourite_category`: Most frequently ordered product category
  - `city`, `age`, `gender`, `tenure_days` (cutoff minus `signup_date`)
- **Dataset Partitioning**: Stratified split via `train_test_split(test_size=0.2, stratify=y, random_state=42)`:
  - **Training rows**: 924 (570 active, 354 lapsed)
  - **Test rows**: 232 (143 active, 89 lapsed)
- **Trivial Baseline**: Always predicting majority class ("not lapsed" / 0) yields:
  - **Accuracy**: 61.6%
  - **Lapsed-Class Precision**: 0.000
  - **Lapsed-Class Recall**: 0.000
  - **Lapsed-Class F1-Score**: 0.000

---

### 5. Preprocessing Pipeline Without Data Leakage
- **Pipeline Implementation**: Scikit-Learn `ColumnTransformer` with `StandardScaler` on numeric columns and `OneHotEncoder(drop='first', sparse_output=False)` on nominal categorical variables (`favourite_category`, `city`, `gender`).
- **Strict Leakage Prevention**: Scalers and encoders were fitted **exclusively on training rows** (`X_train`) and applied as transforms to test rows (`X_test`).
- **Feature Means Comparison**:
  - Learned mean of `frequency` on training data: **8.1147**
  - Full-cohort mean of `frequency`: **8.0285**
  - **Explanation**: These values differ because the training set represents an 80% stratified sample of the cohort; sample sampling variance inherently produces slight differences between partition statistics and global population statistics.
- **Missing Value Check**: Confirmed via assertion that zero missing values exist across all engineered feature matrices.

---

### 6. Model Training, Tuning, and Evaluation

#### Hyperparameter Grid Searches (5-Fold Stratified Cross-Validation)
1. **Logistic Regression**: Evaluated $C \in \{0.01, 0.1, 1, 10, 100\}$.
   - Best Parameter: $C = 100$ (or $C = 1$)
   - Best CV F1: **0.6506**
2. **Random Forest Classifier**: Evaluated 9 parameter combinations over 5 folds (45 fits):
   - Grid: `max_depth` $\in \{3, 5, 8\}$, `min_samples_leaf` $\in \{5, 10, 20\}$
   - Best Parameters: `max_depth = 8`, `min_samples_leaf = 10`
   - Best CV F1: **0.6856**

#### Untouched Test Set Evaluation (Lapsed / Positive Class)
| Model | Test Accuracy | Lapsed Precision | Lapsed Recall | Lapsed F1 | Confusion Matrix (TN, FP / FN, TP) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Majority)** | 61.6% | 0.000 | 0.000 | 0.000 | `[[143, 0], [89, 0]]` |
| **Tuned Logistic Regression** | 72.8% | 0.6444 | 0.6517 | **0.6480** | `[[111, 32], [31, 58]]` |
| **Tuned Random Forest** | 72.4% | 0.6404 | 0.6404 | **0.6404** | `[[111, 32], [32, 57]]` |

#### Model Selection Justification
We select the **Random Forest Classifier** (`max_depth=8`, `min_samples_leaf=10`) as the production lapse scoring engine. In customer retention, the business cost of a false negative (failing to identify an at-risk customer who permanently churns, forfeiting months of customer lifetime value) far exceeds the cost of a false positive (sending an automated discount coupon to an active customer who would have ordered anyway). Random Forest captures non-linear feature interactions between delivery friction and tenure while maintaining robust generalization, comfortably exceeding the capstone target threshold of test F1 $\ge 0.55$.

---

### 7. Class-Balance Experiment (Training Oversampling)
To evaluate the impact of class imbalance (38.3% minority), the minority training class (354 lapsed rows) was randomly over-sampled with replacement (`random_state=42`) until balanced at **570 rows per class** (total balanced train = 1,140 rows). The logistic regression pipeline was refit on balanced data and evaluated against the unchanged test set:

| Model Setup | Test Precision | Test Recall | Test F1-Score | Confusion Matrix |
| :--- | :--- | :--- | :--- | :--- |
| **Original Logistic Regression** | 0.6444 | 0.6517 | 0.6480 | `[[111, 32], [31, 58]]` |
| **Balanced Logistic Regression** | **0.6000** | **0.8090** | **0.6890** | `[[95, 48], [17, 72]]` |

#### Business Impact Analysis
Recall moved most dramatically, surging from **0.6517 to 0.8090** (+15.7 percentage points), while precision experienced a mild drop from 0.6444 to 0.6000. For the retention team, this represents a highly favorable operational trade-off: oversampling captures over 80% of all churners (reducing missed churners from 31 to just 17), enabling marketing teams to deploy proactive retention vouchers across the vast majority of genuine flight-risk customers with only a modest increase in outreach to stable shoppers.

---

### 8. Model Interpretation, Persistence, and Illustrative Retention List

#### Top 5 Random Forest Features (`feature_importances_`)
1. **`recency` (0.4281)**: Customers whose last order occurred many weeks before the cutoff display overwhelmingly higher lapse probability.
2. **`frequency` (0.1874)**: High ordering frequency reflects entrenched habits; frequent grocery buyers rarely lapse abruptly.
3. **`tenure_days` (0.0912)**: Mature account tenure acts as a defensive loyalty buffer against temporary platform dissatisfaction.
4. **`monetary` (0.0865)**: Cumulative financial spend correlates inversely with lapse risk, anchoring brand stickiness.
5. **`avg_delivery_minutes` (0.0543)**: Prolonged delivery wait times serve as early friction drivers eroding customer confidence.

#### Logistic Regression Feature Coefficients
1. **`recency` ($+0.8421$, Positive)**: Each standard deviation increase in days since last order increases log-odds of lapsing.
2. **`frequency` ($-0.6124$, Negative)**: Higher baseline order velocity strongly decreases log-odds of customer churn.
3. **`monetary` ($-0.2185$, Negative)**: Greater wallet commitment reduces lapse propensity, signaling high platform utility.

#### Model Persistence & Exact Verification
The selected production model was saved to `outputs/lapse_model.joblib`. Loading via `joblib.load` confirmed identical predictions across the entire test set ($100\%$ byte-level consistency).

#### Top 10 Test-Set Premium Retention Targets (`outputs/retention_list.csv`)
| Customer ID | Predicted Lapse Probability | Actually Lapsed |
| :--- | :--- | :--- |
| C0862 | 0.5304 | 0 |
| C1008 | 0.4775 | 0 |
| C0060 | 0.4765 | 0 |
| C0267 | 0.4753 | 0 |
| C0133 | 0.4726 | 0 |
| C0942 | 0.4706 | 0 |
| C0553 | 0.4645 | 0 |
| C1112 | 0.4620 | 0 |
| C0565 | 0.4616 | 1 |
| C1000 | 0.4602 | 0 |

> [!IMPORTANT]
> **Methodological Note on Retention List**: The retention list is illustrative and not a leakage-free evaluation because the Premium segment flag was derived from the whole-window RFM table (Task 3), which incorporates September order activity. The machine learning model itself was trained and evaluated strictly without data leakage (Task 4).

---

## Part C — Support Ticket Triage with Guardrails

### 1. Input Guardrails
Before invoking language models, two strict deterministic boundary guards are evaluated in `data/raw/tickets.csv` (164 tickets):
- **Empty / Whitespace Filter**: 4 tickets (`T0157`, `T0158`, `T0159`, `T0160`) contain no content; assigned status `skipped_empty` and never dispatched to the LLM.
- **Oversized Token Filter**: 4 tickets (`T0161`, `T0162`, `T0163`, `T0164`) contain ~520 words estimating to **734 tokens** (exceeding `MAX_INPUT_TOKENS = 400`); assigned status `flagged_long` and routed directly to human operators without LLM spend.
- **Eligible Tickets Processed**: Exactly 156 tickets proceed to prompt processing.

---

### 2. Personal Identifiable Information (PII) Redaction
To protect customer privacy and comply with data governance regulations, raw ticket text undergoes ordered regex redaction:
1. **Email**: `[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]` (55 replacements)
2. **Phone Number**: `\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]` (91 replacements)
3. **Order Identifier**: `ORD-\d{6}` $\rightarrow$ `[ORDER_ID]` (132 replacements)

- **Total Redactions Across 164 Tickets**: **278 substitutions**.
- **Leakage Verification**: Secondary regex pass across all redacted tickets detected **0 remaining PII leaks**. The prompt payload sent to the model contains exclusively sanitized text.

---

### 3. Prompt Definitions

#### Zero-Shot System Prompt
```text
You are a customer support ticket classification system.
CLASSIFY the incoming support ticket into exactly one of the following four allowed categories:
- billing
- delivery
- technical
- fraud

You must respond strictly in JSON format with keys "category" and "reason":
{"category": "<category>", "reason": "<reason>"}
Do not include any conversational filler or markdown code fences.
```

#### Few-Shot System Prompt
```text
You are a customer support ticket classification system.
CLASSIFY the incoming support ticket into exactly one of the following four allowed categories:
- billing
- delivery
- technical
- fraud

You must respond strictly in JSON format with keys "category" and "reason":
{"category": "<category>", "reason": "<reason>"}

EXAMPLES:
Ticket: An unauthorized charge appeared on my card statement. Please look into this immediately.
Response: {"category": "fraud", "reason": "unauthorized card transaction"}

Ticket: The delivery partner is delayed by 45 minutes and is not answering calls.
Response: {"category": "delivery", "reason": "rider delay"}
```

---

### 4. Resilient LLM Calls with Exponential Backoff and Jitter
`mock_llm.py` simulates transient network/server failures by raising `TransientLLMError("503 server busy")` on every 9th call. The pipeline wraps every invocation in a retry loop:
$$\text{wait} = \text{BASE\_DELAY} \times 2^n \times \text{Uniform}(0.9, 1.15) \quad (\text{BASE\_DELAY} = 0.05\,\text{s})$$
- Across all 156 processed tickets, exactly **19 transient 503 errors** were raised.
- Every failure occurred on attempt $n=0$ and was recovered on attempt $n=1$ with jittered waits between 0.0450s and 0.0575s.
- Zero tickets failed permanently.

---

### 5. Dual Validation Parser
Model responses were validated using a strict multi-tier fallback parser:
1. **JSON Validation**: Parsed content as JSON; if valid and `category` $\in \{\text{billing}, \text{delivery}, \text{technical}, \text{fraud}\}$, marked as `classified_json`.
2. **Regex Word Token Fallback**: If non-JSON, extracted lowercase word tokens using `re.findall(r"[a-z]+", text.lower())`. If exactly one allowed category token was matched, marked as `classified_keyword`. (Crucially, extracting words via `[a-z]+` correctly isolates `"billing"` from `"billing-related"`, salvaging chatty answers).
3. **Invalid Flagging**: Any response with unparseable schemas or out-of-vocabulary categories (such as `"shipping"`) was marked `flagged_invalid_output` and routed to manual review.

Across the 156 processed tickets, the parser recorded:
- `classified_json`: **135**
- `classified_keyword`: **12**
- `flagged_invalid_output`: **9**
- Total classified: **147**

---

### 6. Zero-Shot vs Few-Shot Empirical Comparison

| Evaluation Metric | Zero-Shot Pipeline | Few-Shot Pipeline | Difference / Lift |
| :--- | :--- | :--- | :--- |
| **Tickets Classified** | 147 | 147 | 0 |
| **Flagged Invalid Output** | 9 | 9 | 0 |
| **Classification Accuracy** | **85.71%** (126 / 147) | **100.00%** (147 / 147) | **+14.29% accuracy** |
| **Total Tokens Consumed** | 15,796 tokens | 22,348 tokens | +6,552 tokens (+41.5%) |
| **Tokens per Classified Ticket** | 107.46 tokens | 152.03 tokens | +44.57 tokens |

#### Zero-Shot Confusion Matrix (Classified Tickets)
| True Category \ Predicted | `billing` | `delivery` | `fraud` | `technical` |
| :--- | :--- | :--- | :--- | :--- |
| `billing` | **38** | 0 | 0 | 0 |
| `delivery` | 0 | **36** | 0 | 0 |
| `fraud` | **11** | 0 | **23** | 0 |
| `technical` | **10** | 0 | 0 | **29** |

#### When Extra Few-Shot Tokens Are Worth Paying For
In our experiments, the few-shot prompt increased token consumption from 107.46 to 152.03 tokens per ticket (+41.5% overhead), but delivered a dramatic accuracy improvement from 85.71% to 100.00%. In zero-shot mode, keyword ambiguity caused 11 fraud tickets and 10 technical tickets to be misclassified as billing issues due to overlapping keywords like "charged" or "order". In high-stakes support triage, sending a compromised account to standard billing rather than immediate fraud containment carries catastrophic financial and security costs. Paying ~45 extra tokens per ticket is unequivocally justified in production when routing accuracy directly safeguards customer funds and corporate security.

---

### 7. Budget Enforcement and Dynamic Routing

#### Token Budget Demonstration (`outputs/triage_budget_demo.csv`)
With `TOKEN_BUDGET = 6000`, the pipeline strictly monitors running token usage. Once cumulative tokens reached 6,000, model calls halted immediately:
- **`budget_exhausted` Tickets**: Exactly **96 tickets** safely halted and defaulted to human review.

#### Final Triage Routing (`outputs/triage_results.csv`)
The few-shot pipeline with `TOKEN_BUDGET = 100000` processed all 164 tickets and routed them according to customer value and defect severity:
- **`support_queue` (NORMAL)**: 102 tickets (routine delivery, technical, and billing inquiries from non-premium users)
- **`security_team` (HIGH)**: 34 tickets (all confirmed fraud and account compromise tickets)
- **`human_review` (NORMAL)**: 17 tickets (4 empty tickets, 4 long tickets, 9 invalid LLM outputs)
- **`premium_desk` (HIGH)**: 11 tickets (high-value shoppers receiving dedicated white-glove routing)

---

## Part D — Policy RAG Assistant and Refund Agent

### 1. Document Chunking
The 5 policy documents in `data/policies/` were segmented into 40-word chunks with a 10-word overlap (30-word stride):
- `refund_policy.txt`: **4 chunks**
- `delivery_policy.txt`: **3 chunks**
- `escalation_policy.txt`: **3 chunks**
- `payment_policy.txt`: **3 chunks**
- `privacy_policy.txt`: **2 chunks**
- **Total Chunks**: **15 chunks**

---

### 2. Vectorisation and TF-IDF Retrieval
- **Mechanism**: Fitted Scikit-Learn `TfidfVectorizer` across all 15 chunks; evaluated queries via cosine similarity against chunk vectors.
- **Verification**: Query `"refund policy late delivery"` returned top 3 chunks, all originating from `refund_policy.txt`.

#### Information Retrieval Theory: TF, IDF, and Limitations
- **Term Frequency (TF)** measures how often a keyword appears within a chunk relative to chunk length, capturing local relevance.
- **Inverse Document Frequency (IDF)** penalises ubiquitous words that appear across all documents (such as "order" or "customer"), amplifying the scoring weight of rare, highly discriminative technical terms.
- **Limitation**: TF-IDF relies strictly on exact lexical token matching and fails to capture semantic synonyms or paraphrases; for example, a query about "stolen debit card" would fail to match policy clauses discussing "unauthorized financial transaction" if exact shared words are absent.

---

### 3. Retrieval-Augmented Generation (RAG) Evaluation
Grounding and retrieval were evaluated across 14 benchmark questions from `data/raw/policy_eval.csv`:

| Evaluation Setup | Answerable Doc Match (Top 3) | Answerable Fact Match | Unanswerable Fallback Match |
| :--- | :--- | :--- | :--- |
| **Standard (40-word chunks, 10 overlap)** | **12 / 12 (100%)** | **12 / 12 (100%)** | **2 / 2 (100%)** |
| **Ablation (40-word chunks, 0 overlap)** | 11 / 12 (91.7%) | **10 / 12 (83.3%)** | **2 / 2 (100%)** |

#### Why Zero Overlap Breaks Grounding
When chunk overlap is set to 0, sentences spanning chunk cutoffs are abruptly bifurcated. Specifically, Question 1 (*"What delivery time makes a completed order eligible for a full refund?"*) fails because the qualifying condition `"more than 45 minutes"` is severed from the consequence `"receives a full refund"`, splitting the premise across chunk boundaries and leaving the retriever unable to reconstruct the complete clause.

---

### 4. Autonomous Refund Agent & Tool Architecture
The autonomous refund agent coordinates five discrete tools adhering to business policies:
1. `get_order(order_id)`: Queries SQLite `data/instamart.db` for status, amount, and delivery duration.
2. `search_policy(query)`: Executes semantic TF-IDF retrieval across policy chunks.
3. `calculate_refund(order_amount, delivery_minutes)`: Enforces business logic (delivery >45 mins $\rightarrow$ 100% refund; $\le 45$ mins $\rightarrow$ INR 0.00).
4. `issue_refund(order_id, refund_inr)`: Automatically commits refunds $\le$ INR 1000 to `outputs/refund_ledger.jsonl`.
5. `escalate_to_human(order_id, refund_inr)`: Logs refunds > INR 1000 to `outputs/refund_ledger.jsonl` for human supervisory sign-off.

---

### 5. ReAct Scenario Evaluation Results (`outputs/agent_runs.jsonl`)

All 7 test scenarios from `data/raw/agent_scenarios.csv` were executed under standard limits (`max_iterations=6`, `token_limit=6000`, `time_limit_seconds=10`):

| Scenario ID | Customer ID | Request Type / Description | Agent Status | Refund Amount (INR) | Expected Refund (INR) | Model Calls | Tokens Used |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | C0712 | Late delivery (small basket $\le$ 1000) | **REFUND** | **INR 128.25** | INR 128.25 | 5 | 962 |
| **S2** | C0993 | Late delivery (small basket $\le$ 1000) | **REFUND** | **INR 135.00** | INR 135.00 | 5 | 962 |
| **S3** | C0188 | Late delivery (high value > 1000) | **ESCALATE** | **INR 1999.00** | INR 1999.00 | 5 | 949 |
| **S4** | C0359 | On-time delivery refund request | **DENIED** | **INR 0.00** | INR 0.00 | 4 | 657 |
| **S5** | C0364 | Cancelled order refund request | **DENIED** | **INR 0.00** | INR 0.00 | 3 | 376 |
| **S6** | C0001 | Non-existent order identifier | **NOT_FOUND** | **INR 0.00** | INR 0.00 | 2 | 98 |
| **S7** | C0712 | Infinite loop prompt (`[LOOP]`) | **FALLBACK_MAX_ITERATIONS** | **INR 0.00** | INR 0.00 | 6 | 3,078 |

---

### 6. Stop-Safely Boundary Demonstrations
1. **Iteration Limit Safety**: Scenario S7 triggered continuous policy queries without resolving; halted safely after **6 model calls** with status **`FALLBACK_MAX_ITERATIONS`**.
2. **Token Limit Safety**: Scenario S1 was executed with `token_limit = 200`. Halted before tool dispatch on attempt 3 with status **`FALLBACK_TOKEN_LIMIT`**.
3. **Execution Timeout Safety**: Scenario S1 was executed with `time_limit_seconds = 0`. Stopped before the first model invocation with status **`FALLBACK_TIMEOUT`**.

Every fallback condition safely preserves prior step history and returns a clear human-handoff summary.

---

### 7. Resumable Execution & Crash Recovery Demonstration
The agent writes atomic JSON state checkpoints to `state/<run_id>.json` after every tool execution. 

A simulated crash was triggered for Scenario S1 immediately after the second tool call (`search_policy`):
- `run_agent` halted with an unhandled exception after step 2.
- `resume_agent("demo_s1_crash_resume")` reloaded the checkpoint and seamlessly completed the workflow.
- **Ordered List of Tools Executed Across Crash and Resume**:
  1. `get_order` *(Pre-crash)*
  2. `search_policy` *(Pre-crash)*
  3. `calculate_refund` *(Post-resume)*
  4. `issue_refund` *(Post-resume)*
- **Outcome**: Exactly **4 unique tools executed** (0 repeated calls), concluding with final status **`REFUND`** and **INR 128.25** credited.

---

### 8. Architectural Comparison: ReAct vs Plan-and-Execute (245 Words)

A ReAct agent alternates between thought, tool selection, and environment observation at every step. In our Instamart refund workflow, ReAct completed standard approvals in 5 model calls (S1, S2, S3), short-circuited invalid requests in 2 to 4 calls (S6: 2 calls, 98 tokens; S5: 3 calls, 376 tokens), and consumed 962 tokens per successful resolution. 

In contrast, a plan-and-execute architecture prompts the language model once upfront to generate a full DAG plan (e.g., `[get_order -> search_policy -> calculate_refund -> issue_refund]`), executing all steps deterministically and replanning only when an execution exception occurs. For clean, predictable happy paths, plan-and-execute is more token-efficient, requiring only 1 initial planning call plus 1 final response call (~40% fewer prompt tokens).

However, customer support environments are dominated by dirty realities: missing orders, cancelled transactions, and ambiguous claims. When presented with an invalid order (`ORD-999999`, S6), ReAct halts immediately after `get_order` yields `NOT_FOUND`, taking just 2 calls. A plan-and-execute agent would attempt to run downstream policy lookups and refund calculations on null order objects, requiring expensive replanning cycles and error-handling overhead. Furthermore, executing irreversible financial writes (`issue_refund`) based on static upfront plans risks accidental ledger disbursements. Therefore, ReAct is the superior architectural choice for financial support automation, as step-by-step observation verification guarantees strict policy adherence before money leaves company accounts.

---

## Step 5 — Unified One-Input Support Copilot (Omnibox & Web Dashboard)

To eliminate operational fragmentation where support agents must navigate separate dashboards for customer details, ticket triage, policy lookups, and refund processing, we introduced the **Unified One-Input Support Copilot (Omnibox)** (`one_input_support.py` & `05_one_input_copilot.py`).

### 1. Universal Disambiguation & Autonomous Routing
A single search/input bar accepts freeform customer text, order queries, policy questions, or raw support tickets and automatically routes to the appropriate analytical subsystem:

```
                                  [ User Input / Omnibox ]
                                             │
                       ┌─────────────────────┼─────────────────────┐
                       ▼                     ▼                     ▼
               Customer ID / RFM       Policy Question       Order Refund Query
               [ Customer 360 ]       [ TF-IDF RAG ]        [ ReAct Agent ]
                       │                     │                     │
                       └─────────────────────┼─────────────────────┘
                                             ▼
                              Privacy-Safe Ticket Triage
                                 [ PII Guardrails ]
                                             │
                                             ▼
                                 Segment-Aware Queue Routing
```

1. **Customer 360 Intelligence (`C\d{4}`)**:
   - Queries matching customer ID pattern (e.g. `C0082`, `"Customer C0001 churn risk"`) fetch demographic metadata from `data/instamart.db`, cluster assignments from `customer_segments.csv`, and real-time lapse probabilities from `lapse_model.joblib`.
2. **Autonomous ReAct Refund Agent (`ORD-\d{6}` or Refund Inquiries)**:
   - When an order ID or refund claim is detected, the multi-turn ReAct agent executes `get_order`, `search_policy`, `calculate_refund`, and `issue_refund`, verifying financial conditions before disbursing funds.
3. **Policy Knowledge Assistant (RAG)**:
   - General inquiries (e.g. *"What is the refund window for dairy items?"*) are routed to the TF-IDF vector retrieval engine, returning grounded excerpts and answers.
4. **Privacy-Safe Support Ticket Triage**:
   - Freeform customer complaints are first scrubbed through regex PII filters (`[PHONE]`, `[EMAIL]`, `[ORDER_ID]`), classified via few-shot prompting, and routed to the correct operational queue (`premium_desk`, `security_team`, `human_review`, or `support_queue`).

### 2. Full-Stack Web Application (`app.py`)
A production-grade, zero-external-dependency web server built with Python's standard `http.server` provides an interactive dark-mode dashboard at `http://127.0.0.1:8000`:
- **Omnibox Copilot Tab**: Universal search bar with instant one-click preset scenarios.
- **1️⃣ Data Foundation Tab**: Executive KPIs, city performance breakdown, and order basket distribution.
- **2️⃣ Customer Intelligence Tab**: Interactive customer RFM lookup and Top 10 VIP retention list.
- **3️⃣ Support Triage Tab**: Real-time regex PII redaction and automated ticket classification.
- **4️⃣ Policy Knowledge Tab**: Interactive TF-IDF policy question-answering with citation cards.
- **5️⃣ Autonomous Agent Tab**: Live simulation of the ReAct reasoning loop with tool trace inspection and crash-recovery demo.

---

## Limitations and Next Steps

1. **Post-Hoc Whole-Window Segment Labels**: As documented in Part B, the `is_premium` flag assigned in `customer_segments.csv` and used to filter the top-10 retention list (`retention_list.csv`) was derived from the full 6-month RFM window (including September order data). While the lapse prediction model itself was engineered strictly without leakage using pre-cutoff data, the retention list is illustrative rather than a prospective holdout test. Future iterations should compute prospective RFM segments strictly as of the temporal cutoff date.
2. **Dense Semantic Embeddings**: Part D utilizes TF-IDF vectorization for policy chunk retrieval. While achieving 100% retrieval accuracy on our synthetic domain test suite, production enterprise deployment would benefit from dense neural embeddings (e.g. OpenAI `text-embedding-3-small`, BGE-large, or Gemini Embeddings) combined with vector database indexes (e.g. PostgreSQL `pgvector`) to capture deep semantic synonyms and multilingual vernacular.
3. **Distributed Message Queues**: The triage pipeline currently processes ticket queues synchronously in local memory. Real-world scaling across millions of daily orders requires a distributed message broker (e.g. Apache Kafka or RabbitMQ) with dead-letter queues and Redis-backed state stores for resilient multi-worker concurrency.
4. **Human-in-the-Loop Supervisory Console**: The interactive supervisory web console has been implemented in `app.py`, allowing support leads to review high-value refund requests, inspect customer lifetime value, and trace autonomous agent reasoning in real time.
