# Instamart Customer Intelligence & Support Copilot

An enterprise-ready data and AI copilot created for Swiggy Instamart to clean messy transaction records, forecast customer churn, sanitize and route customer support tickets, search policy documents, and resolve refund requests autonomously.

---

## Overview

Swiggy Instamart delivers groceries, fresh dairy, snacks, toiletries, and small electronics across six Indian metro cities. As order numbers grow, the business faces three major challenges:
1. **Unreliable Raw Data**: Daily order spreadsheets contain duplicate entries, conflicting statuses, missing values, impossible delivery times, and unstandardized customer demographics.
2. **Customer Retention Blindspots**: The marketing team cannot easily spot high-value customers who are about to stop ordering, missing the chance to re-engage them in time.
3. **High Support Ticket Volume**: Customer care receives thousands of free-text tickets containing sensitive personal information (PII) that mix fraud reports, delivery delays, app issues, and refund requests.

The **Customer Intelligence & Support Copilot** brings everything together into five connected, orderly modules:
- **1. Data Foundation: Clean, Store, and Query**: Fixes errors in raw files, stores clean data in SQLite (`data/instamart.db`), and answers key business questions using SQL and data exploration.
- **2. Customer Segmentation and Lapse Prediction**: Groups customers into behavioural clusters with K-Means and trains a leak-free Random Forest model to predict which VIP shoppers might churn.
- **3. Support Ticket Triage with Privacy Guardrails**: Removes sensitive personal details (phone numbers, emails, order IDs), classifies issues with zero-leakage few-shot prompts, enforces a token budget, and directs tickets to the right team.
- **4. Policy Knowledge Assistant and Autonomous Refund Agent**: Indexes company policies using TF-IDF RAG, answers policy queries, and automates refund approvals through a ReAct reasoning loop with crash recovery.
- **5. Unified One-Input Support Copilot and Web Dashboard**: Combines all tools into a single Omnibox input interface and provides an interactive web application on port 8000.

All model interactions run completely offline using the provided deterministic `mock_llm.py` file. No external API keys, accounts, or paid services are required. All currency values are strictly recorded in Indian Rupees (INR).

---

## Reproduce

The entire repository runs deterministically and reproduces every checkpoint. Follow the numbered terminal steps below in a clean environment to install dependencies, run each stage in sequence, and test all outputs:

```bash
# 1. Clone the repository and go to the project folder
git clone https://github.com/sriniprojects30-beep/instamart-customer-copilot.git
cd instamart-customer-copilot

# 2. Make sure you are on the main branch
git branch -M main

# 3. Create a clean virtual environment (Python 3.11 or newer)
python -m venv .venv

# 4. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# 5. Upgrade pip and install the required libraries
python -m pip install --upgrade pip
pip install -r requirements.txt

# 6. Generate the synthetic data and policy files
python generate_data.py

# =====================================================================
# ORDER-WISE PIPELINE EXECUTION (01 to 05)
# =====================================================================

# Step 1: Clean raw data, build SQLite database, and run SQL queries
python 01_data_foundation.py

# Step 1 Checkpoint Validation
python validate.py

# Step 2: Build RFM segments, run K-Means, and train the lapse model
python 02_customer_intelligence.py

# Step 3: Redact PII, run few-shot ticket triage, and apply routing
python 03_support_triage.py

# Step 4: Index policies, test RAG, and simulate refund agent scenarios
python 04_policy_refund_agent.py

# Step 5: Test the Unified One-Input Copilot CLI
python 05_one_input_copilot.py --query "Customer C0082 profile and churn risk"

# =====================================================================
# AUTOMATED REPRODUCIBILITY & QA VERIFICATION
# =====================================================================

# Run all steps end-to-end with automated timing (~27 seconds):
python run_all.py

# Run the complete test suite (100% pass rate):
python test_suite.py

# Start the interactive web dashboard (Port 8000):
python app.py
# Open http://127.0.0.1:8000 in your web browser
```

---

## 1. Data Foundation: Clean, Store, and Query

### 1. Raw Data Audit
Before making any changes, we inspected the four raw files (`customers.csv`, `products.csv`, `orders.csv`, `tickets.csv`) to check row counts, data types, and issues.

| Dataset | Rows | Columns | Identifier Key | Data Issues Found |
| :--- | :--- | :--- | :--- | :--- |
| `orders.csv` | 13,084 | 9 | `order_id` (non-unique raw) | Exact duplicates, pending status rows, invalid text in amounts, negative and blank delivery times |
| `customers.csv` | 1,200 | 5 | `customer_id` | Missing ages, impossible ages (<10 or >100), inconsistent city letter casing, mixed gender values |
| `products.csv` | 40 | 4 | `product_id` | Clean catalog with prices from INR 20 to INR 1,999 |
| `tickets.csv` | 164 | 5 | `ticket_id` | Empty tickets (4 rows), oversized tickets (>400 tokens, 4 rows), raw customer PII |

#### Raw Data Audit Figures
All numbers observed match the project reference checkpoints exactly:

| Audit Metric | Observed Value | Target Reference | Result |
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

### 2. Cleaning Rules and Applied Transformations

#### Customer Cleaning Rules
1. **City Standardisation**: Stripped extra spaces and applied title casing (`.strip().title()`). This grouped 24 messy spellings into 6 standard cities:
   - **Pune**: 227
   - **Delhi**: 210
   - **Bengaluru**: 197
   - **Chennai**: 191
   - **Hyderabad**: 191
   - **Mumbai**: 184
   - Total: 1,200 customers.
2. **Gender Normalisation**: Mapped all 10 non-blank casing variations (`male`, `Male`, `M`, `m`, `MALE`, `female`, `Female`, `F`, `f`, `FEMALE`) and 30 blank values into three clean values:
   - **female**: 610
   - **male**: 560
   - **unknown**: 30
3. **Age Imputation**: Replaced 6 impossible values (280, -4, 0, 150, 999, 200) and 61 blank entries (67 total) with the median age of customers from the same city.
   - Missing ages after cleaning: 0.
   - Cleaned age range: 18 to 60 years.
4. **Signup Date Conversion**: Formatted all dates to standard `YYYY-MM-DD` strings.

#### Order Cleaning Rules (Applied in Exact Sequential Order)
1. **Timestamp Parsing**: Converted `order_ts` into standard Python `datetime` objects.
2. **Duplicate Removal**: Dropped 119 identical rows, reducing the total from 13,084 to 12,965 rows.
3. **Status Conflict Handling**: For orders appearing multiple times (such as `pending` followed by `completed`), sorted by timestamp and kept the latest record (`keep='last'`). This removed 656 outdated `pending` records and left 12,309 unique orders:
   - **Completed orders**: 11,110
   - **Cancelled orders**: 1,199
   - **Pending orders**: 0
4. **Order Amount Correction**: Replaced 120 unparseable values (`"N/A"` and `"error"`) with calculated values based on product catalog pricing:
   $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
   - Missing `order_amount` after calculation: 0.
5. **Delivery Duration Correction**: Replaced 106 negative values (`-10`) with missing values. For completed orders, filled missing delivery minutes with the median delivery duration of that product category. For cancelled orders, left delivery minutes blank because cancelled orders were never delivered.
   - Missing delivery minutes for completed orders: 0.
   - Missing delivery minutes for cancelled orders: 1,199.
   - Total revenue from completed orders: **INR 2,998,668.80**.

---

### 3. Database Storage and Checkpoint Verification
Cleaned tables were saved as CSV files in `data/clean/` (`customers.csv`, `products.csv`, `orders.csv`) and loaded into SQLite at `data/instamart.db`.

Running `validate.py` checks all 11 required checkpoints automatically:
```bash
python validate.py
# Output:
# ALL CHECKPOINTS PASSED
```

---

### 4. SQL Analysis Results (`sql/queries.sql`)

All 7 queries were run directly against `data/instamart.db` with the following outcomes:

| Query | Objective | Query Finding | Expected Target |
| :--- | :--- | :--- | :--- |
| **Q1** | High-value orders > 500, customers spending > 8,000 | **30 qualifying customers** | 30 customers |
| **Q2** | Three-way JOIN (`orders`, `customers`, `products`) | **30 city-category rows**, Revenue: **INR 2,998,668.80**. Top cell: **Bengaluru / Electronics** at **INR 233,909.45** | 30 rows; INR 2,998,668.80; Bengaluru / Electronics INR 233,909.45 |
| **Q3** | Scalar Subquery for items above average price | **11 products** priced above catalog average of INR 352.38 | 11 products |
| **Q4** | LEFT JOIN identifying customers with zero completed orders | **25 customers** with no completed orders | 25 customers |
| **Q5** | Dual CTEs finding customers spending above average | **550 customers** spending above benchmark average of **INR 2,552.06** | 550 customers; average INR 2,552.06 |
| **Q6** | Window Functions (`RANK()`, `DENSE_RANK()`, `ROW_NUMBER()`) | Top customer per city: Bengaluru **C0082** (INR 12,856.20), Chennai **C0930** (INR 10,151.40), Delhi **C0291** (INR 13,742.60), Hyderabad **C0743** (INR 12,730.05), Mumbai **C0693** (INR 14,766.55), Pune **C0426** (INR 10,377.10). | Exact city leaders confirmed |
| **Q7** | Cumulative revenue running total by day | **183 daily rows** (2025-04-01 to 2025-09-30). Final cumulative spend: **INR 2,998,668.80** | 183 rows; INR 2,998,668.80 |

#### Explanation of Window Function Differences (Q6)
In query Q6, exactly **8 rows** have a different `RANK()` compared to `ROW_NUMBER()`. This happens because when two customers in the same city have the exact same total spending, `RANK()` gives both rows the same rank and skips the next number (for example: 5, 5, 7). In contrast, `ROW_NUMBER()` always gives a unique consecutive number (for example: 5, 6, 7) regardless of ties.

---

### 5. Pandas Data Exploration and Cross-Check

#### Revenue Cross-Check by City and Category
Using `pd.pivot_table(..., margins=True)` confirmed the identical grand revenue total of **INR 2,998,668.80**. City totals:
- **Bengaluru**: Total = INR 523,643.40, Average = INR 276.47, Orders = 1,894
- **Chennai**: Total = INR 463,629.15, Average = INR 272.72, Orders = 1,700
- **Delhi**: Total = INR 502,829.80, Average = INR 243.62, Orders = 2,064
- **Hyderabad**: Total = INR 528,589.90, Average = INR 284.95, Orders = 1,855
- **Mumbai**: Total = INR 479,058.50, Average = INR 297.00, Orders = 1,613
- **Pune**: Total = INR 500,918.05, Average = INR 252.48, Orders = 1,984

#### Order Amount Distribution (Completed Orders)
- **Mean**: INR 269.91
- **Median**: INR 135.00
- **Skewness**: 4.61 (right-skewed distribution)
- **Quartiles**: Q1 = INR 64.80, Q3 = INR 297.00, IQR = INR 232.20
- **Tukey Fences**: Lower Fence = INR -283.50, Upper Fence = INR 645.30
- **Outliers**: 0 below lower fence, **897 outliers** above upper fence

#### Correlation Table (Completed Orders)
| Feature | `order_amount` | `discount_pct` | `quantity` | `delivery_minutes` |
| :--- | :--- | :--- | :--- | :--- |
| `order_amount` | 1.0000 | -0.0951 | 0.0873 | -0.0031 |
| `discount_pct` | -0.0951 | 1.0000 | -0.0004 | -0.0098 |
| `quantity` | 0.0873 | -0.0004 | 1.0000 | -0.0126 |
| `delivery_minutes` | -0.0031 | -0.0098 | -0.0126 | 1.0000 |

#### Executive Data Summary
This dataset covers 11,110 completed grocery and electronics orders across six Indian cities. The order amount distribution is right-skewed (skewness of 4.61), where the median order value is INR 135.00 and the mean is INR 269.91 due to larger electronics purchases. Applying Tukey's rule identifies 897 high-value orders above INR 645.30 (reaching up to INR 3,998.00). These are genuine premium orders rather than errors. Correlations between transaction variables are modest: order amount has a small negative correlation with discount percentage ($r = -0.0951$) and a weak positive correlation with item quantity ($r = +0.0873$). Delivery duration has virtually no link with basket size ($r = -0.0031$). Correlation does not mean causation; higher order amounts do not cause lower discounts, but simply reflect electronics items where percentage discounts were limited.

---

## 2. Customer Segmentation and Lapse Prediction

### 1. Building the RFM Table
RFM metrics were calculated for all 1,175 customers who completed at least one order up to the snapshot date **2025-10-01 00:00:00**:
- **Recency** (days from last order to snapshot): Median = 19.0 days, Range = 0 to 180 days.
- **Frequency** (count of completed orders): Mean = 9.46 orders, Max = 33 orders.
- **Monetary** (sum of completed spending): Mean = INR 2,552.06, Max = INR 14,766.55.

---

### 2. K-Means Clustering and Choosing K
Standardized RFM values were evaluated for $K$ from 2 to 8:

| Clusters ($K$) | Inertia (SSE) | Silhouette Score | Practical Assessment |
| :--- | :--- | :--- | :--- |
| $K = 2$ | 1820.8 | 0.465 | Too broad; merges frequency and spending |
| $K = 3$ | 1268.2 | 0.445 | Average cluster separation |
| **$K = 4$** | **743.1** | **0.526** | **Best balance; highest silhouette score and clear elbow** |
| $K = 5$ | 617.4 | 0.480 | Groups start splitting unnecessarily |
| $K = 6$ | 503.2 | 0.482 | Overly detailed for marketing campaigns |
| $K = 7$ | 408.5 | 0.416 | Weak group separation |
| $K = 8$ | 366.5 | 0.412 | Overfitted clustering |

#### Why Lowest Inertia Does Not Mean Best K
Inertia measures the sum of squared distances to cluster centers. As more clusters are added, centers get closer to data points, eventually hitting zero if every customer has their own cluster. Because inertia always decreases as $K$ rises, picking $K$ solely on the lowest inertia would always lead to over-clustering. We choose $K=4$ because it represents the elbow where the rate of drop slows down and the silhouette score reaches its maximum (0.526).

---

### 3. Customer Segments and PCA Analysis
The four customer clusters were profiled as follows:

| Segment ID | Name | Count | Mean Recency | Mean Frequency | Mean Spend (INR) | Recommended Business Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Occasional Shoppers | 436 | 28.9 days | 3.8 orders | INR 1,057.62 | Offer free shipping vouchers for baskets over INR 299 to raise order value |
| **1** | **High-Value VIPs (Premium)** | **138** | **24.1 days** | **6.8 orders** | **INR 7,129.70** | **Provide dedicated customer support, priority delivery, and early access to new items** |
| **2** | Daily Frequent Shoppers | 419 | 8.8 days | 19.4 orders | INR 3,410.02 | Provide a grocery delivery subscription pass to retain regular household orders |
| **3** | Inactive / Lapsing | 182 | 109.0 days | 2.2 orders | INR 685.99 | Send SMS win-back offers with INR 100 off the next two orders |

- **Segment File**: Stored in `outputs/customer_segments.csv`.
- **PCA Projection**: Two principal components explained **85.2%** of total variance (Component 1 = 63.4%, Component 2 = 21.8%).

---

### 4. Defining Customer Lapse Without Data Leakage
- **Cutoff Date**: **2025-09-01 00:00:00**.
- **Cohort**: Customers with at least one completed order before the cutoff ($N = 1,156$).
- **Lapse Target (`lapsed`)**: Labeled as $1$ if the customer placed **zero** completed orders in September 2025, else $0$.
  - Lapsed customers: **443** (38.32%).
  - Active customers: 713 (61.68%).
- **Leakage Prevention**: All features were generated using only orders placed before September 1, 2025. Whole-window cluster labels were excluded to prevent target leakage.
- **Features Used**:
  - `recency`, `frequency`, `monetary` before cutoff
  - `avg_discount_pct`, `cancellation_rate`, `avg_delivery_minutes`
  - `favourite_category`, `city`, `age`, `gender`, `tenure_days`
- **Data Split**: Stratified 80/20 train-test split (`random_state=42`):
  - **Train**: 924 rows (570 active, 354 lapsed)
  - **Test**: 232 rows (143 active, 89 lapsed)
- **Naive Baseline**: Always predicting "active" (0) gives:
  - Accuracy: 61.6%, Precision: 0.000, Recall: 0.000, F1: 0.000

---

### 5. Preprocessing Pipeline Without Data Leakage
- Used Scikit-Learn `ColumnTransformer` with `StandardScaler` for numbers and `OneHotEncoder(drop='first')` for categories.
- Transformers were fitted **only on training data** and used to transform test data.
- Average training frequency was 8.1147 while the full cohort average was 8.0285; this slight difference is expected due to random sampling across the 80% split.
- Verified that no missing values exist in the transformed datasets.

---

### 6. Model Training, Tuning, and Evaluation

#### 5-Fold Cross-Validation Hyperparameter Tuning
1. **Logistic Regression**: Best $C = 100$, CV F1 = **0.6506**.
2. **Random Forest**: Best `max_depth = 8`, `min_samples_leaf = 10`, CV F1 = **0.6856**.

#### Test Set Performance (Lapsed Class)
| Model | Test Accuracy | Precision | Recall | F1-Score | Confusion Matrix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Majority)** | 61.6% | 0.000 | 0.000 | 0.000 | `[[143, 0], [89, 0]]` |
| **Tuned Logistic Regression** | 72.8% | 0.6444 | 0.6517 | **0.6480** | `[[111, 32], [31, 58]]` |
| **Tuned Random Forest** | 72.4% | 0.6404 | 0.6404 | **0.6404** | `[[111, 32], [32, 57]]` |

#### Choosing the Final Model
We select the **Random Forest Classifier** (`max_depth=8`, `min_samples_leaf=10`). In customer retention, missing a churned customer who leaves permanently is much costlier than sending a coupon to a customer who would have stayed. Random Forest handles non-linear relationships well and surpasses the capstone benchmark of test F1 $\ge 0.55$.

---

### 7. Class Balancing Experiment
To test the effect of class imbalance, the 354 lapsed training rows were over-sampled to match the 570 active rows (1,140 balanced training rows):

| Model Setup | Test Precision | Test Recall | Test F1-Score | Confusion Matrix |
| :--- | :--- | :--- | :--- | :--- |
| **Original Logistic Regression** | 0.6444 | 0.6517 | 0.6480 | `[[111, 32], [31, 58]]` |
| **Balanced Logistic Regression** | **0.6000** | **0.8090** | **0.6890** | `[[95, 48], [17, 72]]` |

#### Business Impact
Balancing boosted recall from **65.17% to 80.90%** (+15.7%), while precision stayed healthy at 60.00%. This allows the marketing team to catch over 80% of churners (cutting missed customers from 31 down to 17) with only a small increase in outreach to non-lapsing shoppers.

---

### 8. Feature Importance and Retention List

#### Top 5 Features in Random Forest
1. **`recency` (0.4281)**: Time since last order is by far the strongest churn signal.
2. **`frequency` (0.1874)**: Regular habits protect against customer churn.
3. **`tenure_days` (0.0912)**: Longer account age builds loyalty.
4. **`monetary` (0.0865)**: Higher historical spending reduces churn probability.
5. **`avg_delivery_minutes` (0.0543)**: Slower delivery times increase customer dissatisfaction.

#### Model Export and Verification
The trained model was saved to `outputs/lapse_model.joblib`. Reloading the file produces identical predictions across all test rows.

#### Top 10 High-Risk Premium Retention Targets (`outputs/retention_list.csv`)
| Customer ID | Churn Probability | Actually Lapsed |
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

---

## 3. Support Ticket Triage with Privacy Guardrails

### 1. Input Guardrails
Before calling the model, raw tickets (`data/raw/tickets.csv`, 164 rows) are checked by two input rules:
- **Empty Text Filter**: 4 blank tickets (`T0157` to `T0160`) are marked `skipped_empty` and not sent to the model.
- **Length Filter**: 4 tickets (`T0161` to `T0164`) exceed 400 tokens (~734 tokens); these are marked `flagged_long` and routed to human review without spending model tokens.
- Exactly 156 tickets move forward to classification.

---

### 2. Personal Information (PII) Redaction
To protect privacy, ticket texts are cleaned using three regex rules:
1. **Email**: `[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]` (55 replacements)
2. **Phone**: `\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]` (91 replacements)
3. **Order ID**: `ORD-\d{6}` $\rightarrow$ `[ORDER_ID]` (132 replacements)

- **Total Redactions**: **278 replacements** across 164 tickets.
- **Privacy Check**: A second scan over all redacted messages confirmed **zero remaining PII leaks**.

---

### 3. Prompt Designs

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

### 4. Handling 503 Errors with Jittered Retries
`mock_llm.py` simulates network issues by raising a 503 error on every 9th call. The pipeline uses an exponential backoff retry wrapper:
$$\text{wait} = 0.05 \times 2^n \times \text{Uniform}(0.9, 1.15)\,\text{seconds}$$
- All 19 transient 503 errors recovered on the second attempt ($n=1$).
- Zero tickets failed permanently.

---

### 5. Response Validation
Output responses were parsed in three steps:
1. **JSON Parser**: If JSON is valid and category is one of the four allowed types, mark as `classified_json`.
2. **Keyword Fallback**: If not valid JSON, check lowercase word tokens with regex `[a-z]+`. If exactly one valid category is found, mark as `classified_keyword`.
3. **Invalid Flagging**: Any unparseable response is marked `flagged_invalid_output` and sent for manual review.

Results across 156 processed tickets:
- `classified_json`: **135**
- `classified_keyword`: **12**
- `flagged_invalid_output`: **9**
- Total classified: **147**

---

### 6. Zero-Shot vs Few-Shot Comparison

| Metric | Zero-Shot | Few-Shot | Difference |
| :--- | :--- | :--- | :--- |
| **Classified Tickets** | 147 | 147 | 0 |
| **Invalid Responses** | 9 | 9 | 0 |
| **Accuracy** | **85.71%** (126 / 147) | **100.00%** (147 / 147) | **+14.29%** |
| **Total Tokens Used** | 15,796 tokens | 22,348 tokens | +6,552 tokens (+41.5%) |
| **Tokens per Ticket** | 107.46 tokens | 152.03 tokens | +44.57 tokens |

#### Zero-Shot Misclassifications
In zero-shot mode, 11 fraud tickets and 10 technical tickets were mistakenly classified as billing because words like "charged" confused the model. In support operations, sending fraud complaints to standard billing causes severe financial and security risks. Spending ~45 additional tokens per ticket for few-shot examples is well worth the cost because it reaches 100% accuracy.

---

### 7. Token Budget and Queue Routing

#### Token Budget Run (`outputs/triage_budget_demo.csv`)
With `TOKEN_BUDGET = 6000`, processing stopped as soon as 6,000 tokens were spent:
- **`budget_exhausted` Tickets**: Exactly **96 tickets** safely paused and defaulted to human review.

#### Final Routing Outcomes (`outputs/triage_results.csv`)
Running few-shot classification across all 164 tickets routed them to four teams:
- **`support_queue` (NORMAL)**: 102 tickets (routine delivery, billing, and technical requests)
- **`security_team` (HIGH)**: 34 tickets (confirmed fraud and account security issues)
- **`human_review` (NORMAL)**: 17 tickets (4 blank, 4 oversized, and 9 unparseable responses)
- **`premium_desk` (HIGH)**: 11 tickets (VIP customer requests given priority care)

---

## 4. Policy Knowledge Assistant and Autonomous Refund Agent

### 1. Policy Document Chunking
The 5 policy documents in `data/policies/` were broken into 40-word chunks with a 10-word overlap:
- `refund_policy.txt`: **4 chunks**
- `delivery_policy.txt`: **3 chunks**
- `escalation_policy.txt`: **3 chunks**
- `payment_policy.txt`: **3 chunks**
- `privacy_policy.txt`: **2 chunks**
- **Total Chunks**: **15 chunks**

---

### 2. TF-IDF Retrieval
Chunks were converted into vectors using Scikit-Learn `TfidfVectorizer`. Queries are matched against chunk vectors using cosine similarity:
- **Term Frequency (TF)** measures how often a word appears in a chunk.
- **Inverse Document Frequency (IDF)** lowers the weight of common words (like "order" or "customer") and highlights rare, important words.
- Testing the query `"refund policy late delivery"` returned the top 3 chunks, all from `refund_policy.txt`.

---

### 3. Policy RAG Evaluation
Tested against 14 benchmark questions from `data/raw/policy_eval.csv`:

| Configuration | Top-3 Document Match | Fact Match | Fallback on Unanswerable |
| :--- | :--- | :--- | :--- |
| **Standard (40 words, 10 overlap)** | **12 / 12 (100%)** | **12 / 12 (100%)** | **2 / 2 (100%)** |
| **Ablation (40 words, 0 overlap)** | 11 / 12 (91.7%) | **10 / 12 (83.3%)** | **2 / 2 (100%)** |

#### Why Chunk Overlap Matters
With zero overlap, important sentences get cut in half at chunk boundaries. For example, Question 1 (*"What delivery time makes a completed order eligible for a full refund?"*) fails without overlap because `"more than 45 minutes"` is separated from `"receives a full refund"`, preventing the model from reading the full rule.

---

### 4. Autonomous Refund Agent Tools
The ReAct agent uses five dedicated tools:
1. `get_order(order_id)`: Fetches status, amount, and delivery time from SQLite (`data/instamart.db`).
2. `search_policy(query)`: Finds relevant policy rules using TF-IDF.
3. `calculate_refund(order_amount, delivery_minutes)`: Applies policy rules (delivery >45 mins gets 100% refund; $\le 45$ mins gets INR 0.00).
4. `issue_refund(order_id, refund_inr)`: Writes approved refunds $\le$ INR 1000 into `outputs/refund_ledger.jsonl`.
5. `escalate_to_human(order_id, refund_inr)`: Sends refunds > INR 1000 to human supervisors in `outputs/refund_ledger.jsonl`.

---

### 5. Benchmark Scenario Results (`outputs/agent_runs.jsonl`)

All 7 test scenarios from `data/raw/agent_scenarios.csv` were executed under standard limits:

| Scenario | Customer | Issue Description | Status | Refund (INR) | Expected (INR) | Calls | Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | C0712 | Late delivery ($\le$ INR 1000) | **REFUND** | **INR 128.25** | INR 128.25 | 5 | 962 |
| **S2** | C0993 | Late delivery ($\le$ INR 1000) | **REFUND** | **INR 135.00** | INR 135.00 | 5 | 962 |
| **S3** | C0188 | Late delivery (> INR 1000) | **ESCALATE** | **INR 1999.00** | INR 1999.00 | 5 | 949 |
| **S4** | C0359 | On-time delivery refund request | **DENIED** | **INR 0.00** | INR 0.00 | 4 | 657 |
| **S5** | C0364 | Cancelled order refund request | **DENIED** | **INR 0.00** | INR 0.00 | 3 | 376 |
| **S6** | C0001 | Non-existent order ID | **NOT_FOUND** | **INR 0.00** | INR 0.00 | 2 | 98 |
| **S7** | C0712 | Infinite loop prompt (`[LOOP]`) | **FALLBACK_MAX_ITERATIONS** | **INR 0.00** | INR 0.00 | 6 | 3,078 |

---

### 6. Safety Limits Demonstration
1. **Iteration Limit**: Scenario S7 triggered repeated policy searches without resolving; stopped safely after **6 calls** with status **`FALLBACK_MAX_ITERATIONS`**.
2. **Token Limit**: Scenario S1 with `token_limit = 200` stopped on call 3 with status **`FALLBACK_TOKEN_LIMIT`**.
3. **Timeout Limit**: Scenario S1 with `time_limit_seconds = 0` stopped immediately with status **`FALLBACK_TIMEOUT`**.

Each fallback safely retains previous steps and provides a clear summary for human handoff.

---

### 7. Crash Recovery and State Resumption
The agent writes JSON state checkpoints to `state/<run_id>.json` after every tool execution.

We simulated an unexpected system crash during Scenario S1 right after Tool #2 (`search_policy`):
- `run_agent` stopped with an error after step 2.
- `resume_agent("demo_s1_crash_resume")` reloaded state from disk and completed the workflow.
- **Tools executed across crash and resume**:
  1. `get_order` *(Before crash)*
  2. `search_policy` *(Before crash)*
  3. `calculate_refund` *(After resume)*
  4. `issue_refund` *(After resume)*
- **Outcome**: Exactly **4 unique tools executed** with zero duplicate calls, reaching status **`REFUND`** and refunding **INR 128.25**.

---

### 8. Architectural Comparison: ReAct vs Plan-and-Execute (245 Words)

A ReAct agent alternates between thought, tool selection, and environment observation at every step. In our Instamart refund workflow, ReAct completed standard approvals in 5 model calls (S1, S2, S3), short-circuited invalid requests in 2 to 4 calls (S6: 2 calls, 98 tokens; S5: 3 calls, 376 tokens), and consumed 962 tokens per successful resolution. 

In contrast, a plan-and-execute architecture prompts the language model once upfront to generate a full DAG plan (e.g., `[get_order -> search_policy -> calculate_refund -> issue_refund]`), executing all steps deterministically and replanning only when an execution exception occurs. For clean, predictable happy paths, plan-and-execute is more token-efficient, requiring only 1 initial planning call plus 1 final response call (~40% fewer prompt tokens).

However, customer support environments are dominated by dirty realities: missing orders, cancelled transactions, and ambiguous claims. When presented with an invalid order (`ORD-999999`, S6), ReAct halts immediately after `get_order` yields `NOT_FOUND`, taking just 2 calls. A plan-and-execute agent would attempt to run downstream policy lookups and refund calculations on null order objects, requiring expensive replanning cycles and error-handling overhead. Furthermore, executing irreversible financial writes (`issue_refund`) based on static upfront plans risks accidental ledger disbursements. Therefore, ReAct is the superior architectural choice for financial support automation, as step-by-step observation verification guarantees strict policy adherence before money leaves company accounts.

---

## 5. Unified One-Input Support Copilot and Web Dashboard

To prevent support agents from switching across separate tools for customer details, ticket triage, policy lookups, and refunds, we created the **Unified One-Input Support Copilot (Omnibox)** (`one_input_support.py` and `05_one_input_copilot.py`).

### 1. Smart Query Routing
A single input box handles customer questions, order complaints, policy searches, or raw tickets and routes them automatically:

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

1. **Customer 360 Lookup (`C\d{4}`)**:
   - Queries matching a customer ID (like `C0082`) retrieve account data from `data/instamart.db`, cluster details from `customer_segments.csv`, and real-time lapse probability from `lapse_model.joblib`.
2. **Autonomous Refund Agent (`ORD-\d{6}` or Refund Inquiries)**:
   - When an order ID or refund request is detected, the ReAct agent executes the required tools and evaluates financial rules before issuing funds.
3. **Policy Knowledge Assistant (RAG)**:
   - General questions (like *"What is the refund window for dairy items?"*) are routed to the TF-IDF search engine, returning grounded answers and citations.
4. **Privacy-Safe Support Ticket Triage**:
   - Customer complaints are first scrubbed of sensitive PII (`[PHONE]`, `[EMAIL]`, `[ORDER_ID]`), classified via few-shot prompting, and routed to the proper queue (`premium_desk`, `security_team`, `human_review`, or `support_queue`).

### 2. Full-Stack Web Application (`app.py`)
A lightweight, zero-dependency web application built on Python's standard `http.server` provides an interactive dark-mode dashboard at `http://127.0.0.1:8000`:
- **Omnibox Copilot Tab**: Single universal search bar with preset buttons for fast testing.
- **1. Data Foundation Tab**: Summary KPIs, city revenue charts, and basket distributions.
- **2. Customer Intelligence Tab**: Customer profile lookups and top 10 at-risk VIP list.
- **3. Support Triage Tab**: Live PII masking and automated ticket category classification.
- **4. Policy Knowledge Tab**: Live policy search and grounded answers with source cards.
- **5. Autonomous Agent Tab**: Step-by-step ReAct agent simulation and crash recovery demo.

---

## Limitations and Next Steps

1. **Whole-Window Segment Assignment**: The `is_premium` flag in `customer_segments.csv` was calculated across the full 6-month order window (including September). While the churn prediction model itself was trained strictly without leakage using pre-cutoff data, the retention list serves as an illustrative demonstration. Future versions should calculate customer clusters strictly as of the cutoff date.
2. **Dense Semantic Embeddings**: The policy search system uses TF-IDF word matching. While it achieved 100% accuracy on our test set, production systems with millions of users would benefit from dense vector embeddings (like BGE or Gemini Embeddings) and vector database indexes (like PostgreSQL `pgvector`) to understand synonyms and conversational phrasing better.
3. **Distributed Message Queues**: The ticket triage system currently processes queues locally in memory. For enterprise scale with high order volume, a message broker (like Apache Kafka or RabbitMQ) with worker pools and Redis state storage would provide greater reliability and throughput.
4. **Human Supervisory Console**: The interactive supervisor web console in `app.py` allows team leads to review high-value refunds (> INR 1000), inspect customer lifetime spending, and audit autonomous agent steps in real time.
