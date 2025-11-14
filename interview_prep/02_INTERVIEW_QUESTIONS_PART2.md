# Credit Card Fraud Detection - Interview Questions & Answers
# PART 2: ADVANCED & ELITE LEVEL

## 📋 Table of Contents
1. [ADVANCED Level Questions](#advanced-level-questions)
2. [ELITE Level Questions](#elite-level-questions)

---

# ADVANCED Level Questions

## Technical Deep Dive

### Q21: Explain the mathematics behind XGBoost's objective function.

**Answer:**

### Objective Function

XGBoost minimizes a regularized objective function:

```
Obj(θ) = L(θ) + Ω(θ)

where:
L(θ) = Σ loss(y_i, ŷ_i)  # Training loss
Ω(θ) = γT + (λ/2)Σw_j²   # Regularization term
```

### Loss Function (Binary Classification)

**Logistic Loss:**
```
L = -Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]

where:
p_i = 1 / (1 + e^(-f(x_i)))  # Sigmoid
f(x_i) = Σ f_k(x_i)          # Sum of all trees
```

### Gradient Boosting Mathematics

At iteration t, we want to add tree f_t to minimize:

```
Obj^(t) = Σ loss(y_i, ŷ^(t-1) + f_t(x_i)) + Ω(f_t)
```

**Taylor Expansion (2nd order approximation):**
```
loss(y_i, ŷ^(t-1) + f_t(x_i)) ≈ 
    loss(y_i, ŷ^(t-1)) + g_i × f_t(x_i) + (1/2) × h_i × f_t²(x_i)

where:
g_i = ∂loss/∂ŷ^(t-1)           # First gradient
h_i = ∂²loss/∂(ŷ^(t-1))²       # Second gradient (Hessian)
```

**Simplified Objective:**
```
Obj^(t) ≈ Σ[g_i × f_t(x_i) + (1/2) × h_i × f_t²(x_i)] + Ω(f_t)
```

### Tree Structure Scoring

For a tree with structure q and leaf weights w:

```
Obj^(t) = Σ[g_i × w_q(x_i) + (1/2) × h_i × w_q²(x_i)] + γT + (λ/2)Σw_j²

Regroup by leaf j:
Obj^(t) = Σ_j[(Σ_{i∈I_j} g_i) × w_j + (1/2)(Σ_{i∈I_j} h_i + λ) × w_j²] + γT
```

**Define:**
```
G_j = Σ_{i∈I_j} g_i  # Sum of gradients in leaf j
H_j = Σ_{i∈I_j} h_i  # Sum of hessians in leaf j
```

**Optimal leaf weight:**
```
w_j* = -G_j / (H_j + λ)
```

**Optimal objective value:**
```
Obj* = -(1/2) Σ_j [G_j² / (H_j + λ)] + γT
```

### Split Finding

**Gain from splitting:**
```
Gain = (1/2) × [G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L + G_R)²/(H_L + H_R + λ)] - γ

where:
L = left child node
R = right child node
```

**Algorithm:**
1. For each feature
2. For each split point
3. Calculate Gain
4. Choose split with maximum Gain
5. If Gain > 0, make the split

**γ (gamma) role:**
- Minimum gain required for split
- Pruning parameter
- Larger γ → more conservative → less splits

### Why 2nd Order?

**1st order (Gradient Descent):**
```
f^(t) = f^(t-1) - η × ∇f
```

**2nd order (Newton's Method):**
```
f^(t) = f^(t-1) - η × ∇f / ∇²f
```

**Benefits:**
- Faster convergence
- Better approximation of optimal direction
- More accurate step size

### Practical Example

**Sample at leaf:**
```
3 training samples fall in this leaf
g = [0.2, -0.3, 0.5]  # gradients
h = [0.8, 0.7, 0.9]   # hessians
λ = 1.0

G = 0.2 + (-0.3) + 0.5 = 0.4
H = 0.8 + 0.7 + 0.9 = 2.4

Optimal weight: w* = -0.4 / (2.4 + 1.0) = -0.118

Contribution to objective: -0.5 × (0.4² / 3.4) = -0.024
```

---

### Q22: How would you handle concept drift in production?

**Answer:**

**Concept Drift** = Statistical properties of target variable change over time.

### Types of Drift

**1. Sudden Drift**
```
Fraud patterns change overnight
Example: New fraud technique discovered
Detection: Sharp drop in model performance
```

**2. Gradual Drift**
```
Slow change over months
Example: Shift in customer behavior
Detection: Slow performance degradation
```

**3. Recurring Drift**
```
Seasonal patterns
Example: Holiday shopping fraud spikes
Detection: Periodic performance changes
```

**4. Feature Drift**
```
Input distribution changes
Example: More online transactions post-COVID
Detection: Feature statistics shift
```

### Detection Strategies

**1. Performance Monitoring**
```python
# Track metrics over time
daily_metrics = {
    'date': today,
    'accuracy': model_accuracy,
    'recall': model_recall,
    'precision': model_precision,
    'f1': model_f1,
    'auc': model_auc
}

# Alert if drop > 5%
if current_recall < baseline_recall * 0.95:
    trigger_retraining()
```

**2. Statistical Tests**
```python
from scipy.stats import ks_2samp

# Kolmogorov-Smirnov test for distribution shift
for feature in features:
    stat, p_value = ks_2samp(
        training_data[feature],
        production_data[feature]
    )
    if p_value < 0.05:
        print(f"Drift detected in {feature}")
```

**3. Population Stability Index (PSI)**
```python
def calculate_psi(expected, actual, bins=10):
    expected_percents = expected.value_counts(bins=bins, normalize=True)
    actual_percents = actual.value_counts(bins=bins, normalize=True)
    
    psi = sum(
        (actual_percents - expected_percents) * 
        np.log(actual_percents / expected_percents)
    )
    return psi

# PSI < 0.1: No change
# PSI 0.1-0.25: Moderate change
# PSI > 0.25: Significant change (retrain!)
```

**4. Prediction Distribution**
```python
# Monitor fraud probability distribution
current_distribution = get_prediction_distribution()
if wasserstein_distance(baseline_dist, current_dist) > threshold:
    alert_drift()
```

### Mitigation Strategies

**1. Scheduled Retraining**
```python
# Retrain monthly on recent data
@monthly_schedule
def retrain_model():
    recent_data = get_last_n_months(data, n=6)
    new_model = train_xgboost(recent_data)
    
    # A/B test before deployment
    if new_model.score > current_model.score:
        deploy_model(new_model)
```

**2. Online Learning**
```python
# Incremental updates
from river import ensemble

online_model = ensemble.AdaptiveRandomForestClassifier()

# Update with each new labeled transaction
for transaction, label in stream:
    prediction = online_model.predict_one(transaction)
    online_model.learn_one(transaction, label)
```

**3. Ensemble with Time Decay**
```python
# Weighted ensemble of models
predictions = []
weights = []

for model, trained_date in model_history:
    age_days = (today - trained_date).days
    weight = np.exp(-age_days / decay_param)  # Exponential decay
    
    pred = model.predict_proba(X)
    predictions.append(pred * weight)
    weights.append(weight)

final_pred = sum(predictions) / sum(weights)
```

**4. Active Learning**
```python
# Focus on uncertain predictions
def select_for_labeling(predictions, confidence_threshold=0.3):
    uncertain = [
        (idx, pred) for idx, pred in enumerate(predictions)
        if 0.3 < pred < 0.7  # Neither clearly fraud nor legitimate
    ]
    return uncertain

# Get expert labels for uncertain cases
# Add to retraining set
```

**5. Shadow Mode Deployment**
```python
# Run new model in parallel
production_pred = production_model.predict(transaction)
shadow_pred = new_model.predict(transaction)

# Log both predictions
log_predictions(transaction_id, production_pred, shadow_pred)

# Compare performance over time
if shadow_model_performance > production_model_performance:
    promote_shadow_to_production()
```

### Monitoring Dashboard

**Key Metrics to Track:**

```python
metrics_dashboard = {
    'Model Performance': {
        'daily_accuracy': timeseries_plot,
        'recall_by_amount': stratified_plot,
        'precision_trend': trend_line
    },
    
    'Feature Distribution': {
        'transaction_amount': histogram,
        'distance_from_home': distribution_plot,
        'transaction_hour': heatmap
    },
    
    'Prediction Distribution': {
        'fraud_probability': kde_plot,
        'prediction_confidence': histogram
    },
    
    'Business Metrics': {
        'false_positive_rate': gauge,
        'fraud_detection_rate': gauge,
        'avg_investigation_time': timeseries
    }
}
```

### My Production Strategy

**Phase 1: Monitoring (Weeks 1-4)**
```
- Deploy model with extensive logging
- Track all metrics daily
- Establish baseline performance
```

**Phase 2: Detection (Ongoing)**
```
- Statistical tests weekly
- Performance dashboards real-time
- Alert on significant changes
```

**Phase 3: Adaptation (As needed)**
```
- Scheduled retraining monthly
- Emergency retraining if drift detected
- A/B testing before full deployment
```

---

### Q23: How would you explain a prediction to a non-technical stakeholder?

**Answer:**

### Challenge

XGBoost is a "black box" - not trivially interpretable like linear models.

### Explanation Techniques

**1. Feature Importance (Global)**

```python
import matplotlib.pyplot as plt

# Get feature importance
importance = model.feature_importances_
features = X.columns

# Plot
plt.barh(features, importance)
plt.xlabel('Importance Score')
plt.title('Which Features Matter Most?')
```

**Explanation to Stakeholder:**
"Think of this as a ranking of what the model pays attention to. Transaction amount is most important - larger amounts are more suspicious. Distance from home is second - transactions far away are risky."

**2. SHAP Values (Individual Predictions)**

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model)

# Explain single prediction
shap_values = explainer.shap_values(transaction)

# Waterfall plot
shap.waterfall_plot(shap.explanation(shap_values[0]))
```

**Explanation:**
"This shows why the model flagged this specific transaction:
- Base rate: 1% of transactions are fraud (starting point)
- **Amount $5000**: +40% (large amount increases risk)
- **Distance 500km**: +25% (far from home)
- **Time 2AM**: +15% (unusual hour)
- **Weekend**: +5%
- **Category 'online'**: +10%
- **Final probability**: 95% fraud"

**3. Decision Path (Tree-based)**

```python
from sklearn.tree import plot_tree

# Show decision path
tree = model.get_booster().trees_to_dataframe()
path = get_decision_path(tree, transaction)
```

**Explanation:**
"The model makes a series of yes/no questions:
1. Is amount > $1000? YES
2. Is distance > 100km? YES
3. Is time between 10PM-6AM? YES
4. Conclusion: Likely fraud (95% confidence)"

**4. Similar Cases**

```python
# Find similar historical transactions
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5)
nn.fit(historical_transactions)
similar = nn.kneighbors(transaction)

print("Similar past transactions:")
for idx in similar:
    print(f"Transaction {idx}: {label[idx]} (fraud/legit)")
```

**Explanation:**
"Looking at 5 similar past transactions:
- 4 were confirmed frauds
- 1 was legitimate
This reinforces the model's prediction."

### Real Example Walkthrough

**Transaction Details:**
```
Amount: $3,500
Time: 2:30 AM
Location: 800km from home
Category: Electronics
Customer age: 28
Weekend: Yes
```

**Model Output:**
```
Fraud Probability: 87%
Risk Level: HIGH
```

**Explanation to Stakeholder:**

**Version 1: Non-Technical (Bank Manager)**
```
"This transaction has several red flags:

1. **Large Amount** ($3,500): Much higher than customer's typical $80 transactions
2. **Unusual Time** (2:30 AM): Customer rarely shops at night
3. **Wrong Location** (800km away): Customer is in New York, but transaction in Florida
4. **High-Risk Category** (Electronics): Often targeted by fraudsters

Our AI model learned from 500,000 past transactions. When it sees this combination of factors, it's right 98% of the time.

**Recommendation**: Block transaction and call customer to verify."
```

**Version 2: Semi-Technical (Risk Analyst)**
```
"The model assigned 87% fraud probability based on:

Risk Factors (SHAP values):
- Amount deviation: +35% (3.5 std from customer mean)
- Geographic anomaly: +28% (velocity 400km/h impossible)
- Temporal pattern: +15% (outside normal 7AM-10PM window)
- Category risk: +9% (electronics fraud rate: 3.2% vs avg 1%)

Mitigating factors:
- Known device: -5%

Net risk: 82% → 87% after model ensemble

Confidence: High (low variance across 100 trees)"
```

**Version 3: Technical (Data Scientist)**
```
XGBoost Explanation:

Feature Contributions (SHAP):
- amt: +0.42 (log-odds)
- distance_from_home: +0.35
- is_night: +0.18
- velocity: +0.22 (spatial-temporal impossible)
- cat_electronics: +0.12
- weekend: +0.06
- age_group: -0.05
- known_merchant: -0.08

Base rate: log(0.01/0.99) = -4.595
Sum contributions: +1.22
Final log-odds: -3.375
Probability: 1/(1+e^3.375) = 0.87

Model confidence via prediction variance across 100 trees: σ = 0.02
```

### Visualization for Stakeholders

**Dashboard Elements:**

```python
# 1. Risk Gauge
st.plotly_chart(
    go.Indicator(
        value=87,
        title="Fraud Risk",
        gauge={'axis': {'range': [0, 100]}}
    )
)

# 2. Factor Breakdown
st.bar_chart({
    'Large Amount': 35,
    'Wrong Location': 28,
    'Night Time': 15,
    'High-Risk Category': 9
})

# 3. Similar Cases
st.write("Of 100 similar past transactions:")
st.metric("Were Fraud", "94")
st.metric("Were Legitimate", "6")

# 4. Recommendation
if risk > 70:
    st.error("🛑 BLOCK - Call customer")
elif risk > 30:
    st.warning("⚠️ REVIEW - Additional verification")
else:
    st.success("✅ APPROVE - Low risk")
```

### Key Principles

**1. Use Analogies**
"Like a doctor diagnosing illness - looks at multiple symptoms together"

**2. Show Confidence**
"Model is 98% accurate when it says 'high risk'"

**3. Be Transparent**
"No model is perfect - this is why we have human review for medium-risk cases"

**4. Quantify Impact**
"By catching this fraud, we save the customer $3,500 and prevent identity theft"

**5. Provide Context**
"Out of 10,000 daily transactions, model flags ~100. Of those, 94 are actual frauds."

---

### Q24: What would you do differently if dataset was 10x larger?

**Answer:**

### Current: 5,557 samples (after sampling)
### Scenario: 55,570 samples or full 555,718

### Challenges with Larger Data

**1. Memory Constraints**
```python
# Current: Fits in RAM
df = pd.read_csv('data.csv')  # ~50MB

# 10x larger: May not fit
df = pd.read_csv('large_data.csv')  # ~500MB (still ok)

# 100x larger: Definitely won't fit
df = pd.read_csv('huge_data.csv')  # ~5GB (RAM issue)
```

**2. Training Time**
```
Current: 45 seconds
10x data: ~7-8 minutes (not linear due to sorting)
100x data: 1-2 hours
```

**3. Feature Engineering**
```python
# Rolling windows become expensive
df['trans_last_24h'] = df.groupby('cc_num').rolling('24H').count()
# Time: O(n log n) - becomes bottleneck
```

### Solutions for 10x Larger Data

**1. Chunk Processing**

```python
# Read in chunks
chunk_size = 10000
chunks = pd.read_csv('large_data.csv', chunksize=chunk_size)

# Process each chunk
processed_chunks = []
for chunk in chunks:
    chunk_processed = engineer_features(chunk)
    processed_chunks.append(chunk_processed)

df = pd.concat(processed_chunks)
```

**2. Dask for Parallel Processing**

```python
import dask.dataframe as dd

# Lazy evaluation
ddf = dd.read_csv('large_data.csv')

# Parallel operations
ddf['age'] = ddf['dob'].apply(lambda x: 2024 - x)
ddf['distance'] = ddf.apply(calculate_distance, axis=1)

# Compute when needed
result = ddf.compute()
```

**3. Sampling Strategies**

```python
# Stratified sampling to reduce size
from sklearn.model_selection import train_test_split

# Keep all frauds, sample legitimates
frauds = df[df['is_fraud'] == 1]
legits = df[df['is_fraud'] == 0]

# Sample 10% of legit
legit_sample = legits.sample(frac=0.1, random_state=42)

# Combine
df_sampled = pd.concat([frauds, legit_sample])

# Now 90% smaller but keeps all important cases
```

**4. XGBoost External Memory**

```python
# Use external memory mode
dtrain = xgb.DMatrix('train.buffer')
dtest = xgb.DMatrix('test.buffer')

params = {
    'tree_method': 'hist',      # Histogram-based (faster)
    'max_bin': 256,             # Reduce bins
    'grow_policy': 'lossguide'  # Better for large data
}

model = xgb.train(params, dtrain, external_memory=True)
```

**5. Distributed Training**

```python
# Dask-XGBoost for distributed training
import dask_xgboost as dxgb
from dask.distributed import Client

client = Client()  # Start cluster

# Distributed training
dxgb_model = dxgb.train(
    client,
    params,
    ddf,
    y,
    num_boost_round=100
)
```

**6. Feature Selection More Aggressive**

```python
# With more data, can be more selective
# Keep only top 10 features instead of 14

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Reduces memory and training time
```

**7. Database Integration**

```python
# Don't load all in memory
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://...')

# Query in batches
query = """
    SELECT * FROM transactions
    WHERE trans_date >= '2023-01-01'
    AND is_fraud = 1
    UNION ALL
    SELECT * FROM transactions
    WHERE trans_date >= '2023-01-01'
    AND is_fraud = 0
    ORDER BY RANDOM()
    LIMIT 100000
"""

df = pd.read_sql(query, engine)
```

**8. Incremental Feature Engineering**

```python
# Pre-compute expensive features
def precompute_features():
    conn = sqlite3.connect('features.db')
    
    # Compute distance once, store in DB
    cursor.execute("""
        CREATE TABLE feature_distance AS
        SELECT 
            trans_id,
            haversine(lat, long, merch_lat, merch_long) as distance
        FROM transactions
    """)
    
    conn.commit()

# Later, just join
df = pd.read_sql("""
    SELECT t.*, f.distance
    FROM transactions t
    JOIN feature_distance f ON t.trans_id = f.trans_id
""", conn)
```

**9. Approximate Algorithms**

```python
# For KNN-based features, use approximate
from sklearn.neighbors import LSHForest  # Locality Sensitive Hashing

# Fast approximate nearest neighbors
lsh = LSHForest(n_estimators=20, random_state=42)
lsh.fit(X_train)

# Much faster than exact KNN
distances, indices = lsh.kneighbors(X_test, n_neighbors=5)
```

**10. Monitoring Memory**

```python
import psutil
import gc

def check_memory():
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory: {mem_mb:.2f} MB")
    
    if mem_mb > 8000:  # 8GB threshold
        print("Triggering garbage collection...")
        gc.collect()

# Check periodically
check_memory()
```

### Algorithm Choices Change

**For 10x larger data:**

```python
# XGBoost still good, but consider:

# 1. LightGBM (faster on large data)
from lightgbm import LGBMClassifier
model = LGBMClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1
)
# Histogram-based, faster than XGBoost on >100k samples

# 2. CatBoost (handles categorical well)
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=100,
    depth=4,
    learning_rate=0.1
)

# 3. Linear models for baseline
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    C=1.0,
    solver='saga',  # Supports sparse data
    max_iter=100
)
# Much faster, good baseline
```

### Hardware Considerations

**Current (Small Data):**
- CPU: Sufficient
- RAM: 8GB enough
- Time: Minutes

**10x Larger:**
- CPU: 16+ cores helpful
- RAM: 16-32GB recommended
- GPU: Optional (for neural networks)
- Time: Hours

**100x Larger:**
- Cluster: Distributed computing
- RAM: 64GB+ or distributed
- GPU: Beneficial for deep learning
- Storage: SSD for fast I/O

### What I Would Do

**Approach:**

1. **Profile current bottlenecks**
```python
import cProfile
cProfile.run('train_model()')
# Identify slow functions
```

2. **Optimize bottlenecks first**
- Usually feature engineering
- Then training

3. **Sample intelligently**
- Keep all frauds
- Representative sample of legitimate

4. **Use better algorithms**
- LightGBM over XGBoost for speed
- Histogram-based methods

5. **Distributed if needed**
- Dask for preprocessing
- Spark for very large scale

6. **Cache intermediate results**
- Feature matrices
- Avoid recomputation

**Trade-off:**
With more data, can potentially get better accuracy, but need to balance with:
- Training time
- Resource costs
- Diminishing returns (99.9% → 99.95% may not be worth 10x compute)

---

(continuing in ELITE section...)
