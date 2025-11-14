# Credit Card Fraud Detection - Mathematics Deep Dive

## 📐 Complete Mathematical Foundation

---

## Table of Contents
1. [Probability & Statistics Foundations](#probability--statistics-foundations)
2. [Loss Functions](#loss-functions)
3. [Evaluation Metrics Mathematics](#evaluation-metrics-mathematics)
4. [Tree-Based Algorithms](#tree-based-algorithms)
5. [Gradient Boosting Mathematics](#gradient-boosting-mathematics)
6. [Optimization](#optimization)
7. [Distance Metrics](#distance-metrics)

---

# Probability & Statistics Foundations

## Basic Probability

### Probability Definition
```
P(A) = Number of favorable outcomes / Total possible outcomes

Properties:
- 0 ≤ P(A) ≤ 1
- P(A) + P(A') = 1
- P(∅) = 0, P(Ω) = 1
```

### Conditional Probability
```
P(A|B) = P(A ∩ B) / P(B)

In fraud detection:
P(Fraud | High_Amount) = P(Fraud AND High_Amount) / P(High_Amount)
```

### Bayes' Theorem
```
P(A|B) = [P(B|A) × P(A)] / P(B)

Fraud detection application:
P(Fraud | Transaction_Features) = 
    [P(Features | Fraud) × P(Fraud)] / P(Features)

Example:
P(Fraud) = 0.01 (base rate)
P(Large_Amount | Fraud) = 0.7
P(Large_Amount) = 0.1

P(Fraud | Large_Amount) = (0.7 × 0.01) / 0.1 = 0.07 = 7%
```

## Distributions

### Bernoulli Distribution
```
Binary outcome: Fraud (1) or Not Fraud (0)

P(X = 1) = p
P(X = 0) = 1 - p

Mean: μ = p
Variance: σ² = p(1-p)

In our case: p ≈ 0.01 (1% fraud rate)
```

### Normal Distribution
```
PDF: f(x) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))

Properties:
- Mean = Median = Mode = μ
- 68% within μ ± σ
- 95% within μ ± 2σ
- 99.7% within μ ± 3σ

Used for: Feature scaling, anomaly detection
```

## Statistical Tests

### Hypothesis Testing
```
H₀: Null hypothesis (no fraud)
H₁: Alternative hypothesis (fraud)

Type I Error (False Positive): 
- Reject H₀ when it's true
- Flag legitimate transaction as fraud
- α = P(Type I Error)

Type II Error (False Negative):
- Accept H₀ when it's false
- Miss actual fraud
- β = P(Type II Error)

Power = 1 - β = Ability to detect fraud when present
```

### p-value
```
p-value = Probability of observing data this extreme if H₀ is true

Decision rule:
- If p-value < α (typically 0.05): Reject H₀
- If p-value ≥ α: Fail to reject H₀

Example:
Transaction has features with p-value = 0.001
Very unlikely under "legitimate" hypothesis
→ Likely fraud
```

---

# Loss Functions

## Binary Cross-Entropy Loss

### Definition
```
L = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

where:
y ∈ {0, 1}: True label
ŷ ∈ [0, 1]: Predicted probability
```

### Derivation
```
From maximum likelihood estimation:

Likelihood: L(θ) = ∏ P(y_i | x_i, θ)

For Bernoulli:
L(θ) = ∏ [p_i^y_i × (1-p_i)^(1-y_i)]

Log-likelihood:
log L(θ) = Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]

Negative log-likelihood (to minimize):
-log L(θ) = -Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]

This is cross-entropy loss!
```

### Properties
```
When y = 1 (actual fraud):
L = -log(ŷ)
- ŷ = 1 (correct): L = 0 (no penalty)
- ŷ = 0.5: L = 0.693
- ŷ = 0.01 (wrong): L = 4.605 (high penalty)

When y = 0 (actual legitimate):
L = -log(1-ŷ)
- ŷ = 0 (correct): L = 0
- ŷ = 0.5: L = 0.693
- ŷ = 0.99 (wrong): L = 4.605
```

### Gradients
```
∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)

For sigmoid activation σ(z) = 1/(1+e^(-z)):
∂L/∂z = ŷ - y

Elegant result! Gradient = Prediction Error
```

## Weighted Cross-Entropy

### For Imbalanced Data
```
L = -[w₁ × y × log(ŷ) + w₀ × (1-y) × log(1-ŷ)]

where:
w₁ = weight for positive class (fraud)
w₀ = weight for negative class (legitimate)

Typically:
w₁ = n_negative / n_positive
w₁ = 99,000 / 1,000 = 99

Effect: Penalizes missing fraud 99x more than false alarm
```

## Focal Loss

### For Hard Examples
```
FL = -α × (1-ŷ)^γ × y × log(ŷ) - (1-α) × ŷ^γ × (1-y) × log(1-ŷ)

where:
α: Class balance weight
γ: Focusing parameter (typically 2)

Effect:
- Easy examples (ŷ close to y): Low loss
- Hard examples (ŷ far from y): High loss
- Focuses learning on difficult cases
```

---

# Evaluation Metrics Mathematics

## Confusion Matrix Metrics

### Basic Metrics
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)
- "When we predict fraud, how often are we right?"

Recall = TP / (TP + FN)
- "Of all frauds, how many did we catch?"

Specificity = TN / (TN + FP)
- "Of all legitimate, how many did we correctly identify?"
```

### F-Beta Score

```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)

Special cases:
- β = 1: F₁ Score (balanced)
- β = 2: F₂ Score (favor recall)
- β = 0.5: F₀.₅ Score (favor precision)

F₁ Score:
F₁ = 2PR / (P + R) = 2TP / (2TP + FP + FN)

Harmonic mean (not arithmetic):
- Arithmetic: (P + R) / 2
- Harmonic: 2 / (1/P + 1/R)

Why harmonic?
Penalizes extreme values:
- P=100%, R=10% → Arithmetic=55%, Harmonic=18%
- Shows poor balance better
```

### Matthews Correlation Coefficient

```
MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]

Range: [-1, 1]
- +1: Perfect prediction
- 0: Random
- -1: Perfect disagreement

Advantages:
- Balanced measure for imbalanced data
- Accounts for all four confusion matrix values
- Single number summary
```

## ROC and AUC

### ROC Curve Mathematics

```
TPR = TP / (TP + FN)  # True Positive Rate (Recall)
FPR = FP / (FP + TN)  # False Positive Rate

For each threshold t:
- Predict fraud if P(fraud) > t
- Calculate (FPR, TPR) pair
- Plot point

ROC = Plot of TPR vs FPR as threshold varies
```

### AUC Interpretation

```
AUC = ∫₀¹ TPR(FPR) d(FPR)

Probabilistic interpretation:
AUC = P(score(positive) > score(negative))

"Probability that a randomly chosen fraud transaction
has higher predicted probability than a randomly chosen
legitimate transaction"

Example:
AUC = 0.99 means 99% chance fraud scores higher than legit
```

### Calculation (Trapezoidal Rule)

```
AUC ≈ Σᵢ [(FPRᵢ₊₁ - FPRᵢ) × (TPRᵢ₊₁ + TPRᵢ) / 2]

Alternative (Mann-Whitney U):
AUC = (S₀ - n₀(n₀+1)/2) / (n₀ × n₁)

where:
S₀ = sum of ranks of positive samples
n₀ = number of positive samples
n₁ = number of negative samples
```

## Precision-Recall AUC

### Why for Imbalanced Data?

```
ROC can be optimistic for imbalanced data
PR curve is more informative

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

For imbalanced (1% fraud):
Even with 90% fraud caught (high TPR)
And 2% false positive rate (low FPR)
ROC looks great!

But:
Precision = 990 / (990 + 1980) = 33% (poor!)
PR curve shows this problem
```

### Average Precision

```
AP = Σₙ [Recall(n) - Recall(n-1)] × Precision(n)

Weighted average of precisions at each threshold
Weights = change in recall
```

---

# Tree-Based Algorithms

## Decision Tree Splitting

### Information Gain (ID3, C4.5)

```
Entropy: H(S) = -Σ pᵢ log₂(pᵢ)

For fraud detection (binary):
H(S) = -p(fraud) log₂ p(fraud) - p(legit) log₂ p(legit)

Example:
1000 frauds, 99,000 legit
p(fraud) = 0.01, p(legit) = 0.99

H(S) = -0.01 × log₂(0.01) - 0.99 × log₂(0.99)
     = -0.01 × (-6.64) - 0.99 × (-0.0145)
     = 0.0664 + 0.0144
     = 0.081

Low entropy = pure (all same class)
High entropy = mixed (50-50 is maximum)
```

### Information Gain

```
IG(S, A) = H(S) - Σ |Sᵥ|/|S| × H(Sᵥ)

where:
S: Dataset
A: Attribute to split on
Sᵥ: Subset where attribute = v

Example: Split on amount > $1000

Before split:
H(S) = 0.081

After split:
Left (≤ $1000): 98,000 legit, 200 fraud → H(S_L) = 0.032
Right (> $1000): 1,000 legit, 800 fraud → H(S_R) = 0.878

IG = 0.081 - [98,200/100,000 × 0.032 + 1,800/100,000 × 0.878]
   = 0.081 - [0.031 + 0.016]
   = 0.034

Choose split with maximum information gain
```

### Gini Impurity (CART)

```
Gini(S) = 1 - Σ pᵢ²

For binary:
Gini(S) = 1 - [p(fraud)² + p(legit)²]

Example:
p(fraud) = 0.01, p(legit) = 0.99
Gini = 1 - [0.01² + 0.99²]
     = 1 - [0.0001 + 0.9801]
     = 0.0198

Gini = 0: Pure node
Gini = 0.5: Maximum impurity (50-50)
```

### Gini Gain

```
Gini_Gain(S, A) = Gini(S) - Σ |Sᵥ|/|S| × Gini(Sᵥ)

Similar to information gain but uses Gini
Computationally simpler (no logarithms)
```

## Random Forest Mathematics

### Bootstrap Sampling

```
Given dataset D with n samples
Create m bootstrap samples:

For each bootstrap B₁, B₂, ..., Bₘ:
- Sample n times with replacement from D
- ~63.2% unique samples in each bootstrap
- ~36.8% out-of-bag samples

Proof:
P(sample not chosen in one draw) = (n-1)/n
P(sample not chosen in n draws) = [(n-1)/n]ⁿ
As n→∞: [(n-1)/n]ⁿ → e⁻¹ ≈ 0.368
```

### Feature Randomness

```
At each split, randomly select m features
Typically: m = √p (classification) or p/3 (regression)

where p = total number of features

Our case: p = 14, so m = √14 ≈ 4

This decorrelates trees
→ More diverse ensemble
→ Better generalization
```

### Ensemble Aggregation

```
Classification (majority vote):
ŷ = mode{h₁(x), h₂(x), ..., hₘ(x)}

Or soft voting (probabilities):
P(fraud|x) = (1/m) × Σᵢ Pᵢ(fraud|x)

Regression (average):
ŷ = (1/m) × Σᵢ hᵢ(x)
```

### Variance Reduction

```
Variance of ensemble:
Var(Ensemble) = ρσ² + [(1-ρ)/m] × σ²

where:
ρ = average correlation between trees
σ² = variance of individual tree
m = number of trees

As m → ∞:
Var(Ensemble) → ρσ²

Key insight:
- Uncorrelated trees (ρ=0): Variance → 0
- Perfectly correlated (ρ=1): No variance reduction
- Random Forest reduces correlation via:
  1. Bootstrap sampling
  2. Feature randomness
```

---

# Gradient Boosting Mathematics

## General Framework

### Additive Model

```
F(x) = Σₜ₌₁ᵀ fₜ(x)

where:
fₜ(x) = tree added at iteration t
T = total number of trees

Prediction: ŷ = F(x)
For classification: P(fraud) = σ(F(x)) where σ = sigmoid
```

### Objective Function

```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₜ Ω(fₜ)

where:
L = Loss function (e.g., log loss)
Ω = Regularization term

Ω(f) = γT + (λ/2) Σⱼ wⱼ²

T = number of leaves
wⱼ = weight of leaf j
γ, λ = regularization parameters
```

## Gradient Descent Perspective

### Steepest Descent in Function Space

```
At iteration t, we want to find fₜ that minimizes:

Obj^(t) = Σᵢ L(yᵢ, F^(t-1)(xᵢ) + fₜ(xᵢ)) + Ω(fₜ)

Gradient descent in function space:
fₜ(x) = -η × ∇_F L(y, F^(t-1)(x))

where η = learning rate
```

### Computing Gradients

```
For log loss: L = -[y log(p) + (1-y) log(1-p)]

where p = σ(F(x)) = 1/(1 + e^(-F(x)))

First derivative (gradient):
∂L/∂F = p - y

Second derivative (Hessian):
∂²L/∂F² = p(1-p)

For sample i:
gᵢ = ∂L/∂F|_{F=F^(t-1)(xᵢ)} = pᵢ - yᵢ
hᵢ = ∂²L/∂F²|_{F=F^(t-1)(xᵢ)} = pᵢ(1-pᵢ)
```

## XGBoost Specifics

### Taylor Expansion

```
Approximate loss around F^(t-1):

L(y, F^(t-1) + fₜ) ≈ L(y, F^(t-1)) + gfₜ + (1/2)hfₜ²

where:
g = ∂L/∂F|_{F^(t-1)}
h = ∂²L/∂F²|_{F^(t-1)}

This is 2nd order Taylor approximation
(Newton's method vs gradient descent)
```

### Optimal Weights

```
For tree structure q, optimal leaf weights:

wⱼ* = -Gⱼ / (Hⱼ + λ)

where:
Gⱼ = Σ_{i∈Iⱼ} gᵢ  (sum of gradients in leaf j)
Hⱼ = Σ_{i∈Iⱼ} hᵢ  (sum of hessians in leaf j)
λ = L2 regularization parameter
```

### Optimal Objective Value

```
Obj* = -(1/2) Σⱼ [Gⱼ²/(Hⱼ + λ)] + γT

This is the loss after finding optimal weights
Used to compare different tree structures
```

### Split Finding

```
For feature j, split point s:
Gain = (1/2) × [G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L + G_R)²/(H_L + H_R + λ)] - γ

where:
G_L = Σ_{i∈Left} gᵢ
G_R = Σ_{i∈Right} gᵢ
H_L = Σ_{i∈Left} hᵢ
H_R = Σ_{i∈Right} hᵢ

Algorithm:
1. For each feature
2. Sort feature values
3. Scan through split points
4. Calculate gain for each
5. Choose best (feature, split point) combination
6. If gain > 0, make the split
```

### Learning Rate

```
Update rule:
F^(t)(x) = F^(t-1)(x) + η × fₜ(x)

where η ∈ (0, 1] is learning rate

Effect of η:
- η = 1: Full update (aggressive)
- η = 0.1: Cautious update (typical)
- η = 0.01: Very conservative

Trade-off:
- Smaller η: More trees needed, but better generalization
- Larger η: Fewer trees needed, but risk of overfitting

Analogy:
Like step size in gradient descent
Smaller steps = more careful but slower
```

---

# Optimization

## Gradient Descent

### Basic Form

```
θ^(t+1) = θ^(t) - η × ∇_θ J(θ^(t))

where:
θ = parameters
η = learning rate
J(θ) = objective function
∇_θ J = gradient
```

### Variants

**1. Batch Gradient Descent**
```
∇J(θ) = (1/n) Σᵢ ∇_θ L(yᵢ, f(xᵢ; θ))

Uses all training samples
Stable but slow for large datasets
```

**2. Stochastic Gradient Descent (SGD)**
```
∇J(θ) ≈ ∇_θ L(yᵢ, f(xᵢ; θ))  # single sample

Faster, noisier
Good for large datasets
```

**3. Mini-Batch Gradient Descent**
```
∇J(θ) ≈ (1/b) Σᵢ∈B ∇_θ L(yᵢ, f(xᵢ; θ))

where B = mini-batch, |B| = b

Balance between batch and SGD
Most commonly used
```

## Newton's Method

### Update Rule

```
θ^(t+1) = θ^(t) - H⁻¹ × ∇J

where:
H = Hessian matrix (2nd derivatives)
Hᵢⱼ = ∂²J/∂θᵢ∂θⱼ

Advantage: Accounts for curvature
Faster convergence near optimum
```

### Why XGBoost Uses 2nd Order

```
Gradient Descent (1st order):
- Direction: -∇J
- Step size: η (fixed)

Newton's Method (2nd order):
- Direction: -H⁻¹∇J
- Step size: Adaptive based on curvature

Result:
- Faster convergence
- Better handling of ill-conditioned problems
- More accurate near optimum
```

---

# Distance Metrics

## Haversine Formula (Great Circle Distance)

### Derivation

```
Two points on sphere:
P₁ = (lat₁, lon₁)
P₂ = (lat₂, lon₂)

Convert to radians:
φ₁ = lat₁ × π/180
φ₂ = lat₂ × π/180
Δφ = φ₂ - φ₁
Δλ = (lon₂ - lon₁) × π/180

Haversine formula:
a = sin²(Δφ/2) + cos(φ₁) × cos(φ₂) × sin²(Δλ/2)
c = 2 × atan2(√a, √(1-a))
d = R × c

where:
R = Earth's radius
  = 6,371 km (mean)
  = 3,959 miles
```

### Why Haversine?

```
Euclidean distance:
d = √[(x₂-x₁)² + (y₂-y₁)²]

Problems:
1. Assumes flat Earth
2. Lat/long not in distance units
3. Inaccurate for long distances

Haversine:
✓ Accounts for spherical geometry
✓ Accurate for any distance
✓ Standard in geospatial
```

### Example Calculation

```
Customer: NYC (40.7128°N, 74.0060°W)
Merchant: LA (34.0522°N, 118.2437°W)

φ₁ = 40.7128 × π/180 = 0.7103 rad
φ₂ = 34.0522 × π/180 = 0.5942 rad
Δφ = -0.1161 rad
Δλ = -44.2377 × π/180 = -0.7720 rad

a = sin²(-0.1161/2) + cos(0.7103) × cos(0.5942) × sin²(-0.7720/2)
  = 0.0034 + 0.7623 × 0.8321 × 0.1335
  = 0.0034 + 0.0847
  = 0.0881

c = 2 × atan2(√0.0881, √0.9119)
  = 2 × atan2(0.2968, 0.9549)
  = 2 × 0.3014
  = 0.6028

d = 6,371 × 0.6028 = 3,841 km ≈ 2,387 miles

Verification: Actual distance NYC-LA ≈ 3,940 km
Our calculation: 3,841 km (97% accurate!)
```

## K-Nearest Neighbors Distance

### Euclidean Distance

```
d(x, y) = √[Σᵢ (xᵢ - yᵢ)²]

For fraud features:
d(transaction₁, transaction₂) = √[
    (amt₁ - amt₂)² +
    (distance₁ - distance₂)² +
    (age₁ - age₂)² +
    ...
]

Problem: Different scales!
amt: [0, 10000]
age: [18, 90]
is_night: [0, 1]

Solution: Standardization
xᵢ' = (xᵢ - μᵢ) / σᵢ
```

### Manhattan Distance

```
d(x, y) = Σᵢ |xᵢ - yᵢ|

Less sensitive to outliers than Euclidean
```

### Cosine Similarity

```
similarity(x, y) = (x · y) / (||x|| × ||y||)
                 = Σᵢ(xᵢyᵢ) / (√Σᵢxᵢ² × √Σᵢyᵢ²)

distance = 1 - similarity

Used for: High-dimensional sparse data
```

---

## Summary

This mathematical foundation covers:
✓ Probability theory for fraud detection  
✓ Loss functions and their gradients  
✓ All evaluation metrics with derivations  
✓ Tree splitting mathematics  
✓ Complete gradient boosting framework  
✓ XGBoost optimization details  
✓ Distance calculations  

**Master these concepts to confidently explain any aspect of the project!**
