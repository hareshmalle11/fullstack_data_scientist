# Logistic Regression - Loan Default Prediction

## 1. Model Name

- **Full Name:** Logistic Regression
- **Category:** Classification (Binary & Multiclass)
- **Learning Type:** Supervised Learning
- **Output Type:** Class probabilities / Labels

---

## 2. Problem It Solves

- **Task:** Predicting binary (or multiclass) outcomes based on input features
- **Use Case:** Predicting loan default (Yes/No) based on applicant characteristics
- **Why We Need It:**
  - Simple probabilistic classifier for binary classification
  - Outputs interpretable probability scores (0-1)
  - Strong statistical foundation
  - Fast training and prediction
  - Foundation for deep learning

---

## 3. Intuition

### Simple Explanation
Logistic Regression finds a **probability boundary** that separates two classes. Instead of drawing a straight line like Linear Regression, it creates an **S-shaped curve** that smoothly transitions between 0 (Class A) and 1 (Class B).

### Real-World Analogy
Imagine a loan officer gradually increasing confidence in "default" as applicant risk factors increase:
- **Low risk:** Probability of default = 0.05 (5%)
- **Medium risk:** Probability of default = 0.50 (50%)
- **High risk:** Probability of default = 0.95 (95%)

The S-curve captures this gradual increase in probability.

### Why Not Linear Regression for Classification?
```
Linear Regression:
- Can predict values outside [0,1]: 0.6, -0.2, 1.5
- Nonsensical for probabilities
- Can violate probability bounds

Logistic Regression:
- Always predicts between 0 and 1
- Natural probability interpretation
- Uses S-shaped curve (sigmoid)
```

---

## 4. Mathematical Foundation

### Core Formula: The Logistic Function (Sigmoid)

$$P(y=1|x) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$$

Simplified notation:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z = \beta_0 + \beta_1 x$

### Explanation of Terms

| Symbol | Meaning |
|--------|---------|
| $p$ | Probability of class 1 (default) |
| $\sigma(z)$ | Sigmoid function (S-shaped curve) |
| $z$ | Linear combination of features |
| $\beta_0$ | Intercept (log-odds at x=0) |
| $\beta_1$ | Slope (change in log-odds per unit x) |
| $x$ | Input feature |
| $e$ | Euler's number (≈2.718) |

### The Sigmoid Function Visualization

```
Probability
    1.0 |         _______________
        |        /
   0.75 |      /
        |    /
   0.50 |  / ← Inflection point
        | /
   0.25 |/
        |
    0.0 |________________
       -4   -2    0    2    4
                  z
```

### Why Sigmoid?
1. **Bounded:** Always outputs between 0 and 1
2. **Differentiable:** Can calculate gradients
3. **Smooth:** No sharp boundaries
4. **Probabilistic:** Natural interpretation as probability

### Example with Loan Data
```
Feature: Credit Score = 700
β₀ = -10, β₁ = 0.02

z = -10 + 0.02(700) = -10 + 14 = 4

p = 1/(1 + e^(-4)) = 1/(1 + 0.0183) ≈ 0.982

Interpretation: 98.2% probability of loan default
```

---

## 5. Objective Function

### Goal: Maximize Likelihood
Logistic Regression maximizes **likelihood** (minimizes **negative log-likelihood**):

$$L(\beta) = \prod_{i=1}^{n} p_i^{y_i} (1-p_i)^{1-y_i}$$

where:
- $p_i$ = predicted probability for sample i
- $y_i$ = actual label (0 or 1)
- If $y_i = 1$: contributes $p_i$ (want $p_i$ close to 1)
- If $y_i = 0$: contributes $(1-p_i)$ (want $p_i$ close to 0)

### Log-Likelihood (easier to compute)

$$LL(\beta) = \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

### Loss Function (Negative Log-Likelihood / Binary Cross-Entropy)

$$J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

### Why This Loss?
- $\log(p_i)$ penalizes wrong predictions exponentially
- Large penalties for confident wrong predictions
- Natural probability interpretation
- Foundation of information theory

---

## 6. Loss Function Derivation

### Binary Cross-Entropy Explained

$$J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

### Case Analysis

#### When $y_i = 1$ (Actually Default)
$$Loss_i = -\log(p_i)$$
- If $p_i = 0.99$ (confident correct): Loss ≈ 0.01 (small)
- If $p_i = 0.50$ (uncertain): Loss ≈ 0.693 (medium)
- If $p_i = 0.01$ (confident wrong): Loss ≈ 4.60 (large)

#### When $y_i = 0$ (Actually No Default)
$$Loss_i = -\log(1-p_i)$$
- If $p_i = 0.01$ (confident correct): Loss ≈ 0.01 (small)
- If $p_i = 0.50$ (uncertain): Loss ≈ 0.693 (medium)
- If $p_i = 0.99$ (confident wrong): Loss ≈ 4.60 (large)

### Gradient Computation

Taking derivative of loss with respect to $\beta_j$:

$$\frac{\partial J}{\partial \beta_j} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) x_{ij}$$

**Key insight:** Gradient is proportional to prediction error $(p_i - y_i)$!

---

## 7. Optimization Method

### Gradient Descent
Unlike Linear Regression, Logistic Regression **requires iterative optimization** (no closed-form solution):

$$\beta_{new} = \beta_{old} - \alpha \frac{\partial J}{\partial \beta}$$

where $\alpha$ = learning rate

### Update Rule for Each Iteration

$$\beta_j := \beta_j - \alpha \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) x_{ij}$$

### Optimization Algorithms Used

| Algorithm | Characteristics |
|-----------|---|
| **Newton-Raphson** | Quadratic convergence, default in many libraries |
| **Gradient Descent** | Slow, requires learning rate tuning |
| **Stochastic GD** | Fast, noisy but good for large datasets |
| **L-BFGS** | Quasi-Newton, fast convergence |
| **SAG** | Variance-reduced, good balance |

### Convergence
Optimization stops when:
- Gradient close to zero
- Loss stops decreasing
- Maximum iterations reached
- Accuracy threshold met

---

## 8. Training Workflow

```
┌──────────────────────────────────┐
│  Load Dataset                    │
│  (Loan Default Data)             │
│  Features: Credit Score, Income, │
│  Debt Ratio, Employment Years    │
│  Target: Default (Yes/No)        │
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  Data Preprocessing              │
│  - Handle missing values         │
│  - Encode categorical variables  │
│  - Feature scaling (important!)  │
│  - Address class imbalance       │
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  Split Data                      │
│  - Train: 70%                    │
│  - Test: 30%                     │
│  - Stratified split (balanced)   │
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  Initialize Parameters           │
│  - β₀ = 0, β₁ = 0, ..., βₙ = 0  │
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  Iterative Optimization          │
│  ┌─────────────────────────────┐ │
│  │ 1. Predict: p = 1/(1+e^-z) │ │
│  │ 2. Compute loss: -log(p)   │ │
│  │ 3. Compute gradient        │ │
│  │ 4. Update β                │ │
│  │ 5. Check convergence       │ │
│  └─────────────────────────────┘ │
│  Repeat until convergence        │
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  Make Predictions                │
│  - Get probability: p = σ(z)    │
│  - Set threshold (usually 0.5)   │
│  - Class = 1 if p > threshold   │
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  Evaluate Performance            │
│  - Accuracy, Precision, Recall   │
│  - F1, ROC-AUC, Confusion Matrix │
│  - Compare train vs test         │
└──────────────────────────────────┘
```

---

## 9. Worked Numerical Example

### Simplified Dataset: Credit Score vs Loan Default

| Applicant | Credit Score | Default |
|-----------|--------------|---------|
| 1 | 600 | Yes (1) |
| 2 | 650 | Yes (1) |
| 3 | 700 | No (0) |

### Step 1: Initialize Parameters
$$\beta_0 = 0, \beta_1 = 0$$

### Step 2: First Iteration Predictions

For each sample, compute $z = \beta_0 + \beta_1 x$, then $p = \frac{1}{1+e^{-z}}$

**Sample 1:** $x = 600$
$$z = 0 + 0(600) = 0$$
$$p = \frac{1}{1+e^{0}} = \frac{1}{2} = 0.5$$
Actual: $y = 1$, Error: $0.5 - 1 = -0.5$

**Sample 2:** $x = 650$
$$z = 0 + 0(650) = 0$$
$$p = 0.5$$
Actual: $y = 1$, Error: $0.5 - 1 = -0.5$

**Sample 3:** $x = 700$
$$z = 0 + 0(700) = 0$$
$$p = 0.5$$
Actual: $y = 0$, Error: $0.5 - 0 = 0.5$

### Step 3: Compute Gradients (Learning Rate $\alpha = 0.0001$)

$$\frac{\partial J}{\partial \beta_0} = \frac{1}{3}[(-0.5) + (-0.5) + (0.5)] = -\frac{0.5}{3} \approx -0.167$$

$$\frac{\partial J}{\partial \beta_1} = \frac{1}{3}[600(-0.5) + 650(-0.5) + 700(0.5)]$$
$$= \frac{1}{3}[-300 - 325 + 350] = \frac{-275}{3} \approx -91.67$$

### Step 4: Update Parameters

$$\beta_0 := 0 - 0.0001(-0.167) = 0.0000167$$
$$\beta_1 := 0 - 0.0001(-91.67) = 0.009167$$

### Step 5: Second Iteration Predictions (with updated β)

**Sample 1:** $x = 600$
$$z = 0.0000167 + 0.009167(600) = 5.5$$
$$p = \frac{1}{1+e^{-5.5}} \approx 0.9959$$
Improved! Much closer to 1

This process continues until convergence...

---

## 10. Full Manual Training Example

### Dataset: 5 Loan Applicants

| Applicant | Credit Score | Debt Ratio | Default |
|-----------|--------------|-----------|---------|
| 1 | 600 | 0.40 | 1 |
| 2 | 650 | 0.35 | 1 |
| 3 | 700 | 0.30 | 0 |
| 4 | 750 | 0.25 | 0 |
| 5 | 800 | 0.20 | 0 |

### Iteration 0: Initialization
$$\beta_0 = 0, \beta_1 = 0, \beta_2 = 0$$
Learning Rate: $\alpha = 0.001$

### Iteration 1: Predictions

$$z_1 = 0 + 0(600) + 0(0.40) = 0 \rightarrow p_1 = 0.5$$
$$z_2 = 0 + 0(650) + 0(0.35) = 0 \rightarrow p_2 = 0.5$$
$$z_3 = 0 + 0(700) + 0(0.30) = 0 \rightarrow p_3 = 0.5$$
$$z_4 = 0 + 0(750) + 0(0.25) = 0 \rightarrow p_4 = 0.5$$
$$z_5 = 0 + 0(800) + 0(0.20) = 0 \rightarrow p_5 = 0.5$$

### Iteration 2: Compute Errors

| i | $y_i$ | $p_i$ | Error: $p_i - y_i$ |
|---|-------|-------|-------------|
| 1 | 1 | 0.5 | -0.5 |
| 2 | 1 | 0.5 | -0.5 |
| 3 | 0 | 0.5 | 0.5 |
| 4 | 0 | 0.5 | 0.5 |
| 5 | 0 | 0.5 | 0.5 |

### Iteration 3: Compute Gradients

$$\frac{\partial J}{\partial \beta_0} = \frac{1}{5}[(-0.5) + (-0.5) + 0.5 + 0.5 + 0.5] = 0.1$$

$$\frac{\partial J}{\partial \beta_1} = \frac{1}{5}[600(-0.5) + 650(-0.5) + 700(0.5) + 750(0.5) + 800(0.5)]$$
$$= \frac{1}{5}[-300 - 325 + 350 + 375 + 400] = \frac{500}{5} = 100$$

$$\frac{\partial J}{\partial \beta_2} = \frac{1}{5}[0.40(-0.5) + 0.35(-0.5) + 0.30(0.5) + 0.25(0.5) + 0.20(0.5)]$$
$$= \frac{1}{5}[-0.20 - 0.175 + 0.15 + 0.125 + 0.10] = \frac{-0.025}{5} = -0.005$$

### Iteration 4: Update Parameters

$$\beta_0 := 0 - 0.001(0.1) = -0.0001$$
$$\beta_1 := 0 - 0.001(100) = -0.1$$
$$\beta_2 := 0 - 0.001(-0.005) = 0.000005$$

### Iteration 5: New Predictions (with updated β)

**Sample 1:** $x_1 = [600, 0.40]$
$$z_1 = -0.0001 + (-0.1)(600) + 0.000005(0.40) = -60.0001$$
$$p_1 = \frac{1}{1+e^{60}} \approx 0.0000 \text{ (very small!)}$$

Wait, this moved in wrong direction! (We want higher prediction for default)

**Observation:** Gradient descent requires careful learning rate tuning. Large learning rate overshoots.

After many iterations with proper learning rate, parameters eventually converge to stable values...

---

## 11. Parameters (Learned Values)

### In Our Loan Default Model

Suppose after training we get:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $\beta_0$ (Intercept) | -8.5 | Log-odds baseline |
| $\beta_1$ (Credit Score) | 0.015 | Each point increase → 0.015 increase in log-odds of default |
| $\beta_2$ (Debt Ratio) | 3.2 | Each 0.1 increase in debt ratio → 0.32 increase in log-odds |

### Interpretation
- Higher credit score → Lower default probability
- Higher debt ratio → Higher default probability
- A typical applicant has ~5% default probability

### Converting Coefficients to Odds Ratios

Odds Ratio = $e^{\beta}$

For Credit Score ($\beta_1 = 0.015$):
$$OR = e^{0.015} \approx 1.015$$

**Interpretation:** For every 1-point increase in credit score, odds of default multiply by 1.015 (1.5% increase in odds). 

Or: Every 100-point increase → odds multiply by $e^{100(0.015)} = e^{1.5} \approx 4.48$ (odds of default decrease significantly)

---

## 12. Hyperparameters

### Logistic Regression Hyperparameters

| Hyperparameter | Purpose | Typical Values |
|---|---|---|
| `penalty` | Regularization type | 'l2', 'l1', 'elasticnet', None |
| `C` | Inverse regularization strength | [0.01, 0.1, 1, 10, 100] |
| `max_iter` | Maximum iterations | [100, 1000, 5000] |
| `solver` | Optimization algorithm | 'lbfgs', 'liblinear', 'saga', 'newton-cg' |
| `class_weight` | Handle imbalanced classes | 'balanced' or dict |
| `random_state` | For reproducibility | Any integer |

### Understanding Key Hyperparameters

#### penalty and C
- `penalty='l2'` (default): Ridge (Tikhonov) regularization
- `penalty='l1'`: Lasso regularization
- `penalty='elasticnet'`: Combined L1+L2
- `C`: Inverse of regularization strength
  - Small C (0.01): Strong regularization, simpler model
  - Large C (100): Weak regularization, complex model
  - C=1.0: Default, usually good starting point

#### solver
- `'lbfgs'`: Default, good for small-medium datasets
- `'liblinear'`: Fast for binary classification, small datasets
- `'saga'`: Good for large datasets, supports L1 and elasticnet
- `'newton-cg'`: Good when you have many features

#### class_weight
- `None`: Treat classes equally
- `'balanced'`: Adjust weights inversely proportional to class frequency
- **Use 'balanced'** when classes are imbalanced (many 0s, few 1s)

#### max_iter
- Default: 100 (may not converge)
- Increase if warning "ConvergenceWarning" appears
- Try 1000 or 5000 for difficult datasets

---

## 13. Why Each Hyperparameter Exists

### penalty and C - Why Regularization?

**Problem:**
- Without regularization, model can overfit
- Large coefficients lead to extreme probabilities (0.0001, 0.9999)
- Poor generalization to new data

**Solution:**
- L2 (Ridge): Shrinks coefficients, keeps all features
- L1 (Lasso): Shrinks some coefficients to zero, automatic feature selection

**Example:**
```
Without regularization (C=large):
β = [-0.5, 2.3, -1.8, 3.4, -2.1]  (large values)

With regularization (C=small):
β = [-0.1, 0.3, -0.2, 0.4, -0.2]  (shrunken)
```

### solver - Why Different Optimization Algorithms?

**Problem:** Different algorithms have different trade-offs

**Solutions:**
- `lbfgs`: Robust, accurate, slower for large datasets
- `liblinear`: Fast, needs small datasets
- `saga`: Scalable, supports all penalties, good for large data

**Example:** For 100,000 samples, use 'saga' not 'lbfgs'

### class_weight - Why Handle Imbalance?

**Problem:** When 95% of loans don't default, model learns to always predict "no default"
- Accuracy looks good (95%), but useless!
- Predicts minority class poorly

**Solution:** Give more weight to minority class
- Model pays more attention to rare events
- Better balance between classes

### max_iter - Why Limit Iterations?

**Problem:** Sometimes optimization doesn't converge (oscillates)

**Solutions:**
- Set reasonable max_iter (1000, 5000)
- Increase if convergence warning
- Or adjust learning rate (via solver choice)

---

## 14. Assumptions

### Logistic Regression Assumptions

### 1. **Linearity in Log-Odds**
- Linear relationship between features and **log-odds** (not raw probability!)
- Log-odds = $\ln(\frac{p}{1-p}) = \beta_0 + \beta_1 x$
- Can be checked with Box-Tidwell test

### 2. **Independence of Observations**
- Each applicant's loan decision independent of others
- No temporal correlation
- Each row is separate observation

### 3. **No Multicollinearity**
- Independent variables not highly correlated
- High correlation causes unstable coefficients
- Use variance inflation factor (VIF) to check

### 4. **Adequate Sample Size**
- Minimum: 20 events per predictor
- Example: 500 defaults, 5 features → OK
- Better: 50 events per predictor

### 5. **No Perfect Separation**
- Classes should be somewhat overlapping
- If credit score perfectly separates default/non-default → singular matrix

### 6. **Correct Specification**
- Model includes all relevant features
- No important features left out
- No irrelevant features included

---

## 15. Data Requirements

| Requirement | Needed | Notes |
|-------------|--------|-------|
| **Scaling** | Recommended | StandardScaler helps convergence |
| **Encoding** | Yes | Convert categorical to numerical |
| **Missing Values** | Must handle | Imputation or removal required |
| **Outliers** | Consider | Less sensitive than linear regression |
| **Sample Size** | $n > 20p$ | At least 20 samples per feature |
| **Class Balance** | Important | Handle imbalance with class_weight |
| **Feature Engineering** | Optional | Can add polynomial terms if needed |

### Loan Dataset Requirements
- **Features:** Numerical (Credit Score, Income, Debt Ratio)
- **Target:** Binary (Default: 0/1, Yes/No)
- **Missing Values:** Handle via imputation
- **Scaling:** StandardScaler recommended for faster convergence
- **Class Balance:** Check ratio of defaults to non-defaults

### Handling Class Imbalance

```
Example: 900 non-default, 100 default
Imbalance ratio: 9:1

Solutions:
1. Use class_weight='balanced' (default solution)
2. Oversample minority class
3. Undersample majority class
4. SMOTE (Synthetic Minority Oversampling)
5. Adjust decision threshold above 0.5
```

---

## 16. Complexity

### Training Complexity

**With Newton-Raphson (default solver):**
$$O(n \times p^2 \times \text{iterations})$$

- $n$ = number of samples
- $p$ = number of features
- iterations typically 10-100 (usually converges fast)

**Typical Examples:**
- 1,000 samples, 5 features: ~0.01 seconds
- 100,000 samples, 50 features: ~10 seconds
- 1,000,000 samples: Use SGD solver (much faster)

### Prediction Complexity
$$O(p)$$
- Linear in number of features
- Very fast: microseconds per prediction
- Good for real-time applications

### Memory Complexity
$$O(n \times p)$$
- Store feature matrix
- Store coefficient vector (small)
- Typically not limiting factor

---

## 17. Decision Boundary

### Binary Classification: Single Threshold

```
Probability
    1.0 |                       Class 1
        |                      (Default)
   0.75 |                    ━━━━━━
        |                  /
   0.50 | ← Threshold ━━━/━ Decision Boundary
        |            /
   0.25 |          /
        |        /
    0.0 |━━━━━/        Class 0
        |           (No Default)
        |_________________
        0    2    4    6    8
              Feature Value
```

### 2D Visualization (Two Features)

```
High
Debt │  ① ① ①      Credit Score →
Ratio│  ① ①  ╱
     │      ╱ ← Decision Boundary
     │    ╱
     │  ╱
     │╱ ① (Default) / ⓪ (No Default)
Low  ├─────────────────
     Low   Medium   High
```

### Properties
- **Linear in input space**
- **Nonlinear in probability space** (due to sigmoid)
- **Smooth transition** between classes
- **Probabilistic interpretation**

### Threshold Selection
- **Default: 0.5**
  - Equal cost for false positives and false negatives
  - Good when classes equally important

- **Lower threshold (0.3):**
  - Catches more defaults (higher recall)
  - Rejects more loans (higher false positive rate)
  - Use when cost of missing default is high

- **Higher threshold (0.7):**
  - Fewer false alarms (higher precision)
  - Miss some defaults (lower recall)
  - Use when cost of false alarm is high

---

## 18. Overfitting & Underfitting

### Underfitting

**Symptoms:**
- High training error (low training accuracy)
- High test error (low test accuracy)
- Model too simple to capture relationship

**Causes:**
- Too few features
- Not enough training time
- Too strong regularization

**Solution:**
- Add features or polynomial terms
- Increase iterations
- Decrease regularization (increase C)

### Good Fit

**Symptoms:**
- Training accuracy and test accuracy both high
- Similar performance on train and test
- Coefficients stable and interpretable

**Characteristics:**
- Model generalizes well
- Ready for production use

### Overfitting

**Symptoms:**
- Very high training accuracy (99%+)
- Much lower test accuracy
- Model oscillates on new data

**Causes:**
- Too many features
- Weak regularization
- Too many iterations

**Solution:**
- Reduce features (especially for high-dimensional data)
- Increase regularization (decrease C)
- Use early stopping
- Collect more training data

---

## 19. Regularization

### Problem
Without regularization, model can:
- Learn extreme coefficients
- Become unstable with correlated features
- Overfit training data

### L2 Regularization (Ridge)

**Loss Function:**
$$J(\beta) = \frac{1}{n} \sum_i [y_i \log(p_i) + (1-y_i) \log(1-p_i)] + \frac{\lambda}{2n} \sum_j \beta_j^2$$

**Effect:**
- Shrinks coefficients proportionally
- All features retained
- Useful for correlated features

**Scikit-Learn:**
```python
LogisticRegression(penalty='l2', C=1.0)
# C = 1/λ, so smaller C = stronger regularization
```

### L1 Regularization (Lasso)

**Loss Function:**
$$J(\beta) = \frac{1}{n} \sum_i [y_i \log(p_i) + (1-y_i) \log(1-p_i)] + \frac{\lambda}{n} \sum_j |\beta_j|$$

**Effect:**
- Some coefficients shrunk to exactly zero
- Automatic feature selection
- Sparse solutions

**Scikit-Learn:**
```python
LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
```

### Elastic Net

**Loss Function:**
$$J(\beta) = Loss + \frac{\lambda_1}{n} \sum_j |\beta_j| + \frac{\lambda_2}{2n} \sum_j \beta_j^2$$

**Effect:**
- Combines L1 and L2 benefits
- Feature selection + stable coefficients

**Scikit-Learn:**
```python
LogisticRegression(penalty='elasticnet', solver='saga', C=1.0)
```

### Comparison

| Regularization | Effect | Use Case |
|---|---|---|
| **None** | No shrinkage | Small datasets, few features |
| **L2** | Shrink all proportionally | Correlated features, many samples |
| **L1** | Some to zero | Feature selection needed |
| **Elastic Net** | Balanced shrinkage | General-purpose, safe choice |

### Tuning Regularization Strength (C parameter)

- **C = 0.01** (strong regularization): Simple model, may underfit
- **C = 0.1:** Moderate regularization
- **C = 1.0:** Default, good starting point
- **C = 10:** Weak regularization, more complex
- **C = 100:** Very weak, may overfit

---

## 20. Feature Importance

### Method 1: Coefficient Magnitude

Larger magnitude = More important

```python
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)
```

### Method 2: Odds Ratios

$$OR_j = e^{\beta_j}$$

**Interpretation:**
- OR > 1: Unit increase increases odds of default
- OR = 1: Feature has no effect
- OR < 1: Unit increase decreases odds of default

### Method 3: Permutation Importance

Measure accuracy drop when feature is shuffled.

### Example: Loan Default Model

Suppose we have:

| Feature | Coefficient | Odds Ratio | Interpretation |
|---------|-------------|-----------|---|
| Credit Score | -0.015 | 0.985 | 100-point increase → odds of default multiply by 0.85 (16% decrease) |
| Debt Ratio | 3.2 | 24.5 | 0.1 increase → odds multiply by 24.5 (2450% increase!) |
| Income | -0.00001 | 0.99999 | $1000 increase → negligible effect |

**Interpretation:**
1. **Debt Ratio** is most important (largest OR)
2. **Credit Score** important (negative = protective)
3. **Income** nearly useless (OR ≈ 1)

---

## 21. Advantages

✅ **Simple and Fast**
- Fast training and prediction
- Good for real-time applications
- Scales to large datasets with SGD

✅ **Interpretable**
- Direct probability output
- Coefficients show feature importance
- Easy to explain to stakeholders

✅ **Probabilistic**
- Outputs probabilities (0-1), not just classes
- Can adjust threshold for business needs
- Confidence scores for predictions

✅ **Foundation for Complex Models**
- Building block for neural networks
- Extensions available (multiclass, multinomial)
- Well-understood mathematically

✅ **Works Well with Many Features**
- Regularization handles high dimensions
- No exponential complexity like some other methods
- Scales linearly with feature count

✅ **Robust to Outliers**
- Less sensitive than linear regression
- Sigmoid function naturally dampens extremes

✅ **Good for Binary Classification**
- Explicitly designed for 2-class problems
- Natural probability interpretation
- Well-calibrated predictions

---

## 22. Limitations

❌ **Assumes Linear Decision Boundary**
- Can't handle highly nonlinear relationships
- Limited to capturing linear patterns in log-odds space
- Solution: Add polynomial features or use nonlinear models

❌ **Requires Feature Engineering**
- Can't automatically discover nonlinear patterns
- Need to manually add interactions
- Example: Model needs $x^2$ to capture quadratic relationship

❌ **Sensitive to Multicollinearity**
- Correlated features cause unstable coefficients
- Predictions can be unreliable
- Solution: Remove correlated features or use regularization

❌ **Requires Numerical Features**
- Can't directly handle text or images
- All categorical must be encoded
- Extra preprocessing step needed

❌ **Class Imbalance Issues**
- Biased toward majority class by default
- Poor minority class predictions
- Solution: Use class_weight='balanced'

❌ **No Interaction Discovery**
- Manual feature engineering required
- Doesn't automatically find feature interactions
- Example: Can't discover "high debt AND low income" is risky

❌ **Limited to Binary (or One-vs-Rest Multiclass)**
- Original formulation for 2 classes
- Multiclass requires special handling
- Softmax extension less powerful than dedicated methods

---

## 23. Failure Cases

### When Logistic Regression Fails

#### 1. **Nonlinear Decision Boundary**

```
Truth: Circular boundary (distance from center)
Logistic Regression: Linear boundary
Result: Misclassifies many points in center region
```

**Solution:** Use kernel methods or neural networks

#### 2. **Severe Class Imbalance**

```
Dataset: 99% non-default, 1% default
Model learns: Always predict non-default
Accuracy: 99%, but useless!
```

**Solution:** Use class_weight='balanced'

#### 3. **Multicollinearity**

```
Feature 1: Income
Feature 2: Years employed (highly correlated)
Result: Coefficients unstable, oscillate wildly
```

**Solution:** Remove correlated feature or use L2 regularization

#### 4. **Complete Separation**

```
Truth: Credit score 800+ → always repay
       Credit score <600 → always default
No middle ground
Result: Coefficients → ±∞ (singular matrix)
```

**Solution:** Add noise or use regularization

#### 5. **Missing Feature**

```
Truth: Income determines default
Model: Only has credit score
Result: Low accuracy, missing crucial information
```

**Solution:** Collect additional features

#### 6. **Temporal/Sequential Data**

```
Time series: Stock prices, sensor readings
Issue: Logistic Regression assumes independent observations
Result: Poor predictions on correlated time series
```

**Solution:** Use RNN or ARIMA

---

## 24. Edge Cases

### What Happens When...

#### **Dataset is Tiny (n < 20)**
- Parameters unreliable, high variance
- Cross-validation not meaningful
- May converge to local optimum
- **Solution:** Use domain knowledge, stronger regularization

#### **Perfect Separation (n=100, 50 default, 50 no-default)**
- Each group perfectly separated
- Coefficients → ±∞
- Model undefined
- **Solution:** Add regularization or noise

#### **All Features Identical**
```
X = [[5], [5], [5], [5]]
Problem: No variation → can't fit model
Error: Singular matrix
```
**Solution:** Ensure feature variation

#### **Target is Constant**
```
y = [1, 1, 1, 1]  (all defaults)
Result: Model can't learn, probability → 1
Metrics: Perfect accuracy but meaningless
```
**Solution:** Need both classes to train

#### **Features Perfectly Correlated**
```
Feature 1: Age
Feature 2: Birth Year (= 2024 - Age)
Result: Singular matrix, can't invert
```
**Solution:** Remove one feature

#### **Extreme Feature Scales**
```
Feature 1: Age (1-100)
Feature 2: Income (10,000-1,000,000)
Problem: Optimization slow/unstable
Solution: StandardScaler before fitting
```

#### **Missing Values**
```
X = [[600, NaN], [650, 0.35], [700, NaN]]
Problem: Can't compute with NaN
Solutions:
1. Remove rows with NaN
2. Impute with mean/median
3. Use model that handles NaN
```

#### **Categorical Feature (Not Encoded)**
```
X = [['Good', 600], ['Fair', 650], ['Poor', 700]]
Error: Can't compute distances with text
Solution: One-hot encode or label encode
```

---

## 25. Evaluation Metrics

### For Binary Classification

#### **Confusion Matrix Terms**
```
              Predicted
              0      1
Actual  0   TN     FP
        1   FN     TP

TN = True Negative (correct non-default)
FP = False Positive (wrong, predicted default)
FN = False Negative (wrong, predicted non-default)
TP = True Positive (correct default prediction)
```

#### **Accuracy**
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Range:** 0 to 1
- **Interpretation:** Fraction of correct predictions
- **Best for:** Balanced datasets
- **Caution:** Misleading with imbalanced data

#### **Precision** (Positive Predictive Value)
$$Precision = \frac{TP}{TP + FP}$$

- **Range:** 0 to 1
- **Interpretation:** When model predicts default, how often correct?
- **Use case:** High cost of false alarms
- **Example:** 80% precision means 80% of predicted defaults are real

#### **Recall** (True Positive Rate / Sensitivity)
$$Recall = \frac{TP}{TP + FN}$$

- **Range:** 0 to 1
- **Interpretation:** What fraction of actual defaults did model catch?
- **Use case:** High cost of missing defaults
- **Example:** 90% recall means model catches 90% of defaulters

#### **F1 Score** (Harmonic Mean)
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

- **Range:** 0 to 1
- **Interpretation:** Balanced metric between precision and recall
- **Best for:** Imbalanced datasets
- **Use case:** Want both precision and recall

#### **Specificity** (True Negative Rate)
$$Specificity = \frac{TN}{TN + FP}$$

- **Range:** 0 to 1
- **Interpretation:** What fraction of actual non-defaults correctly identified?
- **Complement of False Positive Rate**

#### **ROC-AUC** (Area Under Receiver Operating Characteristic Curve)
- **Range:** 0 to 1
- **Interpretation:** Model's ability to distinguish classes
  - AUC = 0.5: Random guessing
  - AUC = 1.0: Perfect classifier
  - AUC = 0.7-0.8: Good model
  - AUC > 0.8: Excellent model

#### **Precision-Recall Curve**
- Better for **imbalanced datasets**
- X-axis: Recall, Y-axis: Precision
- Higher curves = better models
- Area Under Curve (AP) summarizes performance

### Metrics Interpretation for Loan Default

| Metric | Loan Context | Good Value |
|--------|---|---|
| Accuracy | Overall correctness | >80% |
| Precision | Of loans we reject, % we correctly identified as risky | >70% |
| Recall | % of actual defaulters we catch | >80% |
| F1 | Balance between catching defaults and avoiding false alarms | >0.75 |
| ROC-AUC | Model's ranking ability | >0.75 |

---

## 26. Comparison with Similar Models

### Logistic vs Other Classification Methods

| Model | Decision Boundary | Training Speed | Interpretability | Nonlinear Capability |
|-------|---|---|---|---|
| **Logistic Regression** | Linear | Fast | High | Limited |
| **Decision Tree** | Rectangular | Fast | High | Yes |
| **Random Forest** | Complex (ensemble) | Moderate | Low | Yes |
| **SVM** | Linear or nonlinear (kernel) | Moderate | Low | Yes (with kernel) |
| **Neural Network** | Highly nonlinear | Slow | Very Low | Yes |
| **KNN** | Local/complex | Instant training, slow prediction | Low | Yes |
| **Naive Bayes** | Probabilistic | Very Fast | High | Limited |

### Logistic vs Decision Tree

| Aspect | Logistic Regression | Decision Tree |
|--------|---|---|
| **Linearity** | Assumes linear in log-odds | No assumptions |
| **Interpretability** | Coefficient → feature importance | Decision path → interpretable |
| **Speed** | Very fast prediction | Fast prediction |
| **Overfitting** | Less prone (with regularization) | Prone (needs pruning) |
| **Feature Engineering** | Required (polynomial terms) | Automatic interaction detection |
| **Handling Missing** | Manual imputation needed | Can handle missing |
| **Categorical Data** | Must encode | Handles naturally |

### When to Use Logistic Regression Over Others

1. **Interpretability critical** → Logistic Regression
2. **Real-time prediction needed** → Logistic Regression or KNN
3. **Small-medium dataset** → Logistic Regression
4. **Probabilistic output needed** → Logistic Regression or SVM
5. **Data is truly linear separable** → Logistic Regression
6. **Explainability to non-technical** → Logistic Regression

---

## 27. Real-World Applications

### Finance & Banking
- **Loan Default Prediction** ← Our Use Case
  - Predict which applicants will default
  - Input: Credit score, income, debt ratio
  - Output: Default probability

- **Credit Card Fraud Detection**
  - Real-time fraud/non-fraud classification
  - Fast inference needed (milliseconds)

- **Credit Scoring**
  - Assign credit ratings based on financial metrics

### Healthcare
- **Disease Diagnosis**
  - Predict disease presence (diseased/healthy)
  - Input: Symptoms, lab results
  - High interpretability needed

- **Readmission Risk**
  - Predict hospital readmission probability
  - Input: Patient demographics, diagnoses

### Marketing
- **Customer Churn Prediction**
  - Predict if customer will leave
  - Input: Usage patterns, subscription type

- **Email Campaign Response**
  - Predict if customer opens email
  - Input: Customer segments, past behavior

- **Lead Conversion**
  - Predict if lead converts to sale
  - Input: Lead source, engagement metrics

### Online Platforms
- **Spam Detection**
  - Classify email/comments as spam or legitimate
  - Fast inference required

- **Recommendation Systems**
  - Predict user interest in item
  - Input: User profile, item features

### Other Industries
- **Insurance:** Claim fraud detection
- **HR:** Employee attrition prediction
- **Retail:** Customer purchase prediction
- **Manufacturing:** Equipment failure prediction

---

## 28. Scikit-Learn Implementation

### Basic Usage

```python
# Import
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import pandas as pd

# Create model
model = LogisticRegression(random_state=42)

# Fit to data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate
print(classification_report(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities[:, 1]))
```

### With Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# Fit and evaluate
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

### Handle Class Imbalance

```python
# Method 1: Using class_weight
model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

# Method 2: Manual weights based on frequencies
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
model = LogisticRegression(
    class_weight={0: class_weights[0], 1: class_weights[1]},
    random_state=42
)
```

### Regularization & Hyperparameter Tuning

```python
# L1 Regularization (Lasso)
model_l1 = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=1.0,
    random_state=42
)

# L2 Regularization (Ridge)
model_l2 = LogisticRegression(
    penalty='l2',
    solver='lbfgs',
    C=1.0,
    random_state=42
)

# Grid Search for best C
from sklearn.model_selection import GridSearchCV

params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(
    LogisticRegression(solver='liblinear'),
    params,
    cv=5,
    scoring='roc_auc'
)
grid.fit(X_train, y_train)
print("Best C:", grid.best_params_)
```

### Feature Importance

```python
# Get coefficients
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds_Ratio': np.exp(coefficients)
}).sort_values('Coefficient', ascending=False)

print(feature_importance)

# Interpretation
# Positive coefficient → increases probability of class 1
# Negative coefficient → decreases probability of class 1
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_validate

# Simple cross-validation
cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000),
    X, y,
    cv=5,
    scoring='roc_auc'
)

# Multiple metrics
metrics = cross_validate(
    LogisticRegression(max_iter=1000),
    X, y,
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

for metric, scores in metrics.items():
    print(f"{metric}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 29. Interview Questions

### Question 1: Why Use Logistic Regression for Classification?
**Answer:**
- Outputs probabilities (interpretable)
- Fast training and prediction
- Good baseline classifier
- Strong statistical foundation
- Explains decision-making (feature importance)

### Question 2: Explain the Sigmoid Function
**Answer:**
- Transforms unbounded values (-∞, +∞) to [0, 1]
- Formula: σ(z) = 1/(1+e^-z)
- S-shaped curve
- Derivative used in gradient descent
- Ensures valid probabilities

### Question 3: What's the Difference Between Logistic and Linear Regression?
**Answer:**
| Aspect | Linear | Logistic |
|--------|--------|----------|
| **Task** | Regression (continuous output) | Classification (discrete output) |
| **Output** | Any real number | Probability [0, 1] |
| **Loss** | MSE | Cross-entropy |
| **Boundary** | N/A | Linear decision boundary |
| **Use Case** | Predict continuous values | Predict class membership |

### Question 4: What's the Difference Between L1 and L2 Regularization?
**Answer:**
- **L2 (Ridge):** Penalizes sum of squared coefficients, shrinks all proportionally
- **L1 (Lasso):** Penalizes sum of absolute coefficients, shrinks some to zero
- **L2:** Better when all features useful
- **L1:** Better when need feature selection
- **Elastic Net:** Combines both

### Question 5: How Do You Handle Imbalanced Classes?
**Answer:**
1. Use `class_weight='balanced'`
2. Oversample minority class
3. Undersample majority class
4. SMOTE (synthetic data generation)
5. Adjust decision threshold (use ROC curve)
6. Use F1 or ROC-AUC instead of accuracy

### Question 6: What Does the Coefficient Tell Us?
**Answer:**
- Magnitude: Feature importance
- Sign: Direction (positive/negative effect)
- Odds ratio = e^coefficient
  - OR > 1: Increases odds of class 1
  - OR < 1: Decreases odds of class 1
  - OR = 1: No effect
- Example: β=0.02 → OR=1.02 → 2% increase in odds per unit

### Question 7: When Would Model Fail?
**Answer:**
1. Nonlinear decision boundary (use kernel methods)
2. Missing important features (collect more data)
3. Severe multicollinearity (remove correlated features)
4. Very imbalanced classes (use class_weight)
5. Complete separation (add regularization)

### Question 8: What's ROC-AUC and Why Use It?
**Answer:**
- Plots True Positive Rate vs False Positive Rate
- Shows model performance at all thresholds
- AUC = 0.5 → Random, AUC = 1.0 → Perfect
- Better than accuracy for imbalanced data
- Threshold-independent evaluation

### Question 9: How Do You Choose Threshold for Predictions?
**Answer:**
- **Default: 0.5** (equal cost for errors)
- **Lower (<0.5):** More sensitive, catches more positives, more false alarms
  - Use when missing positive is costly (fraud, disease)
- **Higher (>0.5):** More specific, fewer false alarms
  - Use when false alarm is costly (medical test)
- Choose based on **cost/benefit** of false positives vs false negatives

### Question 10: What Assumptions Does Logistic Regression Make?
**Answer:**
1. Linear relationship between features and log-odds
2. Independent observations
3. No severe multicollinearity
4. No perfect separation
5. Correct model specification
6. No omitted variables

---

## Conclusion

Logistic Regression is a fundamental classification algorithm that:
- **Outputs probabilities** for interpretable predictions
- **Trains fast** for rapid model development
- **Scales well** to large datasets
- **Provides feature importance** for explainability
- **Serves as baseline** for comparing more complex models

Despite its simplicity, Logistic Regression remains a powerful and widely-used tool in industry for classification tasks across finance, healthcare, marketing, and technology.

---

## Additional Resources

### Books
- "An Introduction to Statistical Learning" - James, Witten, Hastie, Tibshirani
- "Logistic Regression: A Self-learning Text" - Kleinbaum & Klein

### Online Resources
- [Scikit-Learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgil8)

### Related Models
- Linear Regression (foundation)
- Decision Trees & Random Forests (nonlinear)
- Support Vector Machines (kernel methods)
- Neural Networks (deep learning)
- Naive Bayes (probabilistic alternative)

---

**Model Location:** `logistic_regression/` folder  
**Dataset:** Loan Default Data  
**Target Variable:** Loan Default (0/1)  
**Features:** Credit Score, Debt Ratio, Income, Employment History, etc.  
**Last Updated:** June 2026
