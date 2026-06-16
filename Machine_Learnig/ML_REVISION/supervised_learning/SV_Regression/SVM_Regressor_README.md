# Support Vector Machine Regressor (SVR) — Complete Study Guide

## 1. Model Name

- **Full Name:** Support Vector Regression (SVR)
- **Category:** Regression
- **Type:** Supervised Learning

---

## 2. Problem It Solves

### What task is it designed for?
SVR predicts **continuous numerical values** by finding a function that deviates from the actual target values by at most $\varepsilon$ (epsilon) for as many training points as possible, while keeping the function as flat (simple) as possible.

Examples:
- Predicting house prices from structural and location features
- Forecasting stock prices or energy demand
- Estimating age from physiological measurements
- Predicting drug response from molecular features

### Why do we need this model?
- Most regression methods (e.g., Linear Regression) minimize the total error across **all** training points. SVR instead tries to fit a "tube" of width $\varepsilon$ around the data — errors within the tube are **ignored**, only large errors matter.
- This $\varepsilon$-insensitive approach makes SVR **robust to small noise** and **resistant to outliers**.
- Via the **kernel trick**, SVR can model highly non-linear relationships without explicit feature engineering.
- SVR provides a **theoretically grounded** generalization guarantee based on structural risk minimization.

---

## 3. Intuition

### Project Context - Real Estate Price Prediction

In this folder, the SVR model is used for a **real estate regression** task.

The local dataset is:

```text
Real estate.csv
```

The target column is:

```text
Y house price of unit area
```

The Streamlit app predicts house price per unit area from:

| Feature | Meaning |
|---|---|
| `X2 house age` | Age of the house in years |
| `X3 distance to the nearest MRT station` | Distance to the nearest MRT station |
| `X4 number of convenience stores` | Nearby convenience store count |

The full dataset also includes transaction date, latitude, and longitude. The current app input form uses the three features listed above.

The saved model file is:

```text
svr_real_estate_model.pkl
```

### Simple Explanation
Imagine fitting a **tube** (a corridor of width $2\varepsilon$) around your data points. All points that fall **inside** the tube are considered "good enough" — their errors are zero in the loss function. Only points that fall **outside** the tube incur a penalty, and the model tries to bring as many points inside the tube as possible while keeping the tube as flat as possible.

The predicted value for a new input is the value of the **center line** of the tube at that point.

### Real-World Analogy
Think of a **road tolerance standard**: a road is considered perfectly built if it stays within ±5mm of the design specification. Any deviation beyond ±5mm counts as an error that needs correction. The construction engineer minimizes deviations beyond the tolerance threshold, not tiny imperfections within tolerance.

SVR does the same — it ignores small prediction errors (within $\varepsilon$) and focuses only on reducing large deviations, leading to a simpler, more generalizable model.

---

## 4. Mathematical Foundation

### Core Formula — The Prediction Function

$$\hat{y} = f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$

For non-linear SVR with kernel $K$:

$$\hat{y} = \sum_{i \in SV} (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b$$

### The ε-Insensitive Tube

$$f(\mathbf{x}) \text{ must lie within } \pm \varepsilon \text{ of } y$$

The constraint region:
$$y_i - \varepsilon \leq f(\mathbf{x}_i) \leq y_i + \varepsilon$$

### Explain Every Term

| Symbol | Meaning |
|---|---|
| $\mathbf{w}$ | Weight vector (determines the slope of the regression function) |
| $b$ | Bias term (vertical offset of the regression line/surface) |
| $\mathbf{x}$ | Input feature vector |
| $\hat{y}$ | Predicted continuous value |
| $\varepsilon$ | Epsilon — the width of the insensitive tube (half-width) |
| $\alpha_i, \alpha_i^*$ | Dual Lagrange multipliers (for above and below tube violations) |
| $K(\mathbf{x}_i, \mathbf{x})$ | Kernel function (dot product in high-dimensional space) |
| $\xi_i$ | Slack variable for points above the tube (overestimation) |
| $\xi_i^*$ | Slack variable for points below the tube (underestimation) |
| SV | Set of support vectors (points outside the $\varepsilon$-tube) |
| $C$ | Regularization parameter (penalty for tube violations) |

---

## 5. Objective Function

### Primal Formulation

**Minimize:**
$$\min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)$$

**Subject to:**
$$y_i - (\mathbf{w}^T \mathbf{x}_i + b) \leq \varepsilon + \xi_i$$
$$(\mathbf{w}^T \mathbf{x}_i + b) - y_i \leq \varepsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

### Intuition of Each Term

| Term | Role |
|---|---|
| $\frac{1}{2}\|\mathbf{w}\|^2$ | Minimize model complexity → maximize the tube "flatness" → better generalization |
| $C \sum (\xi_i + \xi_i^*)$ | Penalize predictions outside the tube |
| $\xi_i$ | How much prediction overestimates $y_i$ beyond $\varepsilon$ |
| $\xi_i^*$ | How much prediction underestimates $y_i$ beyond $\varepsilon$ |
| $\varepsilon$ | Half-width of the insensitive zone — errors within this zone cost nothing |
| $C$ | Trade-off between flatness (low $\|\mathbf{w}\|$) and accuracy (low violations) |

**When $C$ increases:** The model penalizes violations more → fits the tube tighter to the data → lower training error, higher risk of overfitting.

**When $\varepsilon$ increases:** The tube is wider → more points fall inside → fewer support vectors → smoother, simpler model → higher training error, better generalization.

---

## 6. Loss Function Derivation

### ε-Insensitive Loss (Vapnik Loss)

$$L_\varepsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \varepsilon)$$

**Intuition:**
- If $|y - \hat{y}| \leq \varepsilon$: The prediction is "close enough" → **Loss = 0**
- If $|y - \hat{y}| > \varepsilon$: The prediction exceeds tolerance → **Loss = |y - \hat{y}| - \varepsilon**

### Visual Representation

```
Loss
 |
 |         /
 |        /
 |       /    <-- slope = 1
 |      /
 |_____/_________________________ |y - ŷ|
       ε
 (flat zone, zero loss for |error| ≤ ε)
```

### Comparison of Loss Functions

| Loss Function | Formula | Behavior |
|---|---|---|
| MSE (squared) | $(y-\hat{y})^2$ | Large errors penalized quadratically; sensitive to outliers |
| MAE (absolute) | $\|y-\hat{y}\|$ | Linear penalty for all errors |
| ε-Insensitive | $\max(0, \|y-\hat{y}\|-\varepsilon)$ | Zero penalty within tube; linear outside |
| Huber Loss | Quadratic for small errors, linear for large | Hybrid approach |

### Gradient of ε-Insensitive Loss

$$\frac{\partial L_\varepsilon}{\partial \hat{y}} = \begin{cases}
0 & \text{if } |y - \hat{y}| \leq \varepsilon \\
-1 & \text{if } y - \hat{y} > \varepsilon \quad \text{(prediction too low)} \\
+1 & \text{if } \hat{y} - y > \varepsilon \quad \text{(prediction too high)}
\end{cases}$$

**Key property:** The gradient is zero inside the tube → these points do not contribute to the optimization → **sparse solution** (only points outside the tube matter).

### Full SVR Objective as Regularized Loss

$$\mathcal{L} = \frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac{1}{n}\sum_{i=1}^n L_\varepsilon(y_i, f(\mathbf{x}_i))$$

Where $\lambda = \frac{1}{Cn}$ connects the dual $C$ to regularization strength.

---

## 7. Optimization Method

### Dual Formulation

Using Lagrange multipliers $\alpha_i, \alpha_i^* \geq 0$, the SVR dual is:

$$\max_{\alpha, \alpha^*} -\frac{1}{2}\sum_{i,j}(\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)\mathbf{x}_i^T\mathbf{x}_j - \varepsilon\sum_{i}(\alpha_i + \alpha_i^*) + \sum_{i}y_i(\alpha_i - \alpha_i^*)$$

**Subject to:**
$$\sum_{i=1}^{n}(\alpha_i - \alpha_i^*) = 0, \quad 0 \leq \alpha_i, \alpha_i^* \leq C$$

### Recovery of Parameters

$$\mathbf{w} = \sum_{i=1}^{n}(\alpha_i - \alpha_i^*)\mathbf{x}_i$$

The bias $b$ is recovered from any support vector $j$ ($0 < \alpha_j < C$):
$$b = y_j - \mathbf{w}^T \mathbf{x}_j - \varepsilon \quad (\text{if point is above tube: } \alpha_j > 0)$$
$$b = y_j - \mathbf{w}^T \mathbf{x}_j + \varepsilon \quad (\text{if point is below tube: } \alpha_j^* > 0)$$

### Prediction (Kernel Form)

$$\hat{y} = \sum_{i \in SV}(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b$$

### Algorithm
SVR is solved using the **SMO (Sequential Minimal Optimization)** algorithm (via libsvm in scikit-learn), similar to SVC. At each step, it optimizes a pair of $\alpha_i, \alpha_i^*$ while keeping others fixed.

### Kernel Functions (same as SVC)

| Kernel | Formula | Use Case |
|---|---|---|
| Linear | $\mathbf{x}_i^T \mathbf{x}_j$ | Linear relationships |
| RBF | $\exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | Non-linear (default) |
| Polynomial | $(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$ | Polynomial trends |

---

## 8. Training Workflow

```
Input: Labeled Training Data (X, y — continuous target)
        ↓
Scale Features (MANDATORY — SVR is distance-based)
        ↓
Choose Kernel (Linear / RBF / Polynomial)
        ↓
Set Hyperparameters: C, epsilon (ε), gamma (for RBF)
        ↓
Form the ε-Insensitive Tube:
  For each sample i:
    - If |y_i - f(x_i)| ≤ ε → inside tube → zero loss
    - If |y_i - f(x_i)| > ε → outside tube → add slack variable
        ↓
Solve Dual QP Problem (via SMO / libsvm):
  - Find optimal α_i and α_i* (dual variables)
  - α_i > 0 → point is above the tube (support vector)
  - α_i* > 0 → point is below the tube (support vector)
  - α_i = α_i* = 0 → point is inside tube (not a support vector)
        ↓
Compute w = Σ (α_i - α_i*) x_i
Compute b from support vectors
        ↓
Model = support vectors + (α_i - α_i*) coefficients + b
        ↓
Prediction: ŷ = Σ (α_i - α_i*) K(x_i, x) + b
```

---

## 9. Worked Numerical Example

### Dataset (1D Linear)

| Sample | $x$ | $y$ |
|---|---|---|
| 1 | 1 | 2.5 |
| 2 | 2 | 4.0 |
| 3 | 3 | 5.5 |
| 4 | 4 | 7.5 |
| 5 | 5 | 9.0 |

**Assume:** Linear SVR with $\varepsilon = 0.5$, True function: $f(x) = 1.6x + 0.9$

### Step 1: Compute predictions

$$f(x) = 1.6x + 0.9$$

| Sample | $x$ | $y$ | $\hat{y} = 1.6x + 0.9$ | $\|y - \hat{y}\|$ | Inside tube ($\leq 0.5$)? |
|---|---|---|---|---|---|
| 1 | 1 | 2.5 | 2.5 | 0.0 | ✅ |
| 2 | 2 | 4.0 | 4.1 | 0.1 | ✅ |
| 3 | 3 | 5.5 | 5.7 | 0.2 | ✅ |
| 4 | 4 | 7.5 | 7.3 | 0.2 | ✅ |
| 5 | 5 | 9.0 | 8.9 | 0.1 | ✅ |

All points fall inside the $\varepsilon = 0.5$ tube → **No support vectors** → Total loss = 0.

### Step 2: Perturb data to create tube violations

Now suppose sample 3 is noisy: $y_3 = 6.5$ (was 5.5).

| Sample | $x$ | $y$ | $\hat{y}$ | Error | $L_\varepsilon$ |
|---|---|---|---|---|---|
| 3 | 3 | 6.5 | 5.7 | 0.8 | $0.8 - 0.5 = 0.3$ |

Sample 3 is now a **support vector** (above the tube). Its $\alpha_3 > 0$.

### Step 3: Compute ε-Insensitive Loss

$$\mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 + C \cdot 0.3$$

For $C = 1.0$: $\mathcal{L} = \frac{1}{2}(1.6)^2 + 0.3 = 1.28 + 0.3 = 1.58$

### Step 4: Prediction for new point $x = 3.5$

$$\hat{y} = 1.6(3.5) + 0.9 = 5.6 + 0.9 = 6.5$$

---

## 10. Full Manual Training Example

### Dataset

| ID | $x$ | $y$ |
|---|---|---|
| A | 1 | 1.8 |
| B | 2 | 3.9 |
| C | 3 | 6.5 |
| D | 4 | 8.1 |
| E | 5 | 9.8 |

**Settings:** $\varepsilon = 0.5$, $C = 1.0$, Linear kernel.

### Step 1: Fit a Linear Function

Assume $f(x) = 2x - 0.2$ (derived from QP solver).

### Step 2: Compute Errors and Tube Membership

| ID | $x$ | $y$ | $\hat{y} = 2x - 0.2$ | Error $= y - \hat{y}$ | $|error|$ | Inside tube? | $\xi$ or $\xi^*$ |
|---|---|---|---|---|---|---|---|
| A | 1 | 1.8 | 1.8 | 0.0 | 0.0 | ✅ Yes | 0 |
| B | 2 | 3.9 | 3.8 | +0.1 | 0.1 | ✅ Yes | 0 |
| C | 3 | 6.5 | 5.8 | +0.7 | 0.7 | ❌ No | $\xi_C = 0.7 - 0.5 = 0.2$ |
| D | 4 | 8.1 | 7.8 | +0.3 | 0.3 | ✅ Yes | 0 |
| E | 5 | 9.8 | 9.8 | 0.0 | 0.0 | ✅ Yes | 0 |

### Step 3: Compute Loss

$$\mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 + C \sum(\xi_i + \xi_i^*)$$

$$\|\mathbf{w}\| = |2| = 2 \quad (\text{weight for } x)$$

$$\mathcal{L} = \frac{1}{2}(4) + 1.0 \times 0.2 = 2.0 + 0.2 = 2.2$$

**Support Vector:** Only point C (it violates the tube). $\alpha_C > 0$.

### Step 4: Gradient Update (conceptually)

The gradient of the loss w.r.t $\mathbf{w}$ for point C (above tube):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - C \cdot \mathbf{x}_C = [2] - 1.0 \cdot [3] = -1$$

Updated: $\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot (-1)$ → moves $\mathbf{w}$ to better fit point C.

### Step 5: Prediction for $x = 3.5$

$$\hat{y} = 2(3.5) - 0.2 = 7.0 - 0.2 = 6.8$$

---

## 11. Parameters (Learned Values)

| Parameter | Meaning |
|---|---|
| $\mathbf{w}$ | Weight vector (slope of the regression function in feature space) |
| $b$ | Bias (intercept of the regression function) |
| $\alpha_i$ | Dual coefficient for points above the tube |
| $\alpha_i^*$ | Dual coefficient for points below the tube |
| $\alpha_i - \alpha_i^*$ | Net dual coefficient stored for each support vector |
| Support Vectors | Training samples outside the ε-tube ($\alpha_i \neq 0$ or $\alpha_i^* \neq 0$) |

**Key insight:** Points inside the ε-tube have $\alpha_i = \alpha_i^* = 0$ and contribute nothing to the model. Only support vectors (outside the tube) define the regression function.

---

## 12. Hyperparameters

| Hyperparameter | Default | Purpose |
|---|---|---|
| `C` | 1.0 | Penalty for tube violations; trade-off flatness vs accuracy |
| `epsilon` (ε) | 0.1 | Half-width of the insensitive tube |
| `kernel` | 'rbf' | Kernel function: 'linear', 'rbf', 'poly', 'sigmoid' |
| `gamma` | 'scale' | RBF/poly kernel coefficient |
| `degree` | 3 | Degree for polynomial kernel |
| `coef0` | 0.0 | Independent term in poly/sigmoid kernels |
| `tol` | 1e-3 | Convergence tolerance |
| `max_iter` | -1 | Max iterations (-1 = no limit) |
| `shrinking` | True | Use shrinking heuristic (speeds up training) |
| `cache_size` | 200 | Kernel cache size in MB (larger = faster training) |

---

## 13. Why Each Hyperparameter Exists

### `C` — Penalty for Violations
**Problem:** Without penalty, the model may choose a very flat function that ignores data.  
**Solution:** $C$ penalizes predictions outside the tube.
- **High C** (e.g., 1000): Model fits closely to training data, narrow/no-tolerance violations → overfitting risk
- **Low C** (e.g., 0.001): Model allows many violations, very flat function → underfitting risk
- **Rule of thumb:** Tune over $[0.01, 0.1, 1, 10, 100, 1000]$ via cross-validation

### `epsilon` (ε) — Tube Width
**Problem:** Real-world data has noise; we don't want to penalize tiny errors.  
**Solution:** Any prediction within ε of the true value incurs zero loss.
- **Large ε** (e.g., 0.5): Wider tolerance → fewer support vectors → simpler model → higher training error
- **Small ε** (e.g., 0.001): Narrow tolerance → more support vectors → more complex model → lower training error
- **Rule of thumb:** Set ε based on domain knowledge (acceptable prediction error) or tune via CV
- **Practical tip:** ε ≈ 10% of the standard deviation of y is a good starting point

### `gamma` — RBF Kernel Width
**Problem:** Controls how much influence each training point has.
- **High gamma:** Local influence → complex wiggly surface → overfitting
- **Low gamma:** Global influence → smooth surface → underfitting
- **`gamma='scale'`** (default): $\gamma = \frac{1}{n_{features} \cdot \text{Var}(X)}$ — adapts to feature variance

### `kernel`
- **Linear:** When the relationship is roughly linear; fast for high-dimensional data
- **RBF:** General-purpose non-linear regression (default choice)
- **Polynomial:** When polynomial trends exist in the data

---

## 14. Assumptions

1. **No linearity assumption:** With kernels, SVR can model any non-linear function.
2. **Feature scaling mandatory:** SVR uses distance-based computations — scales must be normalized.
3. **Noise assumption:** ε-insensitive tube implicitly assumes noise within ±ε is acceptable.
4. **Stationarity:** The kernel (especially RBF) assumes the function has similar smoothness across the feature space.
5. **No distributional assumption on y:** Unlike Gaussian Processes, SVR makes no assumption on the distribution of the target.
6. **Representative training data:** The support vectors must capture the key structure of the data.

---

## 15. Data Requirements

| Requirement | Needed? | Notes |
|---|---|---|
| Feature Scaling | ✅ **MANDATORY** | Use StandardScaler or MinMaxScaler before SVR |
| Encoding Categorical Features | ✅ Yes | Numerical input required |
| Handling Missing Values | ✅ Yes | Must impute; SVR raises error on NaN |
| Large Dataset | ⚠️ Caution | SVR is $O(n^2)$–$O(n^3)$; slow for $n > 50k$ |
| Outlier Removal | Recommended | Points far outside tube become support vectors; large outliers heavily influence the model |
| Target Scaling | ✅ Recommended | Scale y as well (e.g., to [0,1] or standardize); helps tune ε |

---

## 16. Complexity

### Training
$$O(n^2) \text{ to } O(n^3)$$

The QP problem with $n$ dual variables. In practice, SMO is faster but still quadratic.

**Practical limit:** SVR becomes slow for $n > 50{,}000$. Use LinearSVR or kernel approximations (Nyström) for larger datasets.

### Prediction
$$O(n_{sv} \cdot d)$$

Where $n_{sv}$ = number of support vectors, $d$ = input dimension.  
For linear kernel: $O(d)$ (direct dot product).  
For RBF kernel: $O(n_{sv} \cdot d)$ — can be slow if many support vectors.

### Memory
$$O(n^2)$$

The $n \times n$ kernel matrix must be stored during training. This limits SVR to medium-sized datasets.

---

## 17. Decision Boundary

In regression, the SVR doesn't have a decision boundary — instead, it has a **prediction surface** with an **ε-tube** around it.

### 1D Visualization

```
y
 |          * (outside tube — support vector above)
 |     ε------------------
 |    ---  Tube center   ---
 |   ε------------------
 |  *  *  *  *  * (all inside tube — not support vectors)
 |
 |_____________________________________________ x
```

### 2D Feature Space
The prediction surface is a **hyperplane** (linear kernel) or a **curved surface** (RBF kernel). The ε-tube is a "slab" of thickness $2\varepsilon$ around this surface.

**Key characteristics:**
- Piecewise smooth (not piecewise constant like Decision Trees)
- Can extrapolate slightly (linear kernel) or caps at training range (RBF)
- More flexible than linear regression with kernels

---

## 18. Overfitting & Underfitting

### Overfitting
**Signs:**
- Training MSE ≈ 0, test MSE is much higher
- Almost every training point is a support vector
- The prediction surface wiggles around the training data

**Causes:** High `C`, small `epsilon`, high `gamma`

**Fix:** Decrease `C`, increase `epsilon`, decrease `gamma`

### Underfitting
**Signs:**
- Both training and test MSE are high
- Very few support vectors
- Flat prediction surface that ignores trends

**Causes:** Low `C`, large `epsilon`, low `gamma`

**Fix:** Increase `C`, decrease `epsilon`, increase `gamma`

### Hyperparameter Effect Summary

| Parameter | Increase | Decrease |
|---|---|---|
| `C` ↑ | Less regularization, tighter fit, overfit risk | More regularization, smoother, underfit risk |
| `epsilon` ↑ | Wider tube, fewer SV, simpler model | Narrower tube, more SV, complex model |
| `gamma` ↑ (RBF) | Local influence, complex surface, overfit | Global influence, smooth surface, underfit |

---

## 19. Regularization

### Primary Regularization: `C` Parameter
Minimizing $\frac{1}{2}\|\mathbf{w}\|^2$ is equivalent to **L2 regularization**:

$$\text{Smaller } \|\mathbf{w}\| \Leftrightarrow \text{Larger margin} \Leftrightarrow \text{Flatter function} \Leftrightarrow \text{Stronger regularization}$$

High $C$ → weak regularization (small $\|\mathbf{w}\|$ penalty).  
Low $C$ → strong regularization (large $\|\mathbf{w}\|$ penalty → flatter function).

### Secondary Regularization: `epsilon` (ε)
Larger ε → more points fall inside the tube → fewer support vectors → simpler, more regularized model.

### L1 Regularization: LinearSVR
`LinearSVR` supports `loss='epsilon_insensitive'` with L1 or L2 penalties:
```python
from sklearn.svm import LinearSVR
model = LinearSVR(C=1.0, epsilon=0.1, loss='epsilon_insensitive')
```

---

## 20. Feature Importance

SVR does **not** natively provide feature importance scores.

### For Linear Kernel
The weight vector gives direct importance:
```python
model.coef_  # for LinearSVR or SVR(kernel='linear')
```
Larger $|w_j|$ → feature $j$ is more important.

### For Non-Linear Kernels
Use model-agnostic approaches:

```python
# Permutation Importance
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, scoring='r2')
importances = result.importances_mean
```

**SHAP Values:**
```python
import shap
explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:50])
shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names)
```

**Partial Dependence Plots:** Show the marginal effect of each feature on the SVR prediction.

---

## 21. Advantages

- **Robust to outliers:** ε-insensitive loss ignores small errors; large outliers become support vectors but don't corrupt the whole model
- **Sparse solution:** Only support vectors matter → memory efficient prediction
- **Non-linear regression via kernels:** Models complex relationships without explicit feature engineering
- **Global optimum guaranteed:** Convex QP problem → no local minima
- **Flexible control over precision:** ε lets you specify acceptable prediction tolerance
- **Effective in high dimensions:** Works when features >> samples
- **Theoretically grounded:** Based on structural risk minimization (VC theory)
- **Smooth predictions:** Unlike Decision Tree Regressor's piecewise-constant output

---

## 22. Limitations

- **Slow training:** $O(n^2)$ to $O(n^3)$ — impractical for large datasets
- **Memory intensive:** Kernel matrix is $n \times n$
- **Feature scaling required:** Very sensitive to unscaled features
- **Sensitive to hyperparameters:** `C`, `epsilon`, `gamma` all need careful tuning
- **Cannot extrapolate well:** RBF kernel gives flat predictions outside the training range
- **Hard to interpret:** Non-linear SVR is a black-box
- **No uncertainty quantification:** Provides point predictions only (unlike Gaussian Processes)
- **ε must be pre-specified:** Requires domain knowledge or extensive tuning

---

## 23. Failure Cases

| Scenario | Why It Fails | Solution |
|---|---|---|
| Large dataset ($n > 50k$) | $O(n^3)$ training is too slow | Use LinearSVR or Gradient Boosting |
| Features not scaled | Distance calculations are meaningless | **Always use StandardScaler** |
| Extreme outliers in y | Outliers far outside tube dominate | Use Huber regression or remove outliers |
| Noisy, non-stationary data | RBF assumes local smoothness | Try Gradient Boosting or Neural Networks |
| Need uncertainty estimates | SVR gives only point predictions | Use Gaussian Process Regression |
| Linear relationship | SVR (RBF) is overkill; slower than linear regression | Use Linear Regression or LinearSVR |
| Very large ε vs. data range | All points inside tube; model predicts mean | Decrease ε or scale the target |

---

## 24. Edge Cases

| Situation | Behavior |
|---|---|
| **All points inside tube** | No support vectors; model predicts a flat line (mean-like) |
| **All points outside tube** | Many support vectors; model is complex and may overfit |
| **Single training sample** | Returns that sample's $y$ value for all predictions |
| **Features >> Samples** | SVR works but is slow; use LinearSVR |
| **Target outside training range** | RBF kernel: predictions revert to the mean of training range; cannot extrapolate |
| **Missing values** | Raises ValueError; must impute beforehand |
| **Constant target** | ε-tube covers everything; 0 support vectors; predicts that constant |
| **Very small ε (≈ 0)** | Behaves like standard MAE regression with $C$ regularization |

---

## 25. Evaluation Metrics

### Regression Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Mean absolute deviation; same units as target; robust to outliers |
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Mean squared error; penalizes large errors heavily |
| **RMSE** | $\sqrt{MSE}$ | Root mean squared error; same units as target |
| **R² Score** | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | Proportion of variance explained; 1.0 = perfect |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_i-\hat{y}_i}{y_i}\|$ | Percentage error; useful for scale comparison |

### SVR-Specific Notes
- **Do NOT use SVR's ε-insensitive loss as the evaluation metric on a test set.** Always report standard metrics (MAE, RMSE, R²) for comparability with other models.
- Use **cross-validated RMSE** or **R²** for hyperparameter tuning.
- If your target has been scaled, remember to **inverse-transform predictions** before computing metrics in the original scale.

---

## 26. Comparison with Similar Models

| Feature | SVR | Linear Regression | Decision Tree Regressor | Random Forest Regressor | Gaussian Process |
|---|---|---|---|---|---|
| Non-linearity | ✅ Yes (kernel) | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Feature Scaling | ✅ Required | ✅ Recommended | ❌ Not needed | ❌ Not needed | ✅ Required |
| Training Speed | ❌ Slow ($O(n^3)$) | ✅ Fast | ✅ Fast | ✅ Moderate | ❌ Slow |
| Prediction Speed | ⚠️ Moderate | ✅ Very Fast | ✅ Fast | ✅ Moderate | ✅ Fast |
| Interpretability | ❌ Low | ✅ High | ✅ High | ❌ Low | ⚠️ Moderate |
| Uncertainty Estimates | ❌ No | ⚠️ Partial | ❌ No | ⚠️ Partial | ✅ Yes |
| Outlier Robustness | ✅ Yes (ε-tube) | ❌ Sensitive | ⚠️ Moderate | ✅ Yes | ❌ Sensitive |
| Extrapolation | ⚠️ Limited | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| Global Optimum | ✅ Yes | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| Handles Large Data | ❌ Poor | ✅ Excellent | ✅ Good | ✅ Good | ❌ Poor |

---

## 27. Real-World Applications

| Domain | Application |
|---|---|
| **Finance** | Stock price prediction, currency exchange rate forecasting |
| **Energy** | Electricity load forecasting, solar/wind power output prediction |
| **Healthcare** | Drug response prediction, patient biomarker estimation |
| **Real Estate** | House price estimation from structural features |
| **Manufacturing** | Quality control — predicting product measurements from process parameters |
| **Environmental Science** | Air quality index prediction, rainfall estimation |
| **Robotics** | Trajectory prediction, force estimation in robotic arms |
| **Bioinformatics** | Gene expression level prediction from genomic features |

---

## 28. Scikit-Learn Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ----------------------------
# 1. Load & Prepare Data
# ----------------------------
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# Use a subset for speed (SVR is O(n³))
X, y = X[:2000], y[:2000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 2. Build Pipeline (scaling is MANDATORY for SVR)
# ----------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(
        kernel='rbf',     # 'linear', 'rbf', 'poly', 'sigmoid'
        C=1.0,            # Regularization parameter
        epsilon=0.1,      # ε-tube half-width
        gamma='scale',    # RBF kernel coefficient
    ))
])

pipeline.fit(X_train, y_train)

# ----------------------------
# 3. Predict & Evaluate
# ----------------------------
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)

print("=== SVR Performance ===")
print(f"MAE:        {mae:.4f}")
print(f"MSE:        {mse:.4f}")
print(f"RMSE:       {rmse:.4f}")
print(f"R² (test):  {r2:.4f}")
print(f"R² (train): {r2_train:.4f}")
print(f"Overfit gap: {r2_train - r2:.4f}")

# Number of support vectors
svr_model = pipeline.named_steps['svr']
print(f"\nNumber of Support Vectors: {len(svr_model.support_)}")
print(f"Support Vector Indices (first 10): {svr_model.support_[:10]}")

# ----------------------------
# 4. Hyperparameter Tuning
# ----------------------------
param_grid = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.5, 1.0],
    'svr__gamma': ['scale', 'auto', 0.01, 0.1],
    'svr__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('svr', SVR())]),
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

best_svr = grid_search.best_estimator_
y_pred_best = best_svr.predict(X_test)
print(f"Test R² (best): {r2_score(y_test, y_pred_best):.4f}")

# ----------------------------
# 5. LinearSVR (for larger datasets)
# ----------------------------
linear_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', LinearSVR(C=1.0, epsilon=0.1, max_iter=10000, random_state=42))
])
linear_pipeline.fit(X_train, y_train)
y_pred_linear = linear_pipeline.predict(X_test)
print(f"\nLinearSVR R²: {r2_score(y_test, y_pred_linear):.4f}")

# Feature Importance (LinearSVR only)
coef = linear_pipeline.named_steps['svr'].coef_
sorted_idx = np.argsort(np.abs(coef))[::-1]
print("\nTop Feature Importances (LinearSVR):")
for i in range(len(feature_names)):
    print(f"  {feature_names[sorted_idx[i]]}: {coef[sorted_idx[i]]:.4f}")

# ----------------------------
# 6. Permutation Importance (for non-linear SVR)
# ----------------------------
perm = permutation_importance(
    pipeline, X_test, y_test, n_repeats=10, scoring='r2', random_state=42
)
sorted_perm = np.argsort(perm.importances_mean)[::-1]
print("\nPermutation Importances (RBF SVR):")
for i in range(len(feature_names)):
    print(f"  {feature_names[sorted_perm[i]]}: {perm.importances_mean[sorted_perm[i]]:.4f} "
          f"± {perm.importances_std[sorted_perm[i]]:.4f}")

# ----------------------------
# 7. Visualize: ε-tube effect on 1D data
# ----------------------------
# Simple 1D demo
np.random.seed(42)
X_1d = np.sort(5 * np.random.rand(50, 1), axis=0)
y_1d = np.sin(X_1d).ravel() + np.random.randn(50) * 0.1

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
epsilons = [0.05, 0.2, 0.5]

for ax, eps in zip(axes, epsilons):
    svr_demo = SVR(kernel='rbf', C=100, epsilon=eps, gamma=0.1)
    svr_demo.fit(X_1d, y_1d)

    X_plot = np.linspace(0, 5, 300).reshape(-1, 1)
    y_plot = svr_demo.predict(X_plot)

    ax.scatter(X_1d, y_1d, color='gray', s=20, label='Data', zorder=3)
    ax.plot(X_plot, y_plot, color='navy', lw=2, label=f'SVR (ε={eps})')
    ax.fill_between(X_plot.ravel(), y_plot - eps, y_plot + eps,
                     alpha=0.3, color='skyblue', label='ε-tube')
    ax.scatter(X_1d[svr_demo.support_], y_1d[svr_demo.support_],
               color='red', s=60, zorder=4, label=f'SVs: {len(svr_demo.support_)}')
    ax.set_title(f'ε = {eps}, n_SV = {len(svr_demo.support_)}')
    ax.legend(fontsize=8)

plt.suptitle('SVR ε-tube Effect', fontsize=14)
plt.tight_layout()
plt.savefig("svr_epsilon_tube.png", dpi=150)
plt.show()

# ----------------------------
# 8. Predicted vs Actual Plot
# ----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"SVR: Predicted vs Actual (R² = {r2:.3f})")
plt.tight_layout()
plt.savefig("svr_pred_actual.png", dpi=150)
plt.show()
```

---

## 29. Interview Questions

### Basic

1. **What is SVR and how does it differ from SVC?**  
   SVC finds a hyperplane that maximizes the classification margin. SVR finds a function (regression surface) that keeps predictions within an ε-tube of the true values. Both use the same maximum-margin principle and kernel trick, but applied to different output types (discrete vs. continuous).

2. **What is the ε-tube in SVR?**  
   A corridor of half-width ε around the regression function. Predictions within this corridor incur zero loss. Only points outside the tube (support vectors) contribute to training and influence the model.

3. **What are support vectors in SVR?**  
   Training samples that lie outside the ε-tube — those where $|y_i - f(\mathbf{x}_i)| > \varepsilon$. These are the only points that influence the regression function; all points inside the tube are irrelevant to the learned model.

4. **Why is feature scaling mandatory for SVR?**  
   SVR computes distances and dot products between feature vectors. Features with larger scales dominate these computations, biasing the model toward those features. Always apply StandardScaler before training SVR.

### Intermediate

5. **What is the effect of C in SVR?**  
   $C$ controls how much to penalize predictions that fall outside the ε-tube. High C: tight fit, more support vectors, overfitting risk. Low C: loose fit, fewer support vectors, underfitting risk.

6. **What is the effect of epsilon in SVR?**  
   ε controls the width of the insensitive tube. Large ε: more points inside the tube → fewer support vectors → simpler model → higher training error. Small ε: narrow tube → more support vectors → complex model → lower training error.

7. **How does the kernel trick work in SVR?**  
   Instead of explicitly mapping inputs to a high-dimensional space, SVR replaces dot products $\mathbf{x}_i^T\mathbf{x}_j$ with kernel functions $K(\mathbf{x}_i, \mathbf{x}_j)$ that compute the dot product in the high-dimensional space implicitly. This enables non-linear regression without the computational cost of explicit transformation.

8. **Why does SVR use two slack variables ($\xi_i$ and $\xi_i^*$)?**  
   Because regression errors can be in two directions: the prediction can be too high (overestimation) OR too low (underestimation). $\xi_i$ handles overestimation (prediction above the upper tube wall), $\xi_i^*$ handles underestimation (prediction below the lower tube wall). Classification uses only one direction (wrong class), so SVC has only $\xi_i$.

### Advanced

9. **How is SVR related to ridge regression?**  
   Linear SVR with zero epsilon is mathematically equivalent to Ridge Regression. Both minimize $\|\mathbf{w}\|^2 + C \sum \xi_i$. The key difference is the loss function: Ridge uses squared loss (MSE), SVR uses ε-insensitive loss. When $\varepsilon = 0$, the ε-insensitive loss reduces to MAE, not MSE, so they're not identical but structurally similar.

10. **What happens when ε is set to 0?**  
    Every training point outside the tube has $\xi_i > 0$. In the limit $\varepsilon \to 0$, every point is either on the tube boundary or outside it — this approaches an L1-regularized regression (Lasso). The model tries to fit every point exactly, ignoring no errors.

11. **Why is SVR's optimization problem convex?**  
    The objective $\frac{1}{2}\|\mathbf{w}\|^2 + C\sum(\xi_i + \xi_i^*)$ is a quadratic function (convex), and all constraints are linear (affine). A quadratic objective with linear constraints defines a convex QP — it always has a unique global minimum (or a convex set of minima).

12. **How does LinearSVR differ from SVR with linear kernel?**  
    Both solve linear SVR but use different algorithms. `SVR(kernel='linear')` uses SMO (libsvm) — $O(n^2)$ to $O(n^3)$ complexity. `LinearSVR` uses LIBLINEAR — $O(n \cdot d)$ (linear in samples and features). For large datasets or high-dimensional data, LinearSVR is dramatically faster. The solutions are nearly identical but not numerically the same.

13. **What is the dual form of SVR and why is it important?**  
    The dual reformulates SVR in terms of Lagrange multipliers $\alpha_i, \alpha_i^*$ (one pair per sample). Importance: (1) It shows that only support vectors ($\alpha_i \neq 0$ or $\alpha_i^* \neq 0$) determine the model — sparse solution. (2) It exposes the dot product structure $\mathbf{x}_i^T\mathbf{x}_j$, which can be replaced by kernel functions — enabling non-linear SVR.

14. **Can SVR extrapolate beyond the training data range?**  
    With a linear kernel: yes, to some extent. With RBF kernel: no — predictions converge toward the mean of training data as inputs move further from training points (since kernel values decay to 0 far from training samples). For extrapolation tasks, use linear regression or Gaussian Process Regression.

15. **How would you choose ε for a new regression problem?**  
    Several strategies: (1) **Domain knowledge**: Set ε equal to the acceptable prediction error in the application (e.g., ε = $1000 for house prices). (2) **Rule of thumb**: $\varepsilon \approx 0.1 \times \sigma_y$ where $\sigma_y$ is the standard deviation of the target. (3) **Cross-validation**: Include ε in the grid search over `[0.001, 0.01, 0.1, 0.5, 1.0]`. (4) **After scaling the target**: If you standardize y to mean=0, std=1, start with ε=0.1.
