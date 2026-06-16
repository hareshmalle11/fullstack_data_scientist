# Support Vector Machine (SVM) Classifier — Complete Study Guide

## 1. Model Name

- **Full Name:** Support Vector Machine Classifier (SVC)
- **Category:** Classification
- **Type:** Supervised Learning

---

## 2. Problem It Solves

### What task is it designed for?
SVM Classifier finds the optimal **hyperplane** that best separates data into classes. It is designed for **binary and multi-class classification** tasks where the goal is to find the maximum-margin decision boundary between classes.

Examples:
- Email spam vs. not spam
- Tumor malignant vs. benign (medical imaging)
- Face recognition (image classification)
- Text categorization (news, sentiment)

### Why do we need this model?
- Most classifiers (Logistic Regression, Decision Trees) try to fit the data. SVM tries to find the **widest possible corridor** between classes — this margin maximization leads to better generalization.
- SVM works extremely well in **high-dimensional spaces** (text, genomics).
- Via the **kernel trick**, it can classify data that is **not linearly separable** without explicitly computing features in a higher-dimensional space.
- It is robust in cases where the number of features exceeds the number of samples.

---

## 3. Intuition

### Project Context - Titanic Survival Prediction

In this folder, the SVM Classifier is used for a **Titanic survival classification** task.

The local dataset is:

```text
Titanic-Dataset.csv
```

The target column is:

```text
Survived
```

The app predicts:

```text
Survived
Not Survived
```

The Streamlit app uses these input features:

| Feature | Meaning |
|---|---|
| `Sex` | Passenger sex |
| `Age` | Passenger age |
| `Ticket` | Ticket identifier/text |
| `Fare` | Ticket fare |

The full dataset also contains columns such as `Pclass`, `Name`, `SibSp`, `Parch`, `Cabin`, and `Embarked`, but the current app input form uses the four features above.

The saved model file is:

```text
svm_titanic_model.pkl
```

### Simple Explanation
Imagine two groups of points on a table — red and blue. You want to draw a line between them. Many lines could separate them, but SVM finds the **line with the maximum gap** (margin) from both groups. The closest points to the line on each side are called **support vectors** — they define where the boundary sits.

If the data cannot be separated by a straight line, SVM uses the **kernel trick** to mentally project the data into a higher dimension where it CAN be separated by a flat plane.

### Real-World Analogy
Think of **two armies** facing each other across a field. Instead of drawing the boundary right next to one army, the wisest commander draws it exactly halfway between the two frontlines — creating the maximum buffer zone. The soldiers at the very front of each army (the support vectors) determine where this buffer zone is placed.

Any new soldier (test point) arriving at the field is assigned to whichever army's territory they fall into.

---

## 4. Mathematical Foundation

### Core Formula — The Hyperplane

$$\mathbf{w}^T \mathbf{x} + b = 0$$

This is the decision boundary. For classification:

$$\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$$

- If $\mathbf{w}^T \mathbf{x} + b \geq +1$ → Class +1
- If $\mathbf{w}^T \mathbf{x} + b \leq -1$ → Class -1

### Margin Formula

The **margin** is the distance between the two parallel margin hyperplanes:

$$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$

### Explain Every Term

| Symbol | Meaning |
|---|---|
| $\mathbf{w}$ | Weight vector (normal to the decision boundary hyperplane) |
| $b$ | Bias (offset of the hyperplane from the origin) |
| $\mathbf{x}$ | Input feature vector |
| $\hat{y}$ | Predicted class label (+1 or -1) |
| $\|\mathbf{w}\|$ | Euclidean norm of the weight vector |
| $\frac{2}{\|\mathbf{w}\|}$ | Width of the margin between the two classes |
| Support Vectors | Training samples closest to the decision boundary |

### Margin Planes

$$\mathbf{w}^T \mathbf{x} + b = +1 \quad \text{(margin plane for class +1)}$$
$$\mathbf{w}^T \mathbf{x} + b = -1 \quad \text{(margin plane for class -1)}$$

The distance from any point to the decision boundary:
$$d = \frac{|\mathbf{w}^T \mathbf{x} + b|}{\|\mathbf{w}\|}$$

---

## 5. Objective Function

### Hard Margin SVM (linearly separable data)

**Maximize the margin:**
$$\max_{\mathbf{w}, b} \frac{2}{\|\mathbf{w}\|}$$

Equivalently, **minimize:**
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$$

**Subject to the constraints (all points correctly classified):**
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i$$

**Intuition:**
- Minimizing $\|\mathbf{w}\|$ is equivalent to maximizing the margin $\frac{2}{\|\mathbf{w}\|}$.
- The constraint ensures every training point is on the correct side of its margin plane.
- $\frac{1}{2}$ is added for mathematical convenience (it cancels with the 2 from the derivative).

### Soft Margin SVM (non-separable data) — with C parameter

Real data is rarely perfectly separable. We introduce **slack variables** $\xi_i \geq 0$ that allow some misclassification:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

**Subject to:**
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i \quad \forall i, \quad \xi_i \geq 0$$

**Intuition of terms:**
- $\frac{1}{2}\|\mathbf{w}\|^2$: Maximize margin (minimize $\|\mathbf{w}\|$)
- $C \sum \xi_i$: Penalize margin violations
- $C$: Trade-off parameter — higher $C$ → smaller margin, fewer violations; lower $C$ → larger margin, more violations allowed

---

## 6. Loss Function Derivation

### Hinge Loss

The SVM loss function is called **Hinge Loss**:

$$L_i = \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$

**Per-sample intuition:**
- If the prediction $\hat{f}(x_i) = \mathbf{w}^T \mathbf{x}_i + b$ is correct **and** outside the margin: $y_i \hat{f}(x_i) \geq 1$ → Loss = 0
- If the prediction is inside the margin or wrong: $y_i \hat{f}(x_i) < 1$ → Loss = $1 - y_i \hat{f}(x_i) > 0$

**Total Hinge Loss (equivalent to soft-margin SVM objective):**
$$\mathcal{L} = \frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac{1}{n}\sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$

Where $\lambda = \frac{1}{C \cdot n}$ links the regularization to the $C$ hyperparameter.

### Geometric Intuition of Hinge Loss

```
Loss
 |
1|----\
 |     \
 |      \  (in margin)
 |       \
0|________\___________ y_i * f(x_i)
          1
```

- For correctly classified points far from the boundary: Loss = 0 (no penalty)
- For points inside the margin (even if correctly classified): Linear penalty
- For misclassified points: Large penalty

### Gradient of Hinge Loss

For a single sample:

$$\frac{\partial L_i}{\partial \mathbf{w}} = \begin{cases} -y_i \mathbf{x}_i & \text{if } y_i(\mathbf{w}^T \mathbf{x}_i + b) < 1 \\ 0 & \text{otherwise} \end{cases}$$

$$\frac{\partial L_i}{\partial b} = \begin{cases} -y_i & \text{if } y_i(\mathbf{w}^T \mathbf{x}_i + b) < 1 \\ 0 & \text{otherwise} \end{cases}$$

---

## 7. Optimization Method

### Primal Problem (direct approach)
The soft-margin SVM objective is a **Quadratic Programming (QP)** problem:
- Quadratic objective: $\frac{1}{2}\|\mathbf{w}\|^2$
- Linear constraints: $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$
- Solved using QP solvers (e.g., SMO algorithm)

### Dual Problem (for kernel trick)
Using **Lagrange multipliers** $\alpha_i \geq 0$, we reformulate as:

$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

**Subject to:**
$$0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n} \alpha_i y_i = 0$$

**Key insight:** The dual formulation only involves **dot products** $\mathbf{x}_i^T \mathbf{x}_j$ — these can be replaced by **kernel functions** $K(\mathbf{x}_i, \mathbf{x}_j)$ to implicitly compute dot products in a higher-dimensional space!

### Recovery of Parameters
$$\mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i$$
$$b = y_j - \mathbf{w}^T \mathbf{x}_j \quad \text{for any support vector } j \text{ where } 0 < \alpha_j < C$$

### SMO Algorithm (Sequential Minimal Optimization)
scikit-learn uses **libsvm** which implements SMO:
- Selects two $\alpha_i, \alpha_j$ at a time
- Optimizes the two variables while keeping others fixed
- Analytic solution exists for 2-variable subproblem
- Repeat until convergence

### Kernel Functions

| Kernel | Formula | When to Use |
|---|---|---|
| Linear | $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$ | Linearly separable data, high-dim text |
| RBF (Gaussian) | $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | Non-linear, general purpose |
| Polynomial | $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$ | Polynomial relationships |
| Sigmoid | $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)$ | Neural-network-like |

---

## 8. Training Workflow

```
Input: Labeled Training Data (X, y where y ∈ {-1, +1})
        ↓
Scale Features (MANDATORY — SVM is distance-based)
        ↓
Choose Kernel (Linear / RBF / Polynomial)
        ↓
Set Hyperparameters: C, gamma (for RBF), degree (for poly)
        ↓
Solve Quadratic Programming Problem (via SMO / libsvm):
  - Find optimal Lagrange multipliers α_i
  - Most α_i = 0 (non-support vectors)
  - Non-zero α_i correspond to support vectors
        ↓
Compute w = Σ α_i * y_i * x_i
Compute b from support vectors
        ↓
Model = (w, b) or equivalently the support vectors + α_i values
        ↓
Prediction: sign(w^T x + b) or sign(Σ α_i y_i K(x_i, x) + b)
```

---

## 9. Worked Numerical Example

### Dataset (2D, Binary Classification)

| Sample | $x_1$ | $x_2$ | $y$ |
|---|---|---|---|
| 1 | 1 | 1 | +1 |
| 2 | 2 | 2 | +1 |
| 3 | 3 | 1 | -1 |
| 4 | 4 | 2 | -1 |

### Step 1: Find the separating hyperplane

By inspection, the line $x_1 = 2.5$ separates the classes.

In the form $\mathbf{w}^T \mathbf{x} + b = 0$:
$$w_1 x_1 + w_2 x_2 + b = 0$$

Let's try: $\mathbf{w} = [1, 0]$, $b = -2.5$

$$f(\mathbf{x}) = x_1 - 2.5$$

### Step 2: Check margin constraints

For class +1 (must satisfy $f \geq +1$):
- Sample 1: $f = 1 - 2.5 = -1.5$ ❌ Not satisfied

Try: $\mathbf{w} = [2, 0]$, $b = -5$: $f(\mathbf{x}) = 2x_1 - 5$

- Sample 1: $2(1) - 5 = -3$ → $y \cdot f = (+1)(-3) = -3 < 1$ ❌

Let's solve it properly. With $\mathbf{w} = [1, 0]$, $b = -2.5$:
- Normalize: Scale $\mathbf{w}$ so margin planes are at $\pm 1$.
- Closest +1 point: $x_1 = 2$ → $f = 2 - 2.5 = -0.5$
- Closest -1 point: $x_1 = 3$ → $f = 3 - 2.5 = 0.5$

Scale up by 2: $\mathbf{w} = [2, 0]$, $b = -5$
- Sample 2 (closest +1): $f = 2(2) - 5 = -1$ → $y \cdot f = +1 \cdot (-1) = -1$ ❌

Swap signs: $\mathbf{w} = [2, 0]$, $b = -5$, labels: class on right is +1.
Assign: $y = +1$ for $x_1 \geq 3$, $y = -1$ for $x_1 \leq 2$.

With $\mathbf{w} = [2, 0]$, $b = -5$:
- Sample 3 ($x_1=3$, $y=+1$): $f = 6-5 = 1$ → $y \cdot f = 1 \cdot 1 = 1$ ✅
- Sample 4 ($x_1=4$, $y=+1$): $f = 8-5 = 3$ → $y \cdot f = 1 \cdot 3 = 3 \geq 1$ ✅
- Sample 1 ($x_1=1$, $y=-1$): $f = 2-5 = -3$ → $y \cdot f = (-1)(-3) = 3 \geq 1$ ✅
- Sample 2 ($x_1=2$, $y=-1$): $f = 4-5 = -1$ → $y \cdot f = (-1)(-1) = 1$ ✅

### Step 3: Compute Margin

$$\text{Margin} = \frac{2}{\|\mathbf{w}\|} = \frac{2}{\sqrt{2^2 + 0^2}} = \frac{2}{2} = 1$$

**Support Vectors:** Samples 2 and 3 (they satisfy $y \cdot f = 1$ exactly — on the margin planes).

### Step 4: Predict new point $\mathbf{x} = [2.8, 1.5]$

$$f = 2(2.8) + 0(1.5) - 5 = 5.6 - 5 = 0.6 > 0$$

$$\hat{y} = \text{sign}(0.6) = +1$$

---

## 10. Full Manual Training Example

### Dataset

| ID | $x_1$ | $x_2$ | $y$ |
|---|---|---|---|
| A | 1 | 2 | +1 |
| B | 2 | 3 | +1 |
| C | 4 | 1 | -1 |
| D | 5 | 2 | -1 |
| E | 3 | 2 | ? |

### Initial Setup
We want $\mathbf{w}$ and $b$ such that:
- All +1 samples: $\mathbf{w}^T\mathbf{x} + b \geq +1$
- All -1 samples: $\mathbf{w}^T\mathbf{x} + b \leq -1$

### Using the Dual (Lagrangian) Approach

For this small dataset, let's verify the solution $\mathbf{w} = [1, 0]$, $b = -3$:

| ID | $f(x) = x_1 - 3$ | $y \cdot f(x)$ | Constraint ($\geq 1$)? |
|---|---|---|---|
| A | $1-3 = -2$ | $(+1)(-2) = -2$ | ❌ |
| B | $2-3 = -1$ | $(+1)(-1) = -1$ | ✅ (on margin) |
| C | $4-3 = +1$ | $(-1)(+1) = -1$ | ✅ (on margin) |
| D | $5-3 = +2$ | $(-1)(+2) = -2$ | ✅ |

Point A violates the constraint → Soft margin needed. With $C$ large, the penalty $C \cdot \xi_A$ forces $\xi_A$ small. The slack for A: $\xi_A = 1 - (-2) = 3$.

### Prediction for E ($x_1=3, x_2=2$):
$$f = 3 - 3 = 0$$
$$\hat{y} = \text{sign}(0) = \text{on boundary — ambiguous}$$

Adjust: Try $\mathbf{w} = [1.5, 0]$, $b = -4.5$:
- B: $f = 1.5(2)-4.5 = -1.5$ → $y \cdot f = -1.5 < 1$ → still violates

This shows why we need QP solvers for non-trivial datasets. For this dataset, no perfect linear separator exists, confirming the need for either soft margin ($C$) or a kernel.

### With RBF Kernel (conceptually):
Project to higher-dim space where a hyperplane can separate A and B from C and D perfectly. The kernel handles this automatically without explicit projection.

---

## 11. Parameters (Learned Values)

| Parameter | Meaning |
|---|---|
| $\mathbf{w}$ | Weight vector (normal to decision boundary) |
| $b$ | Bias term (shifts the boundary) |
| $\alpha_i$ | Lagrange multipliers (one per training sample) |
| Support Vectors | Training samples with $\alpha_i > 0$ (non-zero multipliers) |
| Dual Coefficients | $\alpha_i \cdot y_i$ for each support vector |

**Key insight:** Only support vectors (the points closest to the boundary) matter for prediction. All other training samples have $\alpha_i = 0$ and can be "forgotten" after training.

---

## 12. Hyperparameters

| Hyperparameter | Default | Purpose |
|---|---|---|
| `C` | 1.0 | Regularization: trade-off between margin width and violations |
| `kernel` | 'rbf' | Kernel function: 'linear', 'rbf', 'poly', 'sigmoid' |
| `gamma` | 'scale' | RBF/poly/sigmoid kernel coefficient |
| `degree` | 3 | Degree for polynomial kernel |
| `coef0` | 0.0 | Independent term in poly/sigmoid kernels |
| `tol` | 1e-3 | Stopping tolerance for solver |
| `max_iter` | -1 | Max iterations (-1 = no limit) |
| `class_weight` | None | Weight classes for imbalanced data |
| `decision_function_shape` | 'ovr' | Multi-class: 'ovr' (one-vs-rest) or 'ovo' (one-vs-one) |
| `probability` | False | Enable probability estimates (adds Platt scaling cost) |

---

## 13. Why Each Hyperparameter Exists

### `C` — Regularization Parameter
**Problem:** Perfect margin maximization may misclassify some training points.  
**Solution:** $C$ controls the penalty for margin violations.
- **High C** (e.g., 100): Hard margin, few violations allowed, narrow margin → higher variance, lower bias → overfitting risk
- **Low C** (e.g., 0.01): Soft margin, many violations allowed, wide margin → higher bias, lower variance → underfitting risk
- **Rule of thumb:** Start at $C=1$, tune via cross-validation over $[0.001, 0.01, 0.1, 1, 10, 100, 1000]$

### `gamma` — RBF Kernel Width
**Problem:** Controls how far the influence of a single training example reaches.  
- **High gamma:** Small influence radius → each point only affects nearby decisions → complex, wiggly boundary → overfitting
- **Low gamma:** Large influence radius → smooth decision boundary → underfitting
- **`gamma='scale'`** (default): $\gamma = \frac{1}{n_{features} \cdot \text{Var}(X)}$
- **`gamma='auto'`**: $\gamma = \frac{1}{n_{features}}$

### `kernel` Selection Guide
- **Linear:** Text classification, high-dimensional sparse data, linearly separable problems
- **RBF:** General purpose default — works well for most tabular data
- **Polynomial:** Image classification, when polynomial relationships exist
- **Sigmoid:** Neural-network-like behavior (less commonly used)

---

## 14. Assumptions

1. **Binary classification natively:** SVM is a binary classifier. Multi-class requires One-vs-One or One-vs-Rest decomposition.
2. **Feature scaling matters:** SVM uses distances — features on larger scales dominate. **Always scale features before SVM.**
3. **No probabilistic output natively:** SVM outputs a decision score ($\mathbf{w}^T\mathbf{x} + b$), not a probability. Probabilities require Platt scaling.
4. **Kernel assumption:** The chosen kernel must match the true data structure (RBF works well in general).
5. **Labeled data with consistent labeling:** Assumes clean, consistently labeled training data.
6. **No explicit distributional assumption:** SVM is non-parametric in nature.

---

## 15. Data Requirements

| Requirement | Needed? | Notes |
|---|---|---|
| Feature Scaling | ✅ **MANDATORY** | SVM is distance-based; use StandardScaler or MinMaxScaler |
| Encoding Categorical Features | ✅ Yes | Use OrdinalEncoder or OneHotEncoder |
| Handling Missing Values | ✅ Yes | Must impute before training |
| Large Dataset | ⚠️ Caution | SVM scales as $O(n^2)$ to $O(n^3)$; slow for n > 100k |
| Balanced Classes | Recommended | Use `class_weight='balanced'` for imbalanced data |
| Outlier Removal | Recommended | Outliers near boundary become spurious support vectors |

---

## 16. Complexity

### Training
$$O(n^2) \text{ to } O(n^3)$$

Solving the QP problem has cubic complexity in the worst case. SMO reduces this but is still quadratic for large $n$.

**Practical limit:** SVM becomes impractical for $n > 100{,}000$ samples without approximations (e.g., LinearSVC).

### Prediction
$$O(n_{sv} \cdot d)$$

Where $n_{sv}$ = number of support vectors, $d$ = input dimensions.

For **linear kernel**: $O(d)$ (just dot product with $\mathbf{w}$).  
For **RBF kernel**: $O(n_{sv} \cdot d)$ — slower with many support vectors.

### Memory
$$O(n^2)$$

The kernel matrix $K$ (n × n) must be stored for non-linear kernels.

---

## 17. Decision Boundary

### Linear SVM
A straight line (2D), plane (3D), or hyperplane (n-D):

```
x₂
 |      Support Vectors
 |    * |           |
 | +  * |   Margin  | *  -
 | +    |           |    -
 |   +  |     ↑     |  -
 |      |  Decision |
 |______|___Boundary|___________ x₁
        w^T x + b = -1    w^T x + b = +1
                   w^T x + b = 0
```

### Non-Linear SVM (RBF Kernel)
Creates curved, complex decision boundaries in the original space — while remaining a flat hyperplane in the kernel-mapped space.

**Linear boundary:** Yes, in kernel space; appears non-linear in original feature space.

---

## 18. Overfitting & Underfitting

### Overfitting
**Signs:**
- Training accuracy ≈ 100%, test accuracy much lower
- Many support vectors (high $n_{sv}$) — almost every point is a support vector
- Very narrow margin

**Causes:** High `C`, high `gamma` (for RBF)

**Fix:** Decrease `C`, decrease `gamma`, use cross-validation

### Underfitting
**Signs:**
- Both training and test accuracy are low
- Very few support vectors
- Very wide margin (too much regularization)

**Causes:** Very low `C`, very low `gamma`

**Fix:** Increase `C`, increase `gamma`

### Diagnostic Rule of Thumb

| $C$ | $\gamma$ (RBF) | Effect |
|---|---|---|
| High | High | Overfit |
| High | Low | Complex but smooth |
| Low | High | Strange localized effects |
| Low | Low | Underfit |

---

## 19. Regularization

### C Parameter (L2-style regularization)
The $C$ parameter controls the regularization:

$$\min \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \xi_i$$

- Smaller $C$ = stronger regularization (larger margin, more violations allowed)
- Equivalent to an **L2 penalty** on the weight vector

### L1 Regularization (LinearSVC)
`LinearSVC` with `penalty='l1'` adds an L1 penalty:

$$\min \|\mathbf{w}\|_1 + C \sum_i \xi_i$$

This produces **sparse weight vectors** — many weights become exactly 0 (automatic feature selection).

### L2 Regularization (default)
Standard SVC uses L2 (squared norm):

$$\min \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_i \xi_i$$

All features get small but non-zero weights.

---

## 20. Feature Importance

SVM does **not** natively provide feature importance like Decision Trees.

### For Linear Kernel
The weight vector $\mathbf{w}$ gives importance:
```python
importances = model.coef_[0]  # LinearSVC or SVC with linear kernel
```
Larger $|w_j|$ → feature $j$ is more important.

### For Non-Linear Kernels (RBF, Poly)
No direct feature importance. Use instead:
- **Permutation Importance** (model-agnostic)
- **SHAP values** (game-theoretic feature attribution)
- **Partial Dependence Plots**

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10)
```

---

## 21. Advantages

- **Maximum margin principle:** Theoretically motivated generalization guarantee
- **Effective in high dimensions:** Works well when features >> samples (text, genomics)
- **Kernel trick:** Can model complex non-linear boundaries without explicit feature engineering
- **Sparse solution:** Only support vectors matter — memory efficient for prediction
- **Robust to outliers:** Boundary determined only by support vectors, not all training points
- **Global optimum:** The QP problem is convex — always finds the global optimum, no local minima
- **Works with small datasets:** SVM shines when $n$ is small but $d$ is large

---

## 22. Limitations

- **Slow training:** $O(n^2)$ to $O(n^3)$ — impractical for large datasets
- **Memory intensive:** Kernel matrix is $n \times n$
- **Feature scaling required:** Very sensitive to scale
- **No probability output natively:** Requires expensive Platt scaling (`probability=True`)
- **Sensitive to C and gamma:** Poor hyperparameter choice drastically degrades performance
- **Black-box with kernels:** Non-linear SVMs are difficult to interpret
- **Noisy data:** Hard to tune when classes overlap heavily
- **Multi-class is indirect:** No native multi-class support; uses OvR or OvO decomposition

---

## 23. Failure Cases

| Scenario | Why It Fails | Solution |
|---|---|---|
| Large dataset (n > 100k) | Training is $O(n^3)$ — too slow | Use LinearSVC or SGDClassifier |
| Features not scaled | Distance calculations are dominated by large-scale features | **Always use StandardScaler** |
| Highly overlapping classes | No clean margin; many slack variables | Use Random Forest or Gradient Boosting |
| Many irrelevant features | Support vectors become noisy | Feature selection + L1 penalty |
| Need probability estimates | SVM is not probabilistic | Set `probability=True` or use Logistic Regression |
| Very imbalanced classes | Majority class dominates support vectors | Set `class_weight='balanced'` |

---

## 24. Edge Cases

| Situation | Behavior |
|---|---|
| **Tiny dataset (< 20 samples)** | Works well; SVM was designed for small data |
| **Features >> Samples** | SVM works well; kernel matrix is small; use linear kernel |
| **Perfectly separable data** | Hard margin SVM ($C=\infty$) finds exact boundary; risk of non-convergence |
| **Completely overlapping classes** | Even large $C$ can't help; try different kernel or different model |
| **All samples are support vectors** | Model is overfit or data is not linearly separable for chosen kernel |
| **Only 1 class in training** | Raises error; no boundary can be defined |
| **Duplicate samples** | Handled; duplicates may both become support vectors |

---

## 25. Evaluation Metrics

### Binary Classification

| Metric | Formula | When to Use |
|---|---|---|
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | Balanced classes |
| **Precision** | $\frac{TP}{TP+FP}$ | False positives are costly |
| **Recall** | $\frac{TP}{TP+FN}$ | False negatives are costly |
| **F1 Score** | $2 \cdot \frac{P \cdot R}{P+R}$ | Imbalanced classes |
| **ROC-AUC** | Area under ROC curve | Ranking quality (use `decision_function`) |
| **Hinge Loss** | $\frac{1}{n}\sum \max(0, 1-y_i f(x_i))$ | Direct SVM loss monitoring |

**Note:** Use `model.decision_function(X)` (not `predict_proba`) for ROC-AUC with SVM unless `probability=True`.

### Multi-class Classification
Same metrics as above, averaged:
- `average='macro'`: Equal weight per class
- `average='weighted'`: Weight by class frequency
- `average='micro'`: Aggregate all TP/FP/FN globally

---

## 26. Comparison with Similar Models

| Feature | SVM Classifier | Logistic Regression | Random Forest | Neural Network |
|---|---|---|---|---|
| Decision Boundary | Linear / non-linear | Linear (logistic) | Non-linear | Non-linear |
| Training Speed | ❌ Slow (O(n³)) | ✅ Fast | ✅ Moderate | ❌ Slow |
| Feature Scaling | ✅ Required | ✅ Recommended | ❌ Not needed | ✅ Required |
| Probability Output | ⚠️ Via Platt | ✅ Native | ✅ Native | ✅ Native |
| Interpretability | ⚠️ Linear only | ✅ High | ❌ Low | ❌ Low |
| High-dim Data | ✅ Excellent | ✅ Good | ⚠️ OK | ✅ Good |
| Small Dataset | ✅ Excellent | ✅ Good | ⚠️ OK | ❌ Poor |
| Large Dataset | ❌ Poor | ✅ Good | ✅ Good | ✅ Good |
| Kernel Trick | ✅ Yes | ❌ No | ❌ No | ✅ Implicit |
| Global Optimum | ✅ Yes (convex) | ✅ Yes (convex) | ❌ No | ❌ No |

---

## 27. Real-World Applications

| Domain | Application |
|---|---|
| **Bioinformatics** | Cancer classification from gene expression data |
| **Image Recognition** | Handwritten digit recognition (classic MNIST) |
| **Text Classification** | Spam filtering, sentiment analysis, news categorization |
| **Medical Diagnosis** | Disease detection from symptoms or imaging features |
| **Finance** | Credit card fraud detection, stock movement prediction |
| **Computer Vision** | Face detection (Viola-Jones uses SVM at its core) |
| **NLP** | Named entity recognition, part-of-speech tagging |
| **Intrusion Detection** | Network anomaly detection in cybersecurity |

---

## 28. Scikit-Learn Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, RocCurveDisplay)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ----------------------------
# 1. Load & Prepare Data
# ----------------------------
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 2. Scale Features (MANDATORY!)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 3. Train SVM Classifier
# ----------------------------
# Option A: Standard SVC (for small-medium datasets)
svc = SVC(
    C=1.0,            # Regularization
    kernel='rbf',     # 'linear', 'rbf', 'poly', 'sigmoid'
    gamma='scale',    # Kernel coefficient
    class_weight='balanced',  # Handle imbalance
    probability=True, # Enable probability estimates
    random_state=42
)
svc.fit(X_train_scaled, y_train)

# Option B: LinearSVC (for large datasets — much faster)
linear_svc = LinearSVC(C=1.0, max_iter=5000, random_state=42)
linear_svc.fit(X_train_scaled, y_train)

# ----------------------------
# 4. Predict & Evaluate
# ----------------------------
y_pred = svc.predict(X_test_scaled)
y_proba = svc.predict_proba(X_test_scaled)[:, 1]
y_score = svc.decision_function(X_test_scaled)

print("=== SVC Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_score):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nNumber of Support Vectors: {svc.n_support_}")

# ----------------------------
# 5. Pipeline (best practice)
# ----------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print(f"\nPipeline Accuracy: {accuracy_score(y_test, y_pred_pipe):.4f}")

# ----------------------------
# 6. Hyperparameter Tuning
# ----------------------------
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'svc__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    Pipeline([('scaler', StandardScaler()),
              ('svc', SVC(probability=True, random_state=42))]),
    param_grid,
    cv=StratifiedKFold(5),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Test Accuracy (best): {accuracy_score(y_test, y_pred_best):.4f}")

# ----------------------------
# 7. Feature Importance (Linear Kernel)
# ----------------------------
linear_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear', C=1.0))
])
linear_pipe.fit(X_train, y_train)

coef = linear_pipe.named_steps['svc'].coef_[0]
sorted_idx = np.argsort(np.abs(coef))[::-1]
print("\nTop 10 Feature Importances (Linear SVM):")
for i in range(10):
    print(f"  {feature_names[sorted_idx[i]]}: {coef[sorted_idx[i]]:.4f}")

# ----------------------------
# 8. ROC Curve
# ----------------------------
RocCurveDisplay.from_estimator(
    Pipeline([('scaler', StandardScaler()),
              ('svc', SVC(kernel='rbf', probability=True, random_state=42))]).fit(X_train, y_train),
    X_test, y_test
)
plt.title("SVM ROC Curve")
plt.savefig("svm_roc_curve.png", dpi=150)
plt.show()

# ----------------------------
# 9. Multi-Class SVM
# ----------------------------
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier

X_mc, y_mc = load_iris(return_X_y=True)
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    X_mc, y_mc, test_size=0.2, random_state=42
)

multi_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=10, gamma='scale',
                decision_function_shape='ovr'))  # One-vs-Rest
])
multi_svm.fit(X_train_mc, y_train_mc)
print(f"\nMulti-class SVM Accuracy: {multi_svm.score(X_test_mc, y_test_mc):.4f}")
```

---

## 29. Interview Questions

### Basic

1. **What is a Support Vector Machine?**  
   A supervised learning algorithm that finds the optimal hyperplane maximizing the margin between two classes. Only support vectors (the closest points to the boundary) determine the decision boundary.

2. **What are support vectors?**  
   The training samples that lie on or inside the margin boundary ($\alpha_i > 0$). They are the most "informative" points — removing non-support-vector samples doesn't change the model at all.

3. **What is the margin in SVM?**  
   The distance between the two parallel margin hyperplanes ($\mathbf{w}^T\mathbf{x}+b = \pm 1$). Equal to $\frac{2}{\|\mathbf{w}\|}$. SVM maximizes this margin.

4. **Why is feature scaling mandatory for SVM?**  
   SVM computes distances and dot products. Features with larger scales dominate these computations. Without scaling, large-scale features unfairly dominate the boundary. Always use StandardScaler.

### Intermediate

5. **What is the kernel trick?**  
   Instead of explicitly transforming features to a high-dimensional space, we replace the dot product $\mathbf{x}_i^T \mathbf{x}_j$ with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$ that computes the dot product in the high-dimensional space implicitly. This allows non-linear boundaries at the cost of computing dot products in the original space.

6. **What is the effect of the C parameter?**  
   $C$ controls the trade-off between margin width and training error. High $C$: hard margin, fewer violations, narrow margin, risk of overfitting. Low $C$: soft margin, many violations allowed, wide margin, risk of underfitting.

7. **What is the effect of gamma in RBF kernel?**  
   $\gamma = \frac{1}{2\sigma^2}$ — the inverse of the Gaussian width. High $\gamma$: narrow Gaussian, each sample has local influence → complex boundary → overfit. Low $\gamma$: wide Gaussian, smooth global boundary → underfit.

8. **How does SVM handle multi-class problems?**  
   Natively binary. Extended via: (1) **One-vs-Rest (OvR)**: Train $k$ classifiers (each class vs. all others), predict the class with the highest decision score. (2) **One-vs-One (OvO)**: Train $\binom{k}{2}$ classifiers for each pair of classes, majority vote. scikit-learn SVC uses OvO by default.

### Advanced

9. **What is the dual problem and why is it useful?**  
   The Lagrangian dual reformulates the QP in terms of $\alpha_i$ (one per sample) instead of $\mathbf{w}$ and $b$. It's useful because: (1) the solution $\mathbf{w} = \sum \alpha_i y_i \mathbf{x}_i$ shows that only support vectors ($\alpha_i > 0$) matter; (2) the dual only requires dot products, enabling the kernel trick.

10. **Why is SVM's optimization problem convex?**  
    The objective $\frac{1}{2}\|\mathbf{w}\|^2 + C\sum\xi_i$ is quadratic (convex), and the constraints are linear (also convex). A convex problem with linear constraints has a unique global minimum — SVM never gets stuck in local optima.

11. **How does hinge loss compare to logistic loss?**  
    Hinge loss: $\max(0, 1 - y\hat{f})$ — exactly zero for correctly classified points far from the boundary (sparse gradients; only support vectors contribute). Logistic loss: $\log(1 + e^{-y\hat{f}})$ — never exactly zero; all points contribute. Hinge loss creates a sparse solution.

12. **What is Platt scaling in SVM?**  
    SVM outputs a decision score, not a probability. Platt scaling fits a logistic regression on top of the SVM's decision function scores (using cross-validation) to calibrate probabilities. Activated via `probability=True` — adds significant training cost.

13. **When would you choose LinearSVC over SVC(kernel='linear')?**  
    LinearSVC scales as $O(n \cdot d)$ (linear in samples and features) using a different optimizer (LIBLINEAR). SVC with linear kernel scales as $O(n^2)$ to $O(n^3)$. For large $n$ or high-dimensional sparse data, LinearSVC is dramatically faster.

14. **What does it mean if almost all training samples are support vectors?**  
    The model is likely overfitting (C too high) or the classes heavily overlap. A well-trained SVM should have far fewer support vectors than training samples — the sparser, the better generalization.

15. **How does SVM relate to logistic regression?**  
    Both are linear classifiers minimizing convex loss functions with regularization. Key differences: SVM uses hinge loss (sparse solution), logistic regression uses log-loss (probabilistic output). SVM maximizes the margin; logistic regression maximizes the log-likelihood. In high dimensions with large datasets, their performance is often similar.
