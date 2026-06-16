# Decision Tree Classifier — Complete Study Guide

## 1. Model Name

- **Full Name:** Decision Tree Classifier
- **Category:** Classification
- **Type:** Supervised Learning

---

## 2. Problem It Solves

### What task is it designed for?
Decision Tree Classifier is designed to classify data points into discrete categories (classes) by learning a series of if-else rules from the training data.

Examples:
- Is this email spam or not spam?
- Does this patient have diabetes or not?
- Which flower species does this sample belong to?

### Why do we need this model?
- Many real-world problems require mapping inputs to a finite set of categories.
- Decision Trees are highly interpretable — you can visualize exactly why a prediction was made.
- They handle both numerical and categorical features without heavy preprocessing.
- They require no assumptions about the distribution of data.

---

## 3. Intuition

### Project Context - Rain Prediction

In this folder, the Decision Tree Classifier is used for a **weather classification** task.

The local dataset is:

```text
weather_forecast_data.csv
```

The target column is:

```text
Rain
```

The app predicts one of two classes:

```text
rain
no rain
```

The input features used by the Streamlit app are:

| Feature | Meaning |
|---|---|
| `Temperature` | Air temperature |
| `Humidity` | Moisture level in the air |
| `Wind_Speed` | Wind speed |
| `Cloud_Cover` | Amount of cloud coverage |
| `Pressure` | Atmospheric pressure |

For this project, a decision tree learns weather rules such as:

```text
if Cloud_Cover is high and Humidity is high, predict rain
else predict no rain
```

The saved model file is:

```text
decision_tree_rain_model.pkl
```

### Simple Explanation
Imagine you are playing **20 Questions**. You ask yes/no questions one by one to narrow down the answer. Each question splits the possibilities further until you arrive at one answer. That is exactly what a Decision Tree does — it asks questions about the features, one at a time, splitting the dataset at each step, until each group is mostly one class.

### Real-World Analogy
Think of a **doctor diagnosing a disease**:
- First question: "Does the patient have a fever?" → Yes / No
- If Yes → "Is the fever above 39°C?" → Yes / No
- If Yes → "Does the patient have a rash?" → Yes → **Diagnose: Dengue**

The doctor follows a decision path based on the patient's answers. A Decision Tree Classifier builds this exact type of structured question-answering system automatically from data.

---

## 4. Mathematical Foundation

### Core Formula — Impurity Measures

#### Gini Impurity
$$Gini(S) = 1 - \sum_{i=1}^{C} p_i^2$$

#### Information Gain (Entropy-based)
$$Entropy(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

$$Information\ Gain = Entropy(Parent) - \sum_{child} \frac{|child|}{|parent|} \cdot Entropy(child)$$

### Explain Every Term

| Symbol | Meaning |
|---|---|
| S | Current dataset / node |
| C | Total number of classes |
| $p_i$ | Proportion of class $i$ in the node |
| $\log_2$ | Logarithm base 2 (used in entropy) |
| $\|child\|$ | Number of samples in a child node |
| $\|parent\|$ | Number of samples in the parent node |
| $Gini(S)$ | Impurity of the node (0 = pure, 0.5 = max impurity for binary) |
| $Entropy(S)$ | Disorder in the node (0 = pure, 1 = max disorder for binary) |

---

## 5. Objective Function

The Decision Tree Classifier tries to **minimize impurity** at each split. The model selects the feature and threshold that results in the **maximum Information Gain** (or minimum Gini Impurity) at each step.

$$Information\ Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot Entropy(S_v)$$

- **Why this objective?** Splitting on the most informative feature first creates the most efficient tree — reaching pure leaf nodes with the fewest splits.
- **When impurity is high:** The node contains a mix of classes → we need to split further.
- **When impurity is 0:** The node is pure — all samples belong to one class → this becomes a leaf node.

---

## 6. Loss Function Derivation

### Entropy Derivation

Entropy is borrowed from **information theory** (Shannon Entropy). It measures the average amount of information (surprise) in a random variable.

For a node with classes $\{c_1, c_2, ..., c_C\}$ and proportions $\{p_1, p_2, ..., p_C\}$:

$$H(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

**Intuition:**
- If all samples are class A → $p_A = 1$, $p_B = 0$ → $H = -(1 \cdot \log_2 1) = 0$ (no surprise, perfect purity)
- If half class A, half class B → $p_A = 0.5$, $p_B = 0.5$ → $H = -(0.5 \cdot \log_2 0.5 + 0.5 \cdot \log_2 0.5) = 1$ (maximum disorder)

### Gini Impurity Derivation

$$Gini(S) = 1 - \sum_{i=1}^{C} p_i^2$$

**Intuition:**
- $\sum p_i^2$ = probability of picking two samples of the same class
- Gini = probability of misclassifying a randomly chosen sample
- Gini = 0 → pure node; Gini = 0.5 (binary) → maximum impurity

### Gradient
Decision Trees do not use gradient descent. They are greedy, rule-based algorithms. The "optimization" at each node is an exhaustive search over all features and all possible split thresholds.

---

## 7. Optimization Method

Decision Trees use a **Greedy Algorithm** — not gradient descent.

### Algorithm: CART (Classification and Regression Trees)

At each node:
1. Loop over every feature $f$
2. Loop over every unique value $t$ in feature $f$ as a threshold
3. Split the data: Left = samples where $f \leq t$, Right = samples where $f > t$
4. Compute the weighted impurity of the split:

$$Cost(split) = \frac{|Left|}{|S|} \cdot Gini(Left) + \frac{|Right|}{|S|} \cdot Gini(Right)$$

5. Choose the feature-threshold pair that minimizes $Cost(split)$
6. Recurse on each child node

### Stopping Criteria (to avoid infinite splitting)
- Max depth reached
- Minimum samples per node
- Node is already pure (impurity = 0)
- No further information gain

---

## 8. Training Workflow

```
Input Labeled Data (features X, labels y)
        ↓
Start at Root Node (entire dataset)
        ↓
For each node:
    - Find the best feature & threshold to split on
      (minimize Gini or maximize Information Gain)
        ↓
    Split data into Left child and Right child
        ↓
    Recurse on each child
        ↓
    Stop when: max_depth reached, node is pure,
               or min_samples_split not met
        ↓
Assign class label to each leaf node
(majority class of samples in that leaf)
        ↓
Tree is trained → ready to predict
```

---

## 9. Worked Numerical Example

### Dataset

| Sample | Feature: Hours Studied | Feature: Attendance % | Label |
|---|---|---|---|
| 1 | 2 | 60 | Fail |
| 2 | 5 | 80 | Pass |
| 3 | 6 | 85 | Pass |
| 4 | 1 | 50 | Fail |

**Goal:** Split on "Hours Studied" with threshold = 3

**Left node** (Hours ≤ 3): Samples 1, 4 → [Fail, Fail]  
**Right node** (Hours > 3): Samples 2, 3 → [Pass, Pass]

### Compute Gini for each node

**Left node** (2 Fails, 0 Pass):
$$p_{Fail} = 1.0, \quad p_{Pass} = 0.0$$
$$Gini(Left) = 1 - (1.0^2 + 0.0^2) = 0$$

**Right node** (0 Fails, 2 Pass):
$$p_{Fail} = 0.0, \quad p_{Pass} = 1.0$$
$$Gini(Right) = 1 - (0.0^2 + 1.0^2) = 0$$

### Weighted Gini of Split

$$Gini(split) = \frac{2}{4} \cdot 0 + \frac{2}{4} \cdot 0 = 0$$

**Result:** Perfect split! This threshold perfectly separates the two classes. Both leaf nodes are pure → this becomes the final tree.

### Prediction for new sample: Hours = 4
- 4 > 3 → Go to Right node → **Predict: Pass ✓**

---

## 10. Full Manual Training Example

### Dataset (5 rows)

| ID | X1: Age | X2: Income | Label |
|---|---|---|---|
| 1 | 25 | Low | No |
| 2 | 45 | High | Yes |
| 3 | 35 | Medium | No |
| 4 | 50 | High | Yes |
| 5 | 23 | Low | No |

**Total: 3 "No", 2 "Yes"**

### Step 1: Compute Parent Entropy

$$p_{No} = 3/5 = 0.6, \quad p_{Yes} = 2/5 = 0.4$$
$$Entropy(Parent) = -(0.6 \log_2 0.6 + 0.4 \log_2 0.4)$$
$$= -(0.6 \times (-0.737) + 0.4 \times (-1.322))$$
$$= -(−0.442 − 0.529) = 0.971$$

### Step 2: Try Split on Income

**Income = Low:** IDs 1, 5 → [No, No] → Pure → Entropy = 0  
**Income = Medium:** ID 3 → [No] → Pure → Entropy = 0  
**Income = High:** IDs 2, 4 → [Yes, Yes] → Pure → Entropy = 0  

$$IG(Income) = 0.971 - \left(\frac{2}{5}(0) + \frac{1}{5}(0) + \frac{2}{5}(0)\right) = 0.971$$

### Step 3: Try Split on Age (threshold = 35)

**Age ≤ 35:** IDs 1, 3, 5 → [No, No, No] → Pure → Entropy = 0  
**Age > 35:** IDs 2, 4 → [Yes, Yes] → Pure → Entropy = 0  

$$IG(Age \leq 35) = 0.971 - 0 = 0.971$$

### Step 4: Both features give perfect split → Tree has depth = 1

**Final Rule:**  
- If Age ≤ 35 → **No**  
- If Age > 35 → **Yes**

---

## 11. Parameters (Learned Values)

| Parameter | Meaning |
|---|---|
| Split Feature | Which feature is used at each internal node |
| Split Threshold | The value that divides left and right branches |
| Leaf Class Label | The majority class label assigned to each leaf |
| Node Impurity | Gini or Entropy value stored at each node |
| Node Sample Count | Number of training samples reaching each node |

These are **learned** automatically during training. You do not set them manually.

---

## 12. Hyperparameters

| Hyperparameter | Default | Purpose |
|---|---|---|
| `max_depth` | None | Maximum depth of the tree |
| `min_samples_split` | 2 | Minimum samples needed to split a node |
| `min_samples_leaf` | 1 | Minimum samples required in a leaf node |
| `max_features` | None | Number of features to consider at each split |
| `criterion` | "gini" | Impurity measure: "gini" or "entropy" |
| `max_leaf_nodes` | None | Limit on total number of leaf nodes |
| `min_impurity_decrease` | 0.0 | Minimum gain required to make a split |
| `class_weight` | None | Adjust class importance for imbalanced data |

---

## 13. Why Each Hyperparameter Exists

### `max_depth`
**Problem:** An unconstrained tree grows until every leaf is pure — this means it memorizes the training data (overfitting).  
**Solution:** `max_depth` limits how deep the tree can go. A shallow tree generalizes better but may underfit.  
**Rule of thumb:** Start with `max_depth=3` to `5` and tune from there.

### `min_samples_split`
**Problem:** A node with just 2–3 samples can still be split, creating very specific rules that don't generalize.  
**Solution:** Only split a node if it has at least `min_samples_split` samples.

### `min_samples_leaf`
**Problem:** A leaf with 1 sample is essentially a memorized data point.  
**Solution:** Each leaf must contain at least `min_samples_leaf` samples to be valid.

### `criterion`
**Problem:** Two valid measures of impurity exist — Gini and Entropy.  
**Solution:** Both work well. Entropy is slightly more mathematically grounded; Gini is faster to compute. Difference in performance is usually negligible.

### `class_weight`
**Problem:** With imbalanced classes (e.g., 95% class 0, 5% class 1), the model ignores the minority class.  
**Solution:** `class_weight='balanced'` assigns higher importance to the minority class during training.

---

## 14. Assumptions

Decision Trees make **very few assumptions** compared to parametric models:

1. **No linearity assumption:** Can capture any non-linear boundary.
2. **No normality assumption:** Features don't need to follow a normal distribution.
3. **No independence assumption:** Handles correlated features (though redundant features reduce efficiency).
4. **Features can be mixed types:** Both numerical and categorical.
5. **Implicit assumption:** The training data is representative of the true underlying distribution.

---

## 15. Data Requirements

| Requirement | Needed? | Notes |
|---|---|---|
| Feature Scaling (Normalization) | ❌ No | Splits are threshold-based; scale doesn't matter |
| Encoding Categorical Features | ✅ Yes | scikit-learn requires numerical input |
| Handling Missing Values | ✅ Yes | Must impute before training in scikit-learn |
| Large Dataset | Recommended | Small datasets → high variance trees |
| Balanced Classes | Recommended | Use `class_weight` for imbalance |
| Outlier Removal | ❌ Not required | Trees are robust to outliers |

---

## 16. Complexity

### Training
$$O(n \cdot m \cdot \log n)$$

Where $n$ = number of samples, $m$ = number of features.  
At each node, we sort $n$ samples for each of $m$ features, and there are $O(\log n)$ levels in a balanced tree.

### Prediction
$$O(depth)$$

Following one path from root to leaf. For a balanced tree with max depth $d$: $O(d) = O(\log n)$.

### Memory
$$O(n \cdot m)$$

Storing the full training data plus the tree structure. Worst case (fully grown): $O(n)$ leaf nodes.

---

## 17. Decision Boundary

Decision Trees create **axis-aligned, rectangular decision boundaries**.

```
Feature 2 (e.g., Income)
    |
  H |  [No] [No] | [Yes] [Yes]
  i |             |
  g |  [No] [No] | [Yes] [Yes]
  h |_____________|___________
    Low         High        Feature 1 (e.g., Age)
```

**Key characteristics:**
- **Always linear (axis-aligned):** Boundaries are always horizontal or vertical — parallel to the feature axes.
- **Non-linear overall:** The combination of many axis-aligned splits creates complex, non-linear boundaries.
- **Cannot capture diagonal boundaries natively** — requires many splits to approximate a diagonal.

---

## 18. Overfitting & Underfitting

### Overfitting (most common problem)
A fully grown tree with no depth limit memorizes every training sample.

**Signs:**
- Training accuracy: 100%
- Test accuracy: 60–70% (much lower)
- Very deep tree with many small leaf nodes

**Cause:** The tree creates extremely specific rules for noise in the training data.

### Underfitting
A tree that is too shallow fails to capture the true complexity of the data.

**Signs:**
- Both training and test accuracy are low
- Tree has depth = 1 or 2

**Fix:** Increase `max_depth`, decrease `min_samples_split`.

### How to Balance:
- Use cross-validation to find the right `max_depth`
- Use pruning (pre-pruning via hyperparameters, or post-pruning)
- Use ensemble methods (Random Forest) which average many trees to reduce variance

---

## 19. Regularization

Decision Trees are regularized via **structural constraints** (not L1/L2 penalties):

### Pre-Pruning (Early Stopping)
Stop the tree from growing too large during training:
- `max_depth`: Limit depth
- `min_samples_split`: Require minimum samples to split
- `min_samples_leaf`: Require minimum samples in leaves
- `min_impurity_decrease`: Only split if impurity decreases by at least this amount

### Post-Pruning
After building a fully grown tree, remove branches that don't improve validation performance:
- **Cost Complexity Pruning (ccp_alpha in scikit-learn):**

$$TreeScore = Impurity + \alpha \cdot |leaves|$$

Higher `ccp_alpha` → more aggressive pruning → simpler tree.

### L1 / L2
Not directly applicable to Decision Trees. These regularization techniques belong to linear models.

---

## 20. Feature Importance

Decision Trees provide built-in **feature importance** scores:

$$Importance(f) = \sum_{nodes\ that\ split\ on\ f} \frac{n_{node}}{n_{total}} \cdot \Delta Impurity$$

Where $\Delta Impurity$ is the reduction in Gini/Entropy achieved by that split.

**Interpretation:**
- Higher score = feature was more useful for splitting
- Scores sum to 1.0

**Access in scikit-learn:**
```python
model.feature_importances_
```

**Limitations:**
- Biased toward high-cardinality features (features with many unique values)
- Correlated features may split importance between them

---

## 21. Advantages

- **Highly interpretable:** Can be visualized and explained as rules
- **No feature scaling required:** Works with raw numerical and categorical data
- **Handles non-linear relationships:** Can model complex boundaries
- **Fast prediction:** O(log n) per sample
- **Handles mixed feature types:** Numerical and categorical
- **Robust to outliers:** Threshold-based splits minimize outlier impact
- **No distributional assumptions:** Non-parametric
- **Automatic feature selection:** Unimportant features simply don't get split on
- **Handles multi-class problems natively:** No need for one-vs-rest wrappers

---

## 22. Limitations

- **High variance (unstable):** Small changes in data can produce very different trees
- **Prone to overfitting:** Especially without depth constraints
- **Biased toward features with more levels:** High-cardinality categorical features get inflated importance
- **Cannot capture diagonal boundaries:** Axis-aligned splits only
- **Greedy algorithm:** Local optima — the best split at each node is not guaranteed to be globally optimal
- **Imbalanced classes:** Biased toward majority class without `class_weight`
- **Poor extrapolation:** Cannot predict beyond the range of training data values

---

## 23. Failure Cases

| Scenario | Why It Fails |
|---|---|
| XOR problem (non-axis-aligned) | Requires 4 splits to separate; inefficient |
| Linear relationship | Requires many splits to approximate a line |
| Highly imbalanced dataset | Predicts majority class for everything |
| Noisy dataset | Overfits to noise without pruning |
| Regression-like continuous labels | Better handled by Decision Tree Regressor |
| Small dataset with high feature count | Overfits easily; needs strong pruning |

---

## 24. Edge Cases

| Situation | Behavior |
|---|---|
| **Tiny dataset (< 20 samples)** | Very high variance; tree memorizes all points; use `max_depth=2` |
| **Huge dataset (millions of rows)** | Training is slow; use `max_features` to limit features per split |
| **Features > Samples** | Extreme overfitting; must heavily prune or use `max_features` |
| **Imbalanced classes (99:1)** | Predicts majority class; use `class_weight='balanced'` |
| **Missing values** | scikit-learn raises error; must impute first |
| **Duplicate samples** | Tree handles normally; duplicates appear in leaf counts |
| **All features identical** | Cannot split; tree defaults to predicting majority class |
| **Single class in training data** | Tree returns that class for all predictions |

---

## 25. Evaluation Metrics

### Classification Metrics

| Metric | Formula | When to Use |
|---|---|---|
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | Balanced classes |
| **Precision** | $\frac{TP}{TP+FP}$ | Cost of false positives is high (spam detection) |
| **Recall** | $\frac{TP}{TP+FN}$ | Cost of false negatives is high (disease detection) |
| **F1 Score** | $2 \cdot \frac{Precision \times Recall}{Precision + Recall}$ | Imbalanced classes |
| **ROC-AUC** | Area under ROC curve | Probability-based ranking quality |
| **Confusion Matrix** | Table of TP/TN/FP/FN | Full error breakdown |
| **Log Loss** | $-\frac{1}{n}\sum y_i \log(\hat{p}_i)$ | When probabilities matter |

---

## 26. Comparison with Similar Models

| Feature | Decision Tree | Random Forest | Logistic Regression | SVM |
|---|---|---|---|---|
| Interpretability | ✅ Very High | ❌ Low | ✅ High | ❌ Low |
| Training Speed | ✅ Fast | ❌ Slower | ✅ Fast | ❌ Slow (large data) |
| Overfitting Risk | ❌ High | ✅ Low (ensembled) | ✅ Low | ✅ Low |
| Non-linear boundaries | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes (kernels) |
| Feature Scaling Needed | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| Handles Mixed Features | ✅ Yes | ✅ Yes | ⚠️ Partial | ⚠️ Partial |
| Output Probabilities | ✅ Yes (leaf freq) | ✅ Yes (averaged) | ✅ Yes (sigmoid) | ⚠️ Via Platt scaling |

---

## 27. Real-World Applications

| Domain | Application |
|---|---|
| **Healthcare** | Diagnosing diseases based on symptoms and lab results |
| **Finance** | Credit scoring, loan approval decisions |
| **Marketing** | Customer churn prediction, campaign targeting |
| **Manufacturing** | Fault detection and quality control |
| **Security** | Intrusion detection, fraud detection |
| **HR** | Employee attrition prediction |
| **E-commerce** | Product recommendation, customer segmentation |
| **Legal** | Risk assessment in case outcomes |

---

## 28. Scikit-Learn Implementation

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ----------------------------
# 1. Prepare Data
# ----------------------------
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 2. Train Model
# ----------------------------
model = DecisionTreeClassifier(
    criterion='gini',           # or 'entropy'
    max_depth=5,                # Limit depth to prevent overfitting
    min_samples_split=10,       # Min samples to split a node
    min_samples_leaf=5,         # Min samples in a leaf
    class_weight='balanced',    # Handle class imbalance
    random_state=42
)
model.fit(X_train, y_train)

# ----------------------------
# 3. Predict
# ----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ----------------------------
# 4. Evaluate
# ----------------------------
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# 5. Feature Importance
# ----------------------------
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("\nTop 5 Features:")
for i in range(5):
    print(f"  {feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")

# ----------------------------
# 6. Visualize Tree
# ----------------------------
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names,
          class_names=data.target_names, filled=True, max_depth=3)
plt.title("Decision Tree (max_depth=3 shown)")
plt.savefig("decision_tree_classifier.png", dpi=150, bbox_inches='tight')
plt.show()

# Print rules as text
rules = export_text(model, feature_names=list(feature_names))
print("\nDecision Rules (first few lines):")
print('\n'.join(rules.split('\n')[:30]))

# ----------------------------
# 7. Hyperparameter Tuning
# ----------------------------
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# ----------------------------
# 8. Cost-Complexity Pruning
# ----------------------------
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Train trees with different alpha values
train_scores, test_scores = [], []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

# Find optimal alpha
optimal_alpha = ccp_alphas[np.argmax(test_scores)]
print(f"\nOptimal ccp_alpha: {optimal_alpha:.6f}")
```

---

## 29. Interview Questions

### Basic
1. **What is a Decision Tree?**  
   A tree-like model that makes decisions by splitting data based on feature thresholds, assigning a class label at each leaf node.

2. **What is Gini Impurity vs Entropy?**  
   Both measure node impurity. Gini measures the probability of misclassification ($1 - \sum p_i^2$). Entropy measures information disorder ($-\sum p_i \log_2 p_i$). Gini is faster; Entropy is slightly more nuanced. In practice, the difference is minimal.

3. **What is Information Gain?**  
   The reduction in entropy (or impurity) after a split. Higher IG = better split.

4. **What are the assumptions of a Decision Tree?**  
   Almost none — non-parametric, no distribution assumption, no linearity assumption. Only assumes the training data is representative.

### Intermediate
5. **Why do Decision Trees overfit?**  
   A fully grown tree creates rules for every single training sample. It memorizes noise. Solution: pruning, `max_depth`, `min_samples_leaf`.

6. **How does a Decision Tree handle imbalanced classes?**  
   It tends to predict the majority class. Use `class_weight='balanced'` or oversample the minority class.

7. **What is pruning?**  
   Reducing tree complexity by removing branches that don't improve test performance. Pre-pruning: stop early. Post-pruning: grow full, then trim.

8. **What is the time complexity of training?**  
   $O(n \cdot m \cdot \log n)$ — polynomial in the number of samples and features.

9. **How are feature importances calculated?**  
   Weighted average of the impurity decrease across all nodes that use that feature.

### Advanced
10. **Why is Decision Tree called a greedy algorithm?**  
    At each node, it picks the locally best split without considering future nodes. This means the overall tree may not be globally optimal.

11. **Can a Decision Tree solve the XOR problem?**  
    Yes, but it requires multiple splits. It's not efficient — a single split can't solve XOR. A 2-feature XOR requires at least 3 nodes.

12. **What happens when features > samples?**  
    The tree can perfectly overfit with very specific rules. Requires heavy pruning or `max_features` restriction.

13. **How does `ccp_alpha` work in post-pruning?**  
    It adds a penalty for the number of leaf nodes: $Score = Impurity + \alpha \cdot n_{leaves}$. Higher alpha → fewer leaves → simpler tree. Tune via cross-validation.

14. **Decision Tree vs Random Forest — when to use each?**  
    Decision Tree: when interpretability is critical, small dataset, fast training needed.  
    Random Forest: when accuracy matters more than interpretability, overfitting is a concern.

15. **What is the difference between `min_samples_split` and `min_samples_leaf`?**  
    `min_samples_split`: minimum samples a node must have to be split further.  
    `min_samples_leaf`: minimum samples that must remain in any resulting leaf node after a split. `min_samples_leaf` is generally more powerful for controlling tree size.
